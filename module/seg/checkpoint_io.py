"""Read Semantic-FPN weights from the checkpoint formats used in this project.

Supported inputs:

* DeepSpeed checkpoints written by ``engine/trainer.py``: a tag directory such
  as ``log/seg/sfpn_convnext_large/model-30000.pth/`` (``mp_rank_00_model_states.pt``
  with the weights under ``'module'``), or the run directory with a ``latest`` file;
* the authors' original ever checkpoints (``{'model': state_dict, 'opt': ..., 'global_step': ...}``
  with ``module.``-prefixed keys);
* plain state dicts (``.pth``/``.pt``/``.bin``) and ``.safetensors`` files.

``torch.load`` runs with ``weights_only=True``. A file that needs full unpickling,
which can execute code stored in it, is loaded only when the environment variable
``EARTHVL_TRUST_CHECKPOINTS=1`` is set; otherwise loading fails with an explanation.
"""
import contextlib
import glob
import logging
import os
import pickle

import torch

logger = logging.getLogger(__name__)

_PREFIXES = ('module.', '_orig_mod.')

# Set to 1 to allow full unpickling (weights_only=False) of checkpoints that
# torch.load(weights_only=True) refuses. Only do this for files you trust.
TRUST_CHECKPOINTS_ENV = 'EARTHVL_TRUST_CHECKPOINTS'


@contextlib.contextmanager
def _allow_sets():
    """Let the weights_only unpickler build sets, which torch 2.4 refuses by default.

    DeepSpeed stores ``sparse_tensor_module_names`` as a set in every model-states file.
    The previous allowlist is restored afterwards.
    """
    serialization = torch.serialization
    if not hasattr(serialization, 'add_safe_globals'):  # torch < 2.4
        yield
        return
    saved = list(serialization.get_safe_globals())
    serialization.add_safe_globals([set])
    try:
        yield
    finally:
        serialization.clear_safe_globals()
        serialization.add_safe_globals(saved)


def safe_load(load, source):
    """Return ``load(weights_only=True)``; retry with ``weights_only=False`` only if EARTHVL_TRUST_CHECKPOINTS=1."""
    try:
        with _allow_sets():
            return load(weights_only=True)
    except pickle.UnpicklingError as e:
        if os.environ.get(TRUST_CHECKPOINTS_ENV, '0') != '1':
            raise pickle.UnpicklingError(
                f'{source} cannot be loaded with torch.load(weights_only=True): it stores Python objects other than '
                f'tensors and plain containers. Loading it with full unpickling can execute arbitrary code stored '
                f'in the file. If you trust the file, set {TRUST_CHECKPOINTS_ENV}=1 and run again.') from e
        logger.warning('%s=1: loading %s with full unpickling (weights_only=False); this can execute code stored '
                       'in the file.', TRUST_CHECKPOINTS_ENV, source)
        return load(weights_only=False)


def torch_load(path, map_location='cpu'):
    """``torch.load`` of a local file under the policy of :func:`safe_load`."""
    return safe_load(lambda weights_only: torch.load(path, map_location=map_location, weights_only=weights_only),
                     path)


def _deepspeed_model_file(path):
    latest = os.path.join(path, 'latest')
    if os.path.isfile(latest):
        with open(latest) as f:
            tag = f.read().strip()
        path = os.path.join(path, tag)
    model_file = os.path.join(path, 'mp_rank_00_model_states.pt')
    if os.path.isfile(model_file):
        return model_file
    candidates = sorted(glob.glob(os.path.join(path, '*model_states.pt')))
    if not candidates:
        raise FileNotFoundError(f'No DeepSpeed model states (mp_rank_00_model_states.pt) found in {path}')
    if any('zero_pp_rank' in os.path.basename(c) for c in candidates):
        raise RuntimeError(f'{path} looks like a ZeRO-3 checkpoint with partitioned weights; '
                           'consolidate it first with DeepSpeed\'s zero_to_fp32.py.')
    return candidates[0]


def strip_prefixes(state_dict, prefixes=_PREFIXES):
    out = {}
    for k, v in state_dict.items():
        changed = True
        while changed:
            changed = False
            for p in prefixes:
                if k.startswith(p):
                    k = k[len(p):]
                    changed = True
        out[k] = v
    return out


def load_state_dict_any(path, map_location='cpu'):
    """Return a flat ``{name: tensor}`` state dict without ``module.``/``_orig_mod.`` prefixes."""
    if os.path.isdir(path):
        path = _deepspeed_model_file(path)
    if path.endswith('.safetensors'):
        from safetensors.torch import load_file
        state = load_file(path, device=str(map_location))
    else:
        state = torch_load(path, map_location)
    if isinstance(state, dict):
        for key in ('module', 'model', 'state_dict'):
            if isinstance(state.get(key, None), dict):
                state = state[key]
                break
    if not isinstance(state, dict):
        raise TypeError(f'Unsupported checkpoint content in {path}: {type(state)}')
    state = {k: v for k, v in state.items() if isinstance(v, torch.Tensor)}
    return strip_prefixes(state)


def load_weights(model, path, strict=True):
    """Load ``path`` into ``model``; raise with a readable key diff when ``strict``."""
    state = load_state_dict_any(path)
    own = model.state_dict()
    missing = sorted(set(own) - set(state))
    unexpected = sorted(set(state) - set(own))
    mismatched = sorted(k for k in set(own) & set(state) if own[k].shape != state[k].shape)
    if strict and (missing or unexpected or mismatched):
        raise RuntimeError(
            f'Checkpoint {path} does not match the model: missing {len(missing)} {missing[:10]}, '
            f'unexpected {len(unexpected)} {unexpected[:10]}, shape mismatch {mismatched[:10]}')
    state = {k: v for k, v in state.items() if k in own and k not in mismatched}
    model.load_state_dict(state, strict=False)
    logger.info('Loaded %d tensors from %s (missing %d, unexpected %d, mismatched %d)',
                len(state), path, len(missing), len(unexpected), len(mismatched))
    return dict(missing=missing, unexpected=unexpected, mismatched=mismatched)
