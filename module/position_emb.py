"""Fixed sinusoidal positional encodings (Vaswani et al., "Attention Is All You Need", NeurIPS 2017)."""
import math

import torch


def _sinusoid_table(num_positions, dim, dtype):
    """[num_positions, dim] table with channel 2i = sin(p * w_i) and channel 2i + 1 = cos(p * w_i),
    where p is the position and w_i = exp(-2i * log(10000) / dim)."""
    freqs = torch.exp(torch.arange(0, dim, 2, dtype=dtype) * -(math.log(10000.0) / dim))
    angles = torch.arange(num_positions).to(dtype).unsqueeze(1) * freqs  # num_positions x dim / 2
    table = torch.zeros(num_positions, dim)
    table[:, 0::2] = torch.sin(angles)
    table[:, 1::2] = torch.cos(angles)
    return table


def positionalencoding1d(d_model, length):
    """Encoding of the positions 0 .. length - 1, a [length, d_model] tensor (d_model must be even)."""
    if d_model % 2 != 0:
        raise ValueError(f'Sinusoidal positional encoding needs an even dimension (got dim={d_model})')
    return _sinusoid_table(length, d_model, torch.float)
