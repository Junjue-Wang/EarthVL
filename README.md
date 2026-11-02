<h2 align="center">EarthVL: A Progressive Earth Vision-Language Understanding and Generation Framework</h2>

<h5 align="right">by <a href="https://junjue-wang.github.io/homepage/">Junjue Wang</a>,
<a href="http://rsidea.whu.edu.cn/">Yanfei Zhong</a>,
Zihang Chen,
<a href="http://zhuozheng.top/">Zhuo Zheng</a>,
 Ailong Ma, and <a href="http://www.lmars.whu.edu.cn/prof_web/zhangliangpei/rs/index.html">Liangpei Zhang</a></h5>

[[`Paper`](https://arxiv.org/pdf/2601.02783)],
[[`Dataset`](https://huggingface.co/datasets/Kingdrone-Junjue/EarthVLSet)]


<div align="center">
  <img src="./src/dataset_vis.png"/>
</div>

## News
- 2026/11/02, The code of EarthVLNet (land-cover segmentation, VQA training and evaluation) is released.
- 2026/11/01, EarthVL has been accepted by [<b>IEEE TPAMI</b>](https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=34) !!!
- 2026/01/06, New Global-LoveDA !!! We released the global-scale segmentation leaderboard at [<b>Global-LoveDA</b>](https://www.codabench.org/competitions/13040). Just zip all the test images into one file and submit it.
- 2026/01/06, The segmentation data is released at [[`Dataset`](https://huggingface.co/datasets/Kingdrone-Junjue/EarthVLSet)].
- 2026/01/06, We are preparing the code and data for EarthVL.
- 2024/09/25, EarthVL is the extension of our [<b>EarthVQA</b>](https://github.com/Junjue-Wang/EarthVQA) and [<b>LoveDA</b>](https://github.com/Junjue-Wang/LoveDA) projects.

## Introduction

Earth vision has achieved many milestones in geospatial object recognition, but the relations between objects, which
a comprehensive understanding of remote sensing scenes needs, remain little explored. EarthVL is a progressive Earth
vision-language understanding and generation framework, focused on city-planning applications:

- **EarthVLSet** is a multi-task dataset with 10.9k sub-meter resolution remote sensing images, land-cover masks and
  761.5k textual pairs for multiple-choice and open-ended visual question answering (VQA). It consists of EarthVQA
  (6,000 images in China) and EarthVL-GLOBAL (4,950 images of areas around the world).
- **EarthVLNet** answers the questions in an object-centric way, in two stages. Stage 1 segments the land cover with
  Semantic-FPN (ConvNeXt-L) to obtain object semantics. Stage 2 answers from the stage-1 features and predicted masks
  with two branches: an object awareness-based LLM branch (object-guided attention, a multi-modal projector and
  Vicuna-7B fine-tuned with LoRA) for the judging and analysis questions, and a counting branch, trained with the
  numerical difference loss, for the counting questions.

## Repository layout

```text
EarthVL
├── configs
│   ├── seg/sfpn_convnext_large.py  # stage 1: Semantic-FPN (ConvNeXt-L) land-cover segmentation
│   ├── EarthVLNet.py               # stage 2, LLM branch (Vicuna-7B + LoRA)
│   ├── EarthVLCount.py             # stage 2, counting branch
│   ├── EarthVL_cls.py              # data and schedule of the LLM branch
│   └── EarthVL_count.py            # data and schedule of the counting branch
├── data                # EarthVLSet loaders: multiple-choice VQA, open-ended VQA, segmentation
├── engine              # DeepSpeed training / evaluation engine
├── module              # EarthVLNet, losses, metrics, augmentations; module/seg: Semantic-FPN and ConvNeXt
├── score               # CIDEr and ROUGE-L for open-ended answers
├── scripts             # training, evaluation and feature-extraction scripts
├── src                 # figures of this README
├── tools               # prepare_vicuna.py, extract_seg_features.py
├── tests               # CPU smoke tests
├── train_seg.py        # stage 1: train / evaluate the segmentation network
├── train_generation.py # stage 2: train / evaluate the two VQA branches
└── evaluate.py         # merge the predictions of the two branches and compute accuracy / RMSE
```

The repository covers the multiple-choice VQA task. For the open-ended split it provides only the data loader
(`data/earthvl_open.py`) and the metrics (BLEU, ROUGE-L and CIDEr in `module/metric.py` and `score/`), not an
open-ended model.

## Installation

Python 3.10-3.12 and PyTorch 2.4.1. Training runs on CUDA GPUs with DeepSpeed; evaluation and the tests do not need
DeepSpeed.

```bash
conda create -n earthvl python=3.10 -y
conda activate earthvl
# PyTorch 2.4.1 for your CUDA version (https://pytorch.org/get-started/previous-versions/), e.g. CUDA 12.1:
pip install torch==2.4.1 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
# training (the released LLM-branch run used DeepSpeed 0.14.0)
pip install deepspeed==0.14.0
# optional
pip install flash-attn --no-build-isolation  # FlashAttention-2, the default attention of the LLM branch
pip install bitsandbytes                     # only for QLoRA (model.params.load_in_4bit True)
```

- `requirements.txt` pins the versions the tests ran with. Keep `albumentations==1.4.3`: the augmentations and configs
  rely on its behaviour.
- Config entries can be overridden with trailing `key value` pairs. `scripts/train_seg.sh`,
  `scripts/train_generation.sh`, `scripts/train_count.sh`, `scripts/eval_cls.sh` and `scripts/eval_count.sh` pass them
  on (the evaluation scripts after `CKPT` and `MODEL_DIR`), e.g.
  `bash scripts/train_generation.sh model.params.attn_implementation sdpa` without flash-attn.
- The LLM branch uses the fused Adam of DeepSpeed, which DeepSpeed compiles on first use (this needs the CUDA toolkit).
  If that is not possible, the override `deepspeed.optimizer.params.torch_adam True` uses `torch.optim.AdamW` instead.

## Data

EarthVLSet is a gated dataset on Hugging Face (access is granted automatically). Log in, accept the terms on the
[dataset page](https://huggingface.co/datasets/Kingdrone-Junjue/EarthVLSet), then download it (about 18 GB) to
`./EarthVLSet`:

```bash
huggingface-cli login
huggingface-cli download Kingdrone-Junjue/EarthVLSet --repo-type dataset --local-dir ./EarthVLSet
```

```text
EarthVLSet
├── EarthVQA
│   ├── Train
│   │   ├── images_png        # 2,522 images
│   │   └── masks_png
│   ├── Val
│   │   ├── images_png        # 1,669 images
│   │   └── masks_png
│   ├── Test
│   │   └── images_png        # 1,809 images
│   ├── Train_QA.json
│   ├── Val_QA.json
│   └── Test_QA.json          # answers withheld
└── EarthVL-GLOBAL
    ├── Train
    │   ├── images_png        # 2,738 images, including Rotterdam (see below)
    │   └── masks_png
    ├── Val
    │   ├── images_png        # 672 images
    │   └── masks_png
    ├── Test
    │   └── images_png        # 1,540 images
    ├── earthvl_global_multiplechoice
    │   ├── Train.json
    │   ├── Val.json
    │   └── Test.json
    └── earthvl_global_openended
        ├── Train.json
        ├── Val.json
        └── Test.json
```

- Land-cover labels: 1 background, 2 building, 3 road, 4 water, 5 barren, 6 forest, 7 agriculture, 8 playground;
  0 marks no-data regions and is ignored. Images are named by ids that are unique across both subsets (numeric, except
  13 EarthVQA Test images with area-based names such as `urban2.png`), and the QA files are keyed by image name (e.g.
  `"9118.png"`).
- **Test labels.** The Test masks are withheld; segmentation results on Test are evaluated on the
  [Global-LoveDA leaderboard](https://www.codabench.org/competitions/13040), so stage 1 reports mIoU on Val here. The
  answers of `EarthVQA/Test_QA.json` are withheld for the
  [EarthVQA VQA challenge](https://www.codabench.org/competitions/2922) on Codabench, while the EarthVL-GLOBAL
  `Test.json` contains the answers (see [Evaluation](#evaluation)).
- **Rotterdam.** The paper lists Rotterdam (Netherlands) among the Val areas, but in the Hugging Face release its 368
  images (ids 9118-9485, 20,842 multiple-choice QAs) are part of `EarthVL-GLOBAL/Train`. The VQA experiments of the
  paper did not train on them (227,030 training QAs), so both VQA configs exclude them (`ROTTERDAM_IMAGES`, passed to
  the loader option `exclude_images`). Stage 1 trains on them, like the authors' segmentation run.

## Pretrained models

- **Language model of the LLM branch.** `./pretrain/vicuna-7b` holds the Vicuna-7B language model of LLaVA-1.5 and the
  LLaVA processor, whose tokenizer has the `<image>` (32000) and `<pad>` (32001) tokens that `configs/EarthVLNet.py`
  expects. Build it from [llava-hf/llava-1.5-7b-hf](https://huggingface.co/llava-hf/llava-1.5-7b-hf) (about 14 GB to
  download):

  ```bash
  python tools/prepare_vicuna.py --src llava-hf/llava-1.5-7b-hf --dst ./pretrain/vicuna-7b
  ```

- **Tokenizer of the counting branch.** The tokenizer of [google/flan-t5-xl](https://huggingface.co/google/flan-t5-xl)
  is downloaded from the Hugging Face Hub on first use (tokenizer files only). For a local copy, pass
  `model.params.pretrained_model_dir <folder>` to the counting scripts.
- **ImageNet-22k ConvNeXt-L for stage 1.** The official checkpoint
  [convnext_large_22k_224.pth](https://dl.fbaipublicfiles.com/convnext/convnext_large_22k_224.pth) (about 0.9 GB) is
  downloaded into the torch hub cache when `scripts/train_seg.sh` starts. To use a local copy, or the same weights from
  the Hugging Face Hub (`hf-hub:timm/convnext_large.fb_in22k`), change `CONVNEXT_PRETRAINED` in
  `configs/seg/sfpn_convnext_large.py`.

## Training

Run all scripts from the repository root; they write to `./log`.

### Stage 1: land-cover segmentation and features

```bash
bash scripts/train_seg.sh          # Semantic-FPN (ConvNeXt-L), 2 GPUs x 8 images, 30k iterations
bash scripts/extract_features.sh   # stage-1 features and predicted masks of Train and Test (about 63 GB)
```

- `train_seg.sh` trains on `EarthVQA/Train` and `EarthVL-GLOBAL/Train`, saves DeepSpeed checkpoints as
  `./log/seg/sfpn_convnext_large/model-<step>.pth/` and reports mIoU on Val. `NUM_GPUS=4 bash scripts/train_seg.sh`
  keeps the global batch of 16. To evaluate a checkpoint on Val:

  ```bash
  python train_seg.py --config_path=seg.sfpn_convnext_large --model_dir=./log/seg/sfpn_convnext_large \
      --eval_only --ckpt ./log/seg/sfpn_convnext_large/model-30000.pth
  ```

- `extract_features.sh` reads the checkpoint given by `CKPT` (default `./log/seg/sfpn_convnext_large/model-30000.pth`)
  and writes `./features/sfpn_convnext_large/<EarthVQA|EarthVL-GLOBAL>/<Train|Test>/<image_id>.hdf5`, with the
  1536×32×32 feature map and the 1024×1024 predicted mask (about 7.3 MB per image). Finished files are skipped, so an
  interrupted run can be resumed; the script also shows how to split the work over several GPUs. The configs do not
  use Val features; if you need them (about 17 GB more), run
  `python tools/extract_seg_features.py --ckpt <checkpoint> --splits Val`.

### Stage 2: counting branch

```bash
bash scripts/train_count.sh        # 1 GPU, batch 32, 40k steps -> ./log/EarthVLCount/
```

### Stage 2: LLM branch

```bash
bash scripts/train_generation.sh   # 4 GPUs x micro batch 6, 40k steps, DeepSpeed ZeRO-2, bf16 -> ./log/EarthVLNet/
```

- As in the released run, the language model from `./pretrain/vicuna-7b` is loaded in bf16, frozen and adapted with
  LoRA (r=64, alpha=16).
- Without flash-attn: `bash scripts/train_generation.sh model.params.attn_implementation sdpa` (pass the same
  override to `scripts/eval_cls.sh`). For smaller GPUs, `model.params.load_in_4bit True` switches to QLoRA (4-bit NF4
  base, needs bitsandbytes); this deviates from the released run.

Both branches save DeepSpeed checkpoints as `model-<step>.pth/`, evaluate on the test set after training
(`prediction-40000.json`) and resume from the last checkpoint when restarted with the same model directory.

## Evaluation

```bash
bash scripts/eval_count.sh ./log/EarthVLCount/model-40000.pth  # -> ./log/EarthVLCount/prediction-40000.json
bash scripts/eval_cls.sh ./log/EarthVLNet/model-40000.pth      # -> ./log/EarthVLNet/prediction-40000.json
python evaluate.py --cls_pred ./log/EarthVLNet/prediction-40000.json \
    --count_pred ./log/EarthVLCount/prediction-40000.json
```

- `scripts/eval_cls.sh [CKPT] [MODEL_DIR] [key value ...]` and `scripts/eval_count.sh [CKPT] [MODEL_DIR] [key value ...]`
  take a DeepSpeed checkpoint directory, a run directory such as `./log/EarthVLNet` (its last checkpoint), a `.pth`
  state dict or a released weight file, and write `<MODEL_DIR>/prediction-<step>.json`, where the step comes from the
  checkpoint name. `NUM_GPUS=4 bash scripts/eval_cls.sh ...` evaluates on several GPUs.
- `evaluate.py` merges the two branches: the counting questions take the answer of the counting branch, the two
  traffic questions of Comprehensive Analysis take the LLM sentence with the counted number, and all other questions
  the LLM answer. It prints the accuracy of each question type and the overall accuracy (OA), the accuracy of each
  question, and the RMSE of the counting answers. `--save_merged <file>` writes the merged answers, and
  `bash scripts/test.sh` runs `evaluate.py` with the default paths above.
- Test QAs without an answer are predicted but not scored. As the EarthVQA Test answers are withheld, the local scores
  cover the EarthVL-GLOBAL Test QAs only, whereas the tables of the paper cover both test sets. Predictions for
  EarthVQA Test are scored by the [EarthVQA VQA challenge](https://www.codabench.org/competitions/2922).

## Released weights

The weights are published at
[Kingdrone-Junjue/EarthVL-pretrained](https://huggingface.co/Kingdrone-Junjue/EarthVL-pretrained):

| File | Size | Config | Content |
|---|---|---|---|
| `earthvlnet_llm_lora_40000.safetensors` | 368 MB | `configs/EarthVLNet.py` | Trainable parameters of the LLM branch only (462 bf16 tensors): LoRA (r=64, alpha=16), object-guided attention (OGA), multi-modal projector (MMP) and mask embedding. The frozen language model comes from `./pretrain/vicuna-7b`. |
| `earthvl_count_40000.pth` | 87 MB | `configs/EarthVLCount.py` | State dict of the counting branch (53 tensors). |

```bash
huggingface-cli download Kingdrone-Junjue/EarthVL-pretrained --local-dir ./pretrain/EarthVL-pretrained
bash scripts/eval_cls.sh ./pretrain/EarthVL-pretrained/earthvlnet_llm_lora_40000.safetensors
bash scripts/eval_count.sh ./pretrain/EarthVL-pretrained/earthvl_count_40000.pth
python evaluate.py
```

**Stage-1 checkpoint required.** Both files were trained on the features of the authors' stage-1 Semantic-FPN
(ConvNeXt-L) checkpoint, which is not available yet. A stage-1 network trained with this code produces features in a
different feature space, which the released weights cannot be used with; train the two VQA branches on your own
features instead. Without the authors' features, the released weights have not been evaluated with this code. The
LLM-branch weights also come from a later run (August 2024) than the model behind the LLM-branch results of the paper,
so their results can differ from the tables of the paper.

<!-- TODO(authors): link the stage-1 checkpoint (or its features) if it is published. -->

## Notes on reproduction

- The configs use the settings of the runs behind the released checkpoints. A few hyper-parameters differ from the
  summary in the paper:
  - LLM branch: DeepSpeed Adam in AdamW mode (lr 2e-4, betas (0.8, 0.999), weight decay 3e-7) with `WarmupCosineLR`;
    batch 24 (4 GPUs x 6), 40k steps, bf16, ZeRO-2.
  - Counting branch: AdamW (lr 1e-3, betas (0.9, 0.999), weight decay 0.2) with a cosine schedule and a 1,000-step
    linear warm-up; batch 32 on one GPU, 40k steps.
  - Stage 1: bf16 autocast instead of the fp16 automatic mixed precision of the authors' run.
  - QLoRA (`model.params.load_in_4bit True`) is optional; the released LLM branch was trained with LoRA on a bf16 base.
- The LLM branch is trained and tested on the six judging and analysis question types (`configs/EarthVL_cls.py`), as in
  the released run; the counting questions are answered by the counting branch.
- VQA training excludes Rotterdam (see [Data](#data)).
- As in the original code, the test loaders use the sampler of the training loaders, which pads the QA list to a
  multiple of the number of GPUs, and drop the last incomplete batch; a few test QAs can therefore be repeated in or
  missing from the prediction files.
- Checkpoints are read with `torch.load(weights_only=True)`. Files that need full unpickling, which can run code stored
  in the file, are only loaded with `EARTHVL_TRUST_CHECKPOINTS=1`; set it only for files you trust. The released
  weights do not need it.

## Tests

```bash
pip install pytest
python -m pytest tests -q
```

The tests run on the CPU with synthetic data, tiny models and a DeepSpeed stand-in; they need neither the dataset nor
a GPU, DeepSpeed or the 7B weights. With `EARTHVL_COUNT_WEIGHTS=<path to earthvl_count_40000.pth>` and
`EARTHVL_LLM_WEIGHTS=<path to earthvlnet_llm_lora_40000.safetensors>` they also check the released weights against
the models.

## License

- The code is licensed under [CC BY-NC-SA 4.0](LICENSE). Code adapted from other projects keeps its original license;
  see [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
- EarthVLSet is licensed under CC BY-NC-ND 4.0 (see the
  [dataset card](https://huggingface.co/datasets/Kingdrone-Junjue/EarthVLSet)).
- The released weights are licensed under CC BY-NC-SA 4.0. The pretrained models used with them come with their own
  licenses: the Vicuna-7B language model of LLaVA-1.5 is based on Llama 2 and governed by the Llama 2 Community
  License, and the FLAN-T5 tokenizer and the ConvNeXt-L weights have the licenses of their releases.

## Citation
If you use EarthVL in your research, please cite our following papers.
```text
    @article{wang2026earthvl,
      title={EarthVL: A Progressive Earth Vision-Language Understanding and Generation Framework},
      author={Wang, Junjue and Zhong, Yanfei and Chen, Zihang and Zheng, Zhuo and Ma, Ailong and Zhang, Liangpei},
      journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
      year={2026},
      note={to be published}
    }
    @article{wang2024earthvqa,
        title={EarthVQA: Towards Queryable Earth via Relational Reasoning-Based Remote Sensing Visual Question Answering},
        url={https://ojs.aaai.org/index.php/AAAI/article/view/28357},
        DOI={10.1609/aaai.v38i6.28357},
        journal={Proceedings of the AAAI Conference on Artificial Intelligence},
        author={Wang, Junjue and Zheng, Zhuo and Chen, Zihang and Ma, Ailong and Zhong, Yanfei},
        year={2024},
        month={Mar.},
        volume={38},
        number={6},
        pages={5481-5489}}
    @inproceedings{wang2021loveda,
         author = {Wang, Junjue and Zheng, Zhuo and Ma, Ailong and Lu, Xiaoyan and Zhong, Yanfei},
         booktitle = {Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks},
         editor = {J. Vanschoren and S. Yeung},
         pages = {},
         publisher = {Curran Associates, Inc.},
         title = {LoveDA: A Remote Sensing Land-Cover Dataset for Domain Adaptive Semantic Segmentation},
         url = {https://datasets-benchmarks-proceedings.neurips.cc/paper_files/paper/2021/file/4e732ced3463d06de0ca9a15b6153677-Paper-round2.pdf},
         volume = {1},
         year = {2021}
    }
```


## Star History

<a href="https://www.star-history.com/?repos=Junjue-Wang%2FEarthVL&type=date&legend=top-left">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/chart?repos=Junjue-Wang/EarthVL&type=date&theme=dark&legend=top-left" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/chart?repos=Junjue-Wang/EarthVL&type=date&legend=top-left" />
   <img alt="Star History Chart" src="https://api.star-history.com/chart?repos=Junjue-Wang/EarthVL&type=date&legend=top-left" />
 </picture>
</a>
