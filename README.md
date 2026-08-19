# EndoRare

[![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.13%2B-ee4c2c.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-11.7%2B-76b900.svg)](https://developer.nvidia.com/cuda-toolkit)

**EndoRare** is a research codebase for **generating rare endoscopic lesion images** from a
handful of real examples. Rare findings are, by definition, under-represented in endoscopy
datasets, which makes downstream detection and classification models brittle. EndoRare
addresses this by learning a compact textual description of a lesion and then re-synthesising
it with a latent diffusion model.

The pipeline has three stages:

1. **Three-axes attribute extraction** — textual inversion learns placeholder tokens that are
   split into three groups, so a lesion is described along separate semantic axes rather than
   as one entangled embedding.
2. **PSE** — for each real frame, a two-branch procedure either optimises the text embedding
   of a `V* polyp` prompt against the frame, or fine-tunes the diffusion model itself, and
   then samples new images from each branch.
3. **Evaluation** — per-class fidelity and diversity metrics comparing the real and generated
   sets.
---

## Table of Contents

- [Repository Layout](#repository-layout)
- [Requirements](#requirements)
- [Installation](#installation)
- [Data](#data)
- [Usage](#usage)
  - [1. Three-axes attribute extraction](#1-three-axes-attribute-extraction)
  - [2. PSE generation](#2-pse-generation)
  - [3. Evaluation](#3-evaluation)
- [Known Limitations](#known-limitations)
- [Acknowledgements](#acknowledgements)
- [License](#license)

---

## Repository Layout

```text
EndoRare/
├── endorare_pse.py                    # PSE: embedding optimisation + model fine-tuning, then sampling
├── data/                              # local dataset root (see data/README.md)
│   ├── README.md
│   └── example.png
├── evaluation/
│   ├── eval.py                        # full metric suite (FID, IS, MMD, KID, CLIP, DINO, IC-LPIPS)
│   ├── eval_class.py                  # class-conditional evaluation variant
│   └── eval_oneshot.py                # one-shot / few-shot evaluation variant
├── langint-three-axes-extraction/     # main three-axes extraction variant
│   ├── scripts/
│   │   ├── train_clip_inversion.py    # training entry point
│   │   └── extract_clip_inversion.py  # extraction entry point
│   ├── langint/                       # datasets, losses, models, trainers
│   ├── polypdiffusion/                # polyp diffusion sampling pipeline
│   └── tu/                            # training utilities (config, DDP, trainer loop)
├── langit/                            # earlier working copy of the same stack, kept for
│                                      # reproducing the ablations it was used for
└── requirements.txt
```

`langint-three-axes-extraction/` is the variant used for the three-axes results — it adds
`langint/trainers/invert_location.py`, `langint/trainers/attention.py` and
`scripts/extract_clip_inversion.py`. `langit/` is the earlier snapshot and additionally
carries a standalone `ldm/` copy plus assorted probing scripts. The two trees are independent;
pick one and put it on `PYTHONPATH`.

---

## Requirements

### Hardware

| Item | Specification |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4090 (24 GB) — used for all reported runs |
| VRAM | >= 24 GB recommended for diffusion fine-tuning |
| CUDA | 11.7 or higher |

### Software

- Python 3.8+
- PyTorch + torchvision (matching your CUDA build)
- PyTorch Lightning 1.x — the vendored LDM code is **not** compatible with Lightning 2.x
- Latent Diffusion (`ldm`) and taming-transformers
- OpenAI CLIP, DeepFloyd IF (bundled under `langint/third_party/deepfloyd/`)

---

## Installation

```bash
conda create -n endorare python=3.8 -y
conda activate endorare

# 1. PyTorch matching your CUDA version — see https://pytorch.org
pip install torch torchvision

# 2. Everything else
pip install -r requirements.txt

# 3. Packages that are not on PyPI under these names
pip install git+https://github.com/openai/CLIP.git
pip install git+https://github.com/CompVis/taming-transformers.git

# 4. Put the vendored packages on the import path
cd langint-three-axes-extraction
pip install -e .
export PYTHONPATH="$PWD:$PYTHONPATH"
```

See the comment block at the bottom of [`requirements.txt`](requirements.txt) for the optional
extras (`faiss`, `umap-learn`, `hdbscan`, `pydensecrf`) that only individual analysis scripts
need.

---

## Data

All images are resized to **256 x 256** pixels. The evaluation scripts expect the real and
generated roots to contain the *same* class sub-directories, since every metric is computed
per class and then averaged.

See [`data/README.md`](data/README.md) for the expected directory layout.

---

## Usage

### 1. Three-axes attribute extraction

Trains the placeholder-token embeddings. The entry point is driven by a YAML config; `-c`
accepts either a path to a `.yaml` file or a bare name that is resolved as
`configs/<name>.yaml` relative to the working directory. Any trailing `key=value` pairs
override fields in the config.

```bash
cd langint-three-axes-extraction

python scripts/train_clip_inversion.py \
    -c configs/train_deepfloyd_inversion.yaml \
    -d /path/to/data/real \
    -embs /path/to/opt_embs \
    -t polyp_7 \
    training.optimizers.embeddings.kwargs.lr=0.0002 \
    shared_tokens=0 \
    gt_init=0 \
    fruit_blip_coeff=0.00001 \
    mat_blip_coeff=0.00001 \
    color_blip_coeff=0.00001 \
    blip_guidance=0 \
    num_placeholder_groups=3 \
    num_placeholder_words=215
```

| Flag | Meaning |
| --- | --- |
| `-c`, `--config` | Config name or path to a config YAML (**required**) |
| `-d`, `--dataset` | Dataset directory (**required**) |
| `-embs`, `--opt_embs_file_path` | Path to the optimisation embeddings file (**required**) |
| `-t`, `--tag` | Tag appended to the output directory |
| `-s`, `--seed` | Random seed (default `0`) |
| `--log-unique` | Append a timestamp to the logging directory |

Training progress is written to TensorBoard under the config's `log_dir`. To extract the
learned embeddings afterwards, run `scripts/extract_clip_inversion.py` with the same flags.

### 2. PSE generation

For every image in `--real_data_dir`, `endorare_pse.py` runs two branches and writes samples
for each:

- **`V_only/`** — stage 1 optimises the text embedding of the prompt `V* polyp` to reconstruct
  the frame, then samples 100 images from the frozen model with that embedding. The optimised
  embedding is also saved to `opt_embs/<image>_emb_opt.pt`.
- **`finetune_only/`** — stage 2 fine-tunes a copy of the diffusion model against the frame's
  latent while keeping the original embedding, then samples from the fine-tuned model.

```bash
python endorare_pse.py \
    --config /path/to/stable-diffusion.yaml \
    --ckpt /path/to/checkpoint.ckpt \
    --real_data_dir data/real \
    --fake_data_dir outputs/generated \
    --seed 0 \
    --stage1_lr 1e-3 --stage1_num_iter 1000 \
    --stage2_lr 1e-6 --stage2_num_iter 1000
```

| Flag | Default | Meaning |
| --- | --- | --- |
| `--config` | *(hard-coded path)* | LDM / Stable Diffusion model config YAML |
| `--ckpt` | *(hard-coded path)* | Model checkpoint to load |
| `--real_data_dir` | *(hard-coded path)* | Folder of real input frames to invert |
| `--fake_data_dir` | *(hard-coded path)* | Output root for generated images and embeddings |
| `--seed` | `0` | Global seed; also sets cuDNN deterministic mode |
| `--stage1_lr` / `--stage1_num_iter` | `1e-3` / `1000` | Embedding-optimisation schedule |
| `--stage2_lr` / `--stage2_num_iter` | `1e-6` / `1000` | Model fine-tuning schedule |

### 3. Evaluation

```bash
python evaluation/eval.py \
    --real_folder data/real \
    --generated_folder outputs/generated/V_only \
    --output_path results/metrics.json
```

Results are written as JSON with the following keys:

| Metric | Key | Direction |
| --- | --- | --- |
| Fréchet Inception Distance | `FID` | lower is better |
| Inception Score | `Inception Score (IS)` (`mean`, `std`) | higher is better |
| Maximum Mean Discrepancy | `MMD` | lower is better |
| Kernel Inception Distance | `KID` | lower is better |
| CLIP image similarity | `CLIP Similarity` | higher is better |
| DINO feature alignment | `DINO Alignment` | higher is better |
| Intra-cluster LPIPS (diversity) | `IC-LPIPS` | higher is better |

`eval_class.py` and `eval_oneshot.py` take the same `--real_folder`, `--generated_folder` and
`--output_path` arguments for the class-conditional and one-shot settings.

---


## Acknowledgements

This codebase is based on and adapted from the open-source repository
[sharonal10/langint](https://github.com/sharonal10/langint). It further builds on
[Latent Diffusion](https://github.com/CompVis/latent-diffusion),
[taming-transformers](https://github.com/CompVis/taming-transformers),
[DeepFloyd IF](https://github.com/deep-floyd/IF) and
[OpenAI CLIP](https://github.com/openai/CLIP). We thank all of their authors for releasing
their work.

## License

No license has been declared for this repository yet. The upstream
[langint](https://github.com/sharonal10/langint) project it derives from also ships without a
license file, so no redistribution terms can be inherited from it. Until a license is added,
all rights are reserved by the authors — please open an issue if you would like to use this
code, and check the licenses of the third-party components listed above before redistributing
any part of this repository.

