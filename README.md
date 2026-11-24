# AMAL-Net

**Adaptive Multibranch Attention Learning Network** for medical lesion segmentation.

AMAL-Net is a two-stage, multibranch network for segmenting organs and lesions
(e.g. liver/pancreas/lung tumours) from medical images. A shared residual encoder
produces two complementary branches — a *foreground* decoder guided by **cluster
attention** and a *background* decoder guided by **density attention** — and a
second stage refines the result with **score-first patch routing**: the image is
split into a fixed grid of patches, a lightweight scorer selects the most
informative patches, and only those are refined at higher resolution before the
full-resolution prediction is reassembled and fused.

The model is lightweight (≈3–4 M parameters depending on configuration; see the
paper for the exact count of the common protocol).

> **Status:** the paper describing AMAL-Net is currently **under review**. The
> citation below is a placeholder — replace it with the accepted reference once
> available.

---

## Repository contents

This repository is a self-contained, **fair-protocol comparison harness**. It
trains and evaluates AMAL-Net alongside three baseline architectures under an
identical protocol, so all reported numbers are directly comparable.

| File | Purpose |
|---|---|
| `AMALNet_Main.py` | **AMAL-Net** model (self-contained, PyTorch-only): `AMALNet`, `ImprovedTwoStageSegmentation`, `AMALLoss`, `AMALConfig` |
| `COMANet_Model.py` | COMA-Net baseline (paper-based reproduction, no external weights) |
| `UCTransNet_Official.py` | UCTransNet baseline (wraps an explicit local checkout of the authors' code) |
| `nnUNet2D_Model.py` | nnU-Net 2D baseline (PlainConvUNet-style architecture under the same protocol) |
| `baselines_adapters.py` | self-contained local UCTransNet adapter (fallback when the official checkout is absent) |
| `common.py` | data pipeline, augmentation, metrics (Dice / IoU / HD95 / ASSD) |
| `models.py` | model factory, losses, forward/loss helpers |
| `train.py` | unified training loop |
| `evaluate.py` | test/val evaluation from a checkpoint |
| `run_one.sh` | train + evaluate a single (model, dataset, seed) |
| `run_matrix.py` | launch a TSV manifest across multiple GPUs |
| `summarize.py` | aggregate runs into mean ± std tables / CSV / Markdown |
| `check_env.py` | pre-flight environment checks |
| `model_checksums.txt` | SHA-256 of the four model files (provenance) |
| `manifests/` | `smoke.tsv`, `quick.tsv`, `main.tsv` (run manifests) |

The four model files under comparison are imported **unchanged**; their SHA-256
checksums are recorded in `model_checksums.txt`.

---

## Installation

Requirements: Python ≥ 3.8 and a recent PyTorch with CUDA.

```bash
pip install -r requirements.txt
```

The core pipeline only needs `torch`, `torchvision`, `numpy` and `Pillow`.
Optional routes (official nnU-Net pipeline, official UCTransNet checkout) may
need extra packages — see the notes in `requirements.txt` and the `--help` of
each entry point.

---

## Data preparation

The harness expects **patient-level 3-way splits** with this layout:

```
<data-root>/
├── liver_3way/
│   ├── train_images/   train_masks/
│   ├── val_images/     val_masks/
│   └── test_images/    test_masks/
├── pancreas_3way/
│   └── ...
└── lung_3way/
    └── ...
```

* images: `.jpg` / `.png`; masks: `.png`
* mask labels: `0` = background, `1` = organ, `2` = tumour
* images and masks are paired by **filename stem**

Point the harness at the parent directory with `--data-root` or the `DATA_ROOT`
environment variable (it must contain `{liver,pancreas,lung}_3way`, not a single
split).

The datasets used in the paper are public benchmarks (LiTS and the Medical
Segmentation Decathlon Lung/Pancreas tasks), converted to 2D slices with
patient-level splits. *Please obtain and license the raw data from the original
sources; this repository does not redistribute medical images.*

---

## Quick start

Train and evaluate AMAL-Net on the liver split (seed 42):

```bash
export DATA_ROOT=/path/to/data
CUDA_VISIBLE_DEVICES=0 bash run_one.sh amal liver 42
```

This runs `train.py` (best checkpoint selected on **val** tumour Dice) followed
by `evaluate.py` on the **test** split. Outputs land in
`runs/liver_amal_s42/` (checkpoints, metrics JSON, logs).

Run the full comparison (4 models × 3 datasets × 3 seeds):

```bash
python run_matrix.py --manifest manifests/main.tsv
```

Aggregate into mean ± std tables:

```bash
python summarize.py --runs runs --out summary
```

Individual steps, if you prefer manual control:

```bash
python train.py     --model amal --dataset liver --seed 42 --out-dir runs
python evaluate.py  --model amal --dataset liver --seed 42 \
                    --checkpoint runs/liver_amal_s42/ckpt/best_tumor_dice.pth --split test
```

Run `python train.py --help` / `python evaluate.py --help` for the full option
list (epochs, batch size, AMP, augmentation, AMAL-Net routing knobs such as
`--top-k`, `--enlargement`, `--halo`, `--grid`, `--routing`, etc.).

---

## Comparison protocol

All four models are trained and evaluated under an identical protocol:

| Item | Setting |
|---|---|
| Input | 3-channel RGB, resized to 256×256, ImageNet normalization |
| Augmentation | horizontal flip, ±10° rotation, mild colour jitter / blur |
| Optimizer | AdamW, lr 3e-4, weight decay 5e-4 |
| Schedule | cosine annealing + 10-epoch warm-up |
| Epochs | 300 (early stop, patience 60) |
| Batch size | 8, AMP (fp16) |
| Tumour oversampling | 3× via `WeightedRandomSampler` |
| Seeds | 42 / 43 / 44 |
| Checkpoint selection | best tumour Dice on **val** |
| Final evaluation | once on **test** (no test-time selection) |
| Metrics | mean Dice (3 classes), organ Dice, tumour Dice, mIoU, tumour IoU; optional HD95 / ASSD |

Only the model-native loss differs: `AMALLoss` for AMAL-Net, `COMANetLoss` for
COMA-Net, and the shared `StandardSegLoss` for UCTransNet and nnU-Net 2D.

---

## Pretrained weights

Weights and exact experiment logs are not bundled in this repository. If a
public release of the trained checkpoints is made available, it will be linked
here (e.g. Zenodo / Google Drive).

---

## Citation

```bibtex
@article{amalnet2026,
  title   = {AMAL-Net: Adaptive Multibranch Attention Learning Network for Medical Lesion Segmentation},
  author  = {Li Ma},
  journal = {<journal>},
  note    = {Under review},
  year    = {2026}
}
```

---

## License

<to-be-added: see the `LICENSE` file — MIT suggested>
