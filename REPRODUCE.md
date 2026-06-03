# REPRODUCE.md — figure-by-figure reproduction guide

This document gives the exact commands to reproduce every figure and table of the paper. Each section is self-contained; you can jump straight to the figure / table you want.

## Prerequisites

Once for the whole repo:

```bash
# 1. Clone
git clone https://github.com/importZL/MetaTune.git
cd MetaTune

# 2. Build the main env (PyTorch 2.5.1 + most baselines)
conda env create -f environment.yml
conda activate metatune

# 3. (For StarDist only) build the TensorFlow env in parallel
conda env create -f environment-stardist.yml

# 4. Download the SAM ViT-B checkpoint
mkdir -p checkpoints && cd checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
cd ..

# 5. (For Matcher only) download DINOv2 ViT-L weights
mkdir -p baselines/Matcher/models && cd baselines/Matcher/models
wget https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth
cd ../../..

# 6. (For Cellpose-SAM) cpsam weights download automatically on first use to ~/.cellpose/models/cpsam.

# 7. Download / preprocess the eight biological datasets per DATA.md.
#    Place each at <DATA_ROOT>/<dataset>/{train,test}/{Images,Masks}/
```

The reproduction commands below assume the conda env `metatune` is active and `<DATA_ROOT>` is set:

```bash
export DATA_ROOT=/path/to/datasets
export CKPT=$PWD/checkpoints/sam_vit_b_01ec64.pth
```

---

## Figure 2 — MetaTune vs DeepLab / UNet / vanilla SAM on 8 tasks

```bash
# MetaTune (3 seeds per task)
for seed in 42 40 22; do
  for ds in blood osteosarcoma cellBT474 cellHuh7 multimodal cyto fluocellRed sartorius; do
    # See HYPERPARAMETERS.md for per-task LRs; example for blood:
    bash train.sh   # set --dataset blood --base_lr 5e-3 --prompt_base_lr 1e-3 --seed $seed
    bash inference.sh   # auto-uses the best.pth from the just-trained run
  done
done

# DeepLab, UNet — see baselines/ for adapters (or use the public repos directly).
# Vanilla SAM uses GT-derived point/box prompts at inference time; see paper Methods.

# Render the figure:
jupyter notebook figures.ipynb   # run cell 2
```

Expected output: 8 SVG files at `result-figures/result1-{bccd,osteo,bt474,huh7,multimodal,cyto,fluored,sartorius}.svg`.

## Figure 3 — MetaTune vs MedSA / SAMed / uSAM on 8 tasks

```bash
# Same MetaTune sweep as Figure 2 (reuse those checkpoints).

# MedSA, SAMed-style, uSAM baselines — see paper References [23], [24], [3] respectively.
# Run with the same N supports per task; see HYPERPARAMETERS.md.

jupyter notebook figures.ipynb   # run cell 4
```

## Figure 4 — MetaTune vs few-shot baselines (HSNet + Reviewer-added)

```bash
# HSNet — see Min et al., 2021. We use the public repo.

# PerSAM-F (Reviewer Comment #3, added in revision):
bash scripts/launch_persam_gpu0.sh

# Matcher (Reviewer Comment #3, added in revision):
bash scripts/launch_matcher_gpu0.sh

# Aggregate:
python baselines/aggregate_persam.py output_baselines/persam_f persam_f
python baselines/aggregate_persam.py output_baselines/matcher    matcher

jupyter notebook figures.ipynb   # run cell 6 (panels updated to include PerSAM-F, Matcher)
```

## Figure 5 — In-distribution / out-of-distribution evaluation on yeast

```bash
# Train on xy01 (4 supports), evaluate on (a) held-out xy01 and (b) xy02-xy34.
for seed in 42 40 22; do
  bash train.sh   # --dataset yeast-bright --root_path $DATA_ROOT/Yeast/PhaseContrast/train/Images --seed $seed
  bash inference.sh   # --volume_path $DATA_ROOT/Yeast/PhaseContrast/test_in/Images   # ID
  bash inference.sh   # --volume_path $DATA_ROOT/Yeast/PhaseContrast/test_out/Images  # OOD
  # repeat with --dataset yeast-contrast
done
jupyter notebook figures.ipynb   # run cells 5 / 8 / 10
```

## Figure 6 — Bilevel ablation (joint D₁∪D₂ / 1st-order / 2nd-order)

```bash
# 1st-order (default): same MetaTune training as above.

# Joint D₁∪D₂ baseline:
bash train_vanilla.sh   # uses train_vanilla.py + trainer_vanilla.py

# 2nd-order (DARTS unrolled):
# pass --unrolled to train.py via train.sh

# Swap-meta ablation (Reviewer Comment #1, added in revision):
bash launch_swap_gpu0.sh
bash launch_swap_gpu1.sh
# Test-set inference on the swap checkpoints:
bash infer_swap.sh 0 blood osteosarcoma cellBT474
bash infer_swap.sh 1 cellHuh7 multimodal cyto

jupyter notebook figures.ipynb   # cell 10 (Fig 6)
```

## Figure 7 — End-to-end vs separate optimization

End-to-end is the default `train.sh`. The "separate" baseline trains LoRA with prompt frozen, then freezes LoRA and trains prompt. Implementation note: this is conceptually a sequential degradation of `train.py`; see paper Methods. (We do not ship a separate script for this; it can be reproduced by running `train_vanilla.py` then `train.py` with `--lora_ckpt` pointing to the vanilla output.)

## Figure 8 — Component ablation (which SAM component to LoRA)

Vary the `--module` flag in `train.sh`:

```bash
# Mask decoder only (paper default, our best)
bash train.sh   # --module sam_lora_mask_decoder

# Image encoder only:
bash train.sh   # --module sam_lora_image_encoder

# Prompt encoder only:
bash train.sh   # --module sam_lora_prompt_encoder

# All three:
bash train.sh   # --module sam_lora_all
```

```bash
jupyter notebook figures.ipynb   # cell 8 (Fig 8)
```

## Figure 9 — Split-strategy ablation (1:1 vs 3:1)

The 1:1 split is hard-coded in `trainer.py` (`num_train = int(len(db) * 0.5)`). For 3:1, edit that line to `int(len(db) * 0.75)` and re-run. (We have not parameterized this since it's a one-line code change with no measurable effect; see paper Fig. 9.)

## Figure 10 — No-prompt ablation

Freeze `no_mask_embed` at random initialization and train only the LoRA + decoder on D₁ ∪ D₂. This is the same as the joint-optimization baseline (Fig 6 D₁+D₂) with the prompt frozen — produced by `train_vanilla.sh` with an additional flag (or by editing `trainer_vanilla.py` to disable gradients on `no_mask_embed`).

## Figures 11-15 — Qualitative examples

These are visualizations of predicted masks; reproducing them requires:
1. Running inference on the test set with `--is_savenii`.
2. Rendering the saved mask PNGs with the layout in `figures.ipynb`.

## Table 1 — Paired t-tests (8 tasks × 3 baselines)

After running the 24-cell Dice grid (8 tasks × MetaTune/MedSA/SAMed/uSAM × 3 seeds), compute the paired two-sided t-tests via the snippet at the bottom of `figures.ipynb` (or manually with `scipy.stats.ttest_rel`).

**Important note about Table 1**: in the original Table 1 of the submission, t / p / effect-size values were not reproducible from the per-seed Dice values stored in `figures.ipynb`. The revised Table 1 in the resubmission uses standard paired two-sided t-tests at n=3, df=2 (computed by `scipy.stats.ttest_rel`); see Reviewer Comment #4 response.

---

## Instance-segmentation extensions (Reviewer Comment #6)

### Data preparation (one-time)

```bash
# Regenerate instance-ID masks from the original COCO annotations
python baselines/regen_cytonuke_instances.py
python baselines/regen_fluored_instances.py

# Convert to YOLO polygon labels for the YOLOv7+SAM-bilevel comparison
python baselines/instance_to_yolo.py \
    --out_root $DATA_ROOT/yolo_CytoNuke \
    --train_imgs $DATA_ROOT/CytoNuke/train/Images \
    --train_masks $DATA_ROOT/CytoNuke/train/Masks_instance \
    --test_imgs   $DATA_ROOT/CytoNuke/test/Images \
    --test_masks  $DATA_ROOT/CytoNuke/test/Masks_instance

# Sample N-shot YOLO splits (seeds 42, 40, 22):
for s in 42 40 22; do
    python baselines/sample_yolo_nshot.py --root $DATA_ROOT/yolo_CytoNuke --n 4 --seed $s
done
```

### Baselines

```bash
# Cellpose-SAM (cpsam) and classical Cellpose (cyto3)
bash scripts/launch_cellpose_gpu0.sh

# StarDist — note: separate env
conda activate stardist
bash scripts/launch_stardist_gpu1.sh
conda activate metatune

# YOLOv7+SAM-bilevel
bash scripts/launch_yolosam_sweep.sh           # 4/10-shot
bash scripts/launch_yolosam_fulldata.sh        # full training data

# BLO-SAM-instance (Route B, our flow head)
bash scripts/launch_instance.sh 1              # 1 = GPU index

# cpsam + BLO-SAM-bilevel (ours, best)
bash scripts/launch_cpsam_bilevel.sh
```

### Aggregating all instance-seg results

```bash
python baselines/aggregate_persam.py output_baselines/cellpose       cellpose
python baselines/aggregate_persam.py output_baselines/stardist       stardist
python baselines/aggregate_persam.py output_baselines/cpsam_bilevel  cpsam_bilevel
# YOLOv7+SAM results are in yolov7-sam/yolosam_runs/*/results.csv (best-epoch row).
```

---

## Software versions used

Everything pinned in `requirements.txt` and `environment.yml`. The headline versions:

| | Version |
|---|---|
| Python | 3.10 |
| PyTorch | 2.5.1+cu121 |
| torchvision | 0.20.1+cu121 |
| segment-anything | 1.0 |
| cellpose | 4.1.1 |
| stardist | 0.9.2 (in separate env) |
| TensorFlow | 2.21.0 (StarDist env only) |
| CUDA driver | 560.35.03 |
| CUDA runtime (sys) | 12.6 |

## Hardware used

See [HARDWARE.md](HARDWARE.md). Single-GPU on NVIDIA A100 80GB. The full reproduction takes ~32-40 GPU-hours.

## Released checkpoints

All `best.pth` checkpoints for the semantic-segmentation runs reported in the paper (MetaTune main results, vanilla-joint baseline, swap-meta ablation) are deposited on Zenodo at DOI [10.5281/zenodo.20517421](https://doi.org/10.5281/zenodo.20517421). Each checkpoint is bundled with the exact `config.txt` (full argparse namespace) of the run that produced it.

Instance-segmentation checkpoints (BLO-SAM-instance, YOLOv7+SAM-bilevel, Cellpose / Cellpose-SAM / cpsam+BLO-SAM-bilevel, StarDist) are not included in v1: the baselines auto-load their own publicly-available pretrained weights, and the fine-tunes train from those public weights using the runners in `baselines/` and `scripts/`.
