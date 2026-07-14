# REPRODUCE.md — figure-by-figure reproduction guide

This guide maps repository procedures to manuscript Figures 1–7 and Supplementary Figures S1–S11. It distinguishes runnable MetaTune workflows from comparison settings whose source or provenance is not present in this repository.

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
export PYTHON=python
# Launch scripts also accept GPU, output, and checkpoint variables documented below.
```

---

## Figure 1 — Method overview

Figure 1 is a conceptual diagram; the source asset is `figures/method_overview.png` and requires no experiment.

## Figure 2 — MetaTune vs DeepLab / UNet / vanilla SAM on 8 tasks

```bash
# Configure each task with the paths and learning rates in HYPERPARAMETERS.md. Example: BCCD.
for seed in 42 40 22; do
  DATASET=blood TRAIN_IMAGES="$DATA_ROOT/blood-cell/train/Images" \
  BASE_LR=5e-3 PROMPT_LR=1e-3 SEED="$seed" NUM_DATA=4 bash train.sh
done

# Set LORA_CKPT to each generated best.pth before evaluation.
DATASET=blood VOLUME_PATH="$DATA_ROOT/blood-cell/test/Images" \
LORA_CKPT="/path/to/best.pth" bash inference.sh

# DeepLab, UNet — see baselines/ for adapters (or use the public repos directly).
# Vanilla SAM uses GT-derived point/box prompts at inference time; see paper Methods.

# Render the figure:
# Plotting notebook is not distributed; consume the reported CSV/SVG artifacts or your aggregated arrays.
```

The repository does not include the original plotting notebook; use the resulting scores to recreate the manuscript panels.

## Figure 3 — MetaTune vs MedSA / SAMed / uSAM on 8 tasks

```bash
# Same MetaTune sweep as Figure 2 (reuse those checkpoints).

# MedSA, SAMed-style, uSAM baselines — see paper References [23], [24], [3] respectively.
# Run with the same N supports per task; see HYPERPARAMETERS.md.

# Plotting notebook is not distributed; consume the reported CSV/SVG artifacts or your aggregated arrays.
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

# Plotting notebook is not distributed; consume the reported CSV/SVG artifacts or your aggregated arrays.
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
# Plotting notebook is not distributed; consume the reported CSV/SVG artifacts or your aggregated arrays.
```

## Figure 6 — Bilevel ablation (joint D₁∪D₂ / 1st-order / 2nd-order)

```bash
# 1st-order (default): same MetaTune training as above.

# Joint D₁∪D₂ baseline:
${PYTHON:-python} train_vanilla.py --root_path "$TRAIN_IMAGES" --dataset "$DATASET" --base_lr "$BASE_LR" --seed "$SEED" --ckpt "$CKPT"

# 2nd-order (DARTS unrolled):
# pass --unrolled to train.py via train.sh

# Swap-meta ablation (Reviewer Comment #1, added in revision):
bash launch_swap_gpu0.sh
bash launch_swap_gpu1.sh
# Test-set inference on the swap checkpoints:
bash infer_swap.sh 0 blood osteosarcoma cellBT474
bash infer_swap.sh 1 cellHuh7 multimodal cyto

# Plotting notebook is not distributed; consume the reported CSV/SVG artifacts or your aggregated arrays.
```

## Figure 7 — End-to-end vs separate optimization

End-to-end is the default `train.sh`. The "separate" baseline trains LoRA with prompt frozen, then freezes LoRA and trains prompt. Implementation note: this is conceptually a sequential degradation of `train.py`; see paper Methods. (We do not ship a separate script for this; it can be reproduced by running `train_vanilla.py` then `train.py` with `--lora_ckpt` pointing to the vanilla output.)

## Supplementary Figure S1 — Component ablation (which SAM component to LoRA)

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
# Plotting notebook is not distributed; consume the reported CSV/SVG artifacts or your aggregated arrays.
```

## Supplementary Figure S2 — Split-strategy ablation (1:1 vs 3:1)

Use `--train_split 0.5` for 1:1 and `--train_split 0.75` for 3:1; no source edit is required.

## Supplementary Figure S3 — No-prompt ablation

Freeze `no_mask_embed` at random initialization and train only the LoRA + decoder on D₁ ∪ D₂. This is the same as the joint-optimization baseline (Fig. 6 D₁+D₂) with the prompt frozen — produced by `train_vanilla.py --freeze_prompt`.

## Supplementary Figures S4-S8 — Qualitative examples

Run `inference.py --is_savenii --output_dir "$OUTPUT_DIR"`; one binary PNG is written per test image. The original plotting notebook is not distributed, so panel layout must be recreated from these PNGs and the manuscript.

## Table 1 — Paired t-tests (8 tasks × 3 baselines)

After collecting the three paired seed scores per method and task, compute two-sided paired tests with `scipy.stats.ttest_rel`.

---

## Instance-segmentation extensions (Reviewer Comment #6)

### Data preparation (one-time)

```bash
# Regenerate instance-ID masks from the original COCO annotations
python baselines/regen_cytonuke_instances.py \
  --coco-json "$DATA_ROOT/CytoNuke/coco.json" --data-root "$DATA_ROOT/CytoNuke"
python baselines/regen_fluored_instances.py \
  --train-json "$DATA_ROOT/fluocell_v2/red/trainval_ori/ground_truths/COCO/annotations_red_trainval.json" \
  --test-json "$DATA_ROOT/fluocell_v2/red/test_ori/ground_truths/COCO/annotations_red_test.json" \
  --data-root "$DATA_ROOT/fluocell_v2/red"

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

# YOLOv7+SAM-bilevel (requires an external checkout and checkpoint)
export YOLOSAM_ROOT=/path/to/yolov7-sam
export YOLO_WEIGHTS=/path/to/detector/best.pt
export SAM_CKPT="$CKPT"
bash scripts/launch_yolosam_sweep.sh           # 4/10-shot
bash scripts/launch_yolosam_fulldata.sh        # full training data

# BLO-SAM-instance (Route B, our flow head)
bash scripts/launch_instance.sh 1              # 1 = GPU index

# cpsam + BLO-SAM-bilevel (ours, best)
bash scripts/launch_cpsam_bilevel.sh
```

### Aggregating all instance-seg results

```bash
python baselines/aggregate_persam.py output_baselines/cellpose       cpsam
python baselines/aggregate_persam.py output_baselines/cellpose       cyto3
python baselines/aggregate_persam.py output_baselines/stardist       stardist
python baselines/aggregate_persam.py output_baselines/cpsam_bilevel  cpsam_bilevel
# YOLOv7+SAM results are in yolov7-sam/yolosam_runs/*/results.csv (best-epoch row).
```

## Supplementary Figures S9-S11 — Instance-segmentation extension

Use the data preparation, baseline launch, and aggregation commands above. The aggregator reads `result.json` for Cellpose, StarDist, and cpsam-bilevel layouts and exits with an error if no result is found.

## External YOLOv7+SAM-bilevel provenance

The MetaTune repository history contains neither the upstream repository URL nor a commit hash for the `yolov7-sam` checkout used in the experiments. Therefore an exact checkout cannot be stated responsibly from the available files. The launchers now require `YOLOSAM_ROOT` and `YOLO_WEIGHTS` instead of embedding an author-machine path. Reproducing the reported YOLO result remains blocked until the authors record the URL, commit hash, and detector-checkpoint provenance here.

---

## Software versions used

Dependency declarations are in `requirements.txt` and `environment.yml`; NumPy and OpenCV use compatible ranges in the Conda environment. The headline versions:

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
