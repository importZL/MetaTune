#!/bin/bash
# Package every MetaTune semantic-segmentation checkpoint reported in the paper
# (main results + ablations + swap-meta) into a Zenodo-ready tarball.
#
# Includes ONLY the (dataset, N) pairs reported in the paper:
#   N=4:  blood, osteosarcoma, cellBT474, cellHuh7, multimodal, cyto,
#         yeast-bright, yeast-contrast
#   N=10: fluocellRed, sartorius
#
# Other historical runs (caiman, fluo, marrow, nuclei, openscope-*) and other
# N values found in output_bio/ are intentionally excluded.
#
# Instance-segmentation checkpoints (BLO-SAM-instance, YOLOv7+SAM-bilevel,
# Cellpose / Cellpose-SAM / cpsam+BLO-SAM-bilevel, StarDist) are NOT included
# in this v1 deposit.
#
# Output: $WORK/zenodo_metatune_v1.tar.gz

set -e
WORK=${WORK:-$PWD}
SRC_SAMED=$WORK/SAMed
STAGE=$WORK/zenodo_stage
OUT_ROOT="metatune_zenodo_v1"

# Paper-reported (dataset, N) pairs as one regex per category:
#   ALLOW_N4   = N=4 tasks (8 datasets)
#   ALLOW_N10  = N=10 tasks (2 datasets)
# Each regex is matched against the leading "{dataset}{N}" of the run-dir basename.
ALLOW_REGEX='^(blood4|osteosarcoma4|cellBT4744|cellHuh74|multimodal4|cyto4|yeast-bright4|yeast-contrast4|fluocellRed10|sartorius10)_'

include_run() {
    # Returns 0 (success) iff the basename matches ALLOW_REGEX.
    local name=$1
    echo "$name" | grep -qE "$ALLOW_REGEX"
}

rm -rf "$STAGE"; mkdir -p "$STAGE/$OUT_ROOT"
cd "$STAGE/$OUT_ROOT"

# 1. MetaTune semantic-seg (bilevel "auto_first") — main paper results
echo "[1/3] MetaTune semantic-seg (auto_first)..."
mkdir -p semantic_main
n_sem=0
for src in $SRC_SAMED/output_bio/*_auto_first_img256_* $SRC_SAMED/output_repeat/*_auto_first_img256_*; do
    [ -f "$src/best.pth" ] || continue
    name=$(basename "$src")
    include_run "$name" || continue
    mkdir -p "semantic_main/$name"
    cp "$src/best.pth"   "semantic_main/$name/"
    cp "$src/config.txt" "semantic_main/$name/" 2>/dev/null || true
    n_sem=$((n_sem + 1))
done
echo "  -> $n_sem MetaTune checkpoints kept"

# 2. Joint-optimization baseline ("vanilla")
echo "[2/3] Joint-optimization baseline (vanilla)..."
mkdir -p ablations/vanilla_joint
n_van=0
for src in $SRC_SAMED/output_bio/*_vanilla_img256_*; do
    [ -f "$src/best.pth" ] || continue
    name=$(basename "$src")
    include_run "$name" || continue
    mkdir -p "ablations/vanilla_joint/$name"
    cp "$src/best.pth"   "ablations/vanilla_joint/$name/"
    cp "$src/config.txt" "ablations/vanilla_joint/$name/" 2>/dev/null || true
    n_van=$((n_van + 1))
done
echo "  -> $n_van vanilla-joint checkpoints kept"

# 3. Swap-meta ablation (Comment #1) — already restricted to 6 ablation tasks × 3 seeds
echo "[3/3] Swap-meta ablation..."
mkdir -p ablations/swap_meta
n_swap=0
for src in $SRC_SAMED/output_swap/*_swap_meta_img256_*; do
    [ -f "$src/best.pth" ] || continue
    name=$(basename "$src")
    include_run "$name" || continue
    mkdir -p "ablations/swap_meta/$name"
    cp "$src/best.pth"   "ablations/swap_meta/$name/"
    cp "$src/config.txt" "ablations/swap_meta/$name/" 2>/dev/null || true
    n_swap=$((n_swap + 1))
done
echo "  -> $n_swap swap-meta checkpoints kept"

# README inside the archive
cat > README_zenodo.md <<EOF
# MetaTune — trained checkpoints (Zenodo deposit, v1)

This archive accompanies:

  "Meta-Finetuning Foundation Models for Generalizable Biological Image
   Segmentation in Ultra Low-Data Regimes"
  Li Zhang, Youwei Liang, Phuc Nguyen, Fanny Chapelin, Nan Hao, Pengtao Xie
  Cell Reports Methods (under review), manuscript CR-METHODS-D-26-00020.

The deposit contains every semantic-segmentation checkpoint reported in the
paper. Code is at https://github.com/importZL/MetaTune.

  DOI (this deposit): 10.5281/zenodo.20517421
                      https://doi.org/10.5281/zenodo.20517421

## Contents

  metatune_zenodo_v1/
    semantic_main/<run_name>/                       MetaTune (bilevel) main results.
                                                    8 biological tasks + yeast (bright/contrast)
                                                    at the (dataset, N) configurations used in
                                                    the paper (see below), 3 seeds each.
        best.pth                                    LoRA + decoder + prompt embedding state.
        config.txt                                  Full argparse namespace used.
    ablations/vanilla_joint/<run_name>/             Joint-optimization baseline:
                                                    same architecture as MetaTune trained
                                                    without the bilevel split. Used in
                                                    Fig. 6 D1+D2 bar and as the "SAMed*"
                                                    entries in Fig. 3 / Table S1.
        best.pth, config.txt
    ablations/swap_meta/<run_name>/                 Reviewer Comment #1 ablation:
                                                    meta/non-meta assignment inverted
                                                    (prompt -> non-meta on D1,
                                                     LoRA + decoder -> meta on D2).
        best.pth, config.txt

Total checkpoints: $((n_sem + n_van + n_swap))
  - $n_sem  MetaTune semantic main results (bilevel)
  - $n_van  vanilla joint baseline
  - $n_swap swap-meta ablation

## (Dataset, N) coverage

Paper-reported configurations only — historical runs at other N values or for
datasets not in the paper (caiman, fluo, marrow, nuclei, openscope-*) are
excluded.

  N=4   blood, osteosarcoma, cellBT474, cellHuh7, multimodal, cyto,
        yeast-bright, yeast-contrast
  N=10  fluocellRed, sartorius

## Filename convention

  <dataset><N>_<exp_type>_img256_<YYYYMMDD-HHMMSS>_<NNNN>/

Where:
  <exp_type>   auto_first (bilevel) / vanilla (joint) / swap_meta
  <NNNN>       test-set Dice * 10000 (e.g., _8649 = Dice 0.8649)

## Loading a checkpoint

\`\`\`python
import torch
from segment_anything import sam_model_registry
from sam_lora_mask_decoder import LoRA_Sam   # from the GitHub repo

sam, _ = sam_model_registry["vit_b"](
    image_size=256, num_classes=1,
    checkpoint="sam_vit_b_01ec64.pth",   # original SAM ViT-B
    pixel_mean=[0, 0, 0], pixel_std=[1, 1, 1],
)
net = LoRA_Sam(sam, r=4)
net.load_lora_parameters(
    "semantic_main/blood4_auto_first_img256_20240826-055350_8649/best.pth",
    device=torch.device("cuda"),
)
\`\`\`

The exact hyperparameters used to *produce* each checkpoint are saved alongside
it in the run directory as \`config.txt\` (full argparse namespace from
\`train.py\`).

## Cross-reference to the paper

  HYPERPARAMETERS.md   per-task / per-method hyperparameter tables
  REPRODUCE.md         figure-by-figure reproduction commands
in the GitHub repository (https://github.com/importZL/MetaTune).

## Instance-segmentation checkpoints

Not included in v1. Reproducing them does not require shared weights — the
instance-seg baselines (Cellpose-SAM, classical Cellpose, StarDist) auto-load
their own publicly-available pretrained models, and the fine-tunes train from
those public checkpoints via the runners in \`baselines/\` and \`scripts/\` of
the GitHub repository.

## License

CC BY 4.0.

## Citation

\`\`\`bibtex
@article{zhang2026metatune,
  title  = {Meta-Finetuning Foundation Models for Generalizable Biological
            Image Segmentation in Ultra Low-Data Regimes},
  author = {Zhang, Li and Liang, Youwei and Nguyen, Phuc and Chapelin, Fanny
            and Hao, Nan and Xie, Pengtao},
  journal = {Cell Reports Methods},
  year   = {2026},
  note   = {Manuscript CR-METHODS-D-26-00020}
}
@inproceedings{zhang2024blosam,
  title     = {BLO-SAM: Bi-level Optimization for Generalizable Segmentation
               with the Segment Anything Model},
  author    = {Zhang, Li and others},
  booktitle = {ICML},
  year      = {2024}
}
\`\`\`
EOF

# Build the tarball
cd "$WORK"
TARBALL=zenodo_metatune_v1.tar.gz
rm -f "$TARBALL"
echo
echo "Creating $TARBALL ..."
tar -czf "$TARBALL" -C "$STAGE" "$OUT_ROOT"

echo
echo "=== Inventory ==="
echo "Total weight files: $(find $STAGE/$OUT_ROOT -name 'best.pth' | wc -l)"
echo
echo "=== Per-section sizes ==="
du -sh "$STAGE/$OUT_ROOT"/* | sed 's/^/  /'
echo
echo "=== Tarball ==="
ls -lh "$TARBALL" | awk '{print "  size:", $5}'
echo "  location: $WORK/$TARBALL"
echo
echo "Done."
