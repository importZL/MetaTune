#!/usr/bin/env bash
set -euo pipefail

DATASET=${DATASET:-cyto}
TRAIN_IMAGES=${TRAIN_IMAGES:?Set TRAIN_IMAGES to the dataset train/Images directory}
BASE_LR=${BASE_LR:?Set BASE_LR from HYPERPARAMETERS.md}
PROMPT_LR=${PROMPT_LR:?Set PROMPT_LR from HYPERPARAMETERS.md}
SEED=${SEED:-42}
NUM_DATA=${NUM_DATA:-4}
GPU=${GPU:-0}
SAM_CKPT=${SAM_CKPT:-./checkpoints/sam_vit_b_01ec64.pth}
MODULE=${MODULE:-sam_lora_mask_decoder}
TRAIN_SPLIT=${TRAIN_SPLIT:-0.5}
OUTPUT=${OUTPUT:-./output}
EXTRA_ARGS=("$@")
${PYTHON:-python} train.py \
    --root_path "$TRAIN_IMAGES" \
    --output "$OUTPUT" \
    --module "$MODULE" \
    --max_epochs 100 \
    --num_data "$NUM_DATA" \
    --train_split "$TRAIN_SPLIT" \
    --wandb_mode disabled \
    --batch_size 1 \
    --dataset "$DATASET" \
    --exp_type auto_first \
    --base_lr "$BASE_LR" \
    --prompt_base_lr "$PROMPT_LR" \
    --gpu_id "$GPU" \
    --num_classes 1 \
    --dice_param 0.8 \
    --rank 4 \
    --seed "$SEED" \
    --ckpt "$SAM_CKPT" \
    "${EXTRA_ARGS[@]}"
