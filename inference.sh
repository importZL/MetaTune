#!/usr/bin/env bash
set -euo pipefail

VOLUME_PATH=${VOLUME_PATH:?Set VOLUME_PATH to the test/Images directory}
LORA_CKPT=${LORA_CKPT:?Set LORA_CKPT to the trained best.pth}
DATASET=${DATASET:?Set DATASET to the dataset code}
GPU=${GPU:-0}
SAM_CKPT=${SAM_CKPT:-./checkpoints/sam_vit_b_01ec64.pth}
OUTPUT_DIR=${OUTPUT_DIR:-./predictions/$DATASET}
MODULE=${MODULE:-sam_lora_mask_decoder}
SAVE_PREDICTIONS=${SAVE_PREDICTIONS:-0}
EXTRA_ARGS=("$@")
SAVE_ARGS=()
if [[ "$SAVE_PREDICTIONS" == "1" ]]; then SAVE_ARGS+=(--is_savenii); fi
${PYTHON:-python} -W ignore inference.py \
    --volume_path "$VOLUME_PATH" \
    --lora_ckpt "$LORA_CKPT" \
    --gpu_id "$GPU" \
    --module "$MODULE" \
    --dataset "$DATASET" \
    --num_classes 1 \
    --ckpt "$SAM_CKPT" \
    --output_dir "$OUTPUT_DIR" \
    "${SAVE_ARGS[@]}" \
    "${EXTRA_ARGS[@]}"
