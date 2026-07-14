#!/bin/bash
# Route B (BLO-SAM-instance) full sweep: 2 datasets x 3 seeds = 6 runs.
# Pass the GPU index as the first argument.

set -e
REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
DATA_ROOT=${DATA_ROOT:?Set DATA_ROOT to the dataset parent directory}
cd "$REPO_ROOT"
mkdir -p logs

PY=${PYTHON:-python}
GPU=${1:-0}
LOG=logs/instance_gpu${GPU}.log

train_and_infer() {
    local ds=$1 num=$2 base_lr=$3 prompt_lr=$4 seed=$5 trim=$6 teim=$7 tema=$8
    echo "===== $(date '+%F %T')  ${ds}  seed=${seed}  num=${num} =====" | tee -a $LOG

    # Train
    $PY train_instance.py \
        --root_path "$trim" \
        --output ./output_instance \
        --module sam_lora_mask_decoder_instance \
        --dataset "$ds" --num_data "$num" --seed "$seed" \
        --gpu_id $GPU --num_classes 1 \
        --max_epochs 300 --batch_size 1 --img_size 256 --rank 4 \
        --base_lr "$base_lr" --prompt_base_lr "$prompt_lr" --dice_param 0.8 \
        --exp_type instance --wandb_mode disabled >> $LOG 2>&1

    # Find the just-written run dir
    LATEST=$(ls -td ./output_instance/${ds}${num}_instance_img256_* | head -1)
    echo "  ckpt dir: $LATEST" | tee -a $LOG

    # Inference
    $PY inference_instance.py \
        --test_imgs "$teim" --test_masks "$tema" \
        --lora_ckpt "${LATEST}/best.pth" \
        --module sam_lora_mask_decoder_instance \
        --dataset "$ds" --num_classes 1 --img_size 256 --rank 4 \
        --gpu_id $GPU --seed "$seed" \
        --output_dir "$LATEST" >> $LOG 2>&1

    # Append AP@0.5 (x10000) suffix to the dir for consistency with other baselines
    AP=$(python3 -c "import json; d=json.load(open('${LATEST}/instance_result.json')); print(int(round(d['AP@0.5']*10000)))" 2>/dev/null || echo "0000")
    NEW=${LATEST}_${AP}
    mv "$LATEST" "$NEW" 2>/dev/null && echo "  renamed -> $NEW" | tee -a $LOG
}

CYTO_TR=${DATA_ROOT}/CytoNuke/train/Images
CYTO_TE_IMG=${DATA_ROOT}/CytoNuke/test/Images
CYTO_TE_MSK=${DATA_ROOT}/CytoNuke/test/Masks_instance
FL_TR=${DATA_ROOT}/fluocell_v2/red/train/Images
FL_TE_IMG=${DATA_ROOT}/fluocell_v2/red/test/Images
FL_TE_MSK=${DATA_ROOT}/fluocell_v2/red/test/Masks_instance

for seed in 42 40 22; do
    train_and_infer cyto         4 1e-3 5e-3 $seed "$CYTO_TR" "$CYTO_TE_IMG" "$CYTO_TE_MSK"
    train_and_infer fluocellRed 10 5e-3 1e-3 $seed "$FL_TR"   "$FL_TE_IMG"   "$FL_TE_MSK"
done

echo "===== $(date '+%F %T')  Instance sweep DONE =====" | tee -a $LOG
