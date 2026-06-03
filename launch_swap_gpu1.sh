#!/bin/bash
# Swap-meta ablation queue — GPU 1
# Tasks: cellHuh7, multimodal, cyto × seeds 42, 40, 22  (9 runs)
# LR mirror: each parameter set keeps its original LR; only slots swap.

set -e
cd /data2/li/workspace/SAMed
mkdir -p logs

PY=/home/li/anaconda/envs/yolo/bin/python
LOG=logs/swap_gpu1.log

run() {
    local ds=$1 root=$2 base=$3 prompt=$4 seed=$5
    echo "===== $(date '+%F %T')  ${ds}  seed=${seed}  base_lr=${base}  prompt_base_lr=${prompt} =====" | tee -a $LOG
    $PY -W ignore train.py \
        --root_path "$root" \
        --output ./output_swap \
        --module sam_lora_mask_decoder \
        --max_epoch 100 \
        --num_data 4 \
        --wandb_mode disabled \
        --batch_size 1 \
        --dataset "$ds" \
        --exp_type swap_meta \
        --base_lr "$base" \
        --prompt_base_lr "$prompt" \
        --gpu_id 1 \
        --num_classes 1 \
        --dice_param 0.8 \
        --rank 4 \
        --seed "$seed" \
        --swap_meta >> $LOG 2>&1
}

# cellHuh7: original 5e-3 / 1e-3 → swap 1e-3 / 5e-3
run cellHuh7   /data2/li/workspace/data/LiveCell_datasets/Huh7/train/Images                  1e-3 5e-3 42
run cellHuh7   /data2/li/workspace/data/LiveCell_datasets/Huh7/train/Images                  1e-3 5e-3 40
run cellHuh7   /data2/li/workspace/data/LiveCell_datasets/Huh7/train/Images                  1e-3 5e-3 22

# multimodal: original 1e-3 / 5e-3 → swap 5e-3 / 1e-3
run multimodal /data2/li/workspace/data/multi-modal-bio/train/Images                         5e-3 1e-3 42
run multimodal /data2/li/workspace/data/multi-modal-bio/train/Images                         5e-3 1e-3 40
run multimodal /data2/li/workspace/data/multi-modal-bio/train/Images                         5e-3 1e-3 22

# cyto: original 1e-3 / 5e-3 → swap 5e-3 / 1e-3
run cyto       /data2/li/workspace/data/CytoNuke/train/Images                                5e-3 1e-3 42
run cyto       /data2/li/workspace/data/CytoNuke/train/Images                                5e-3 1e-3 40
run cyto       /data2/li/workspace/data/CytoNuke/train/Images                                5e-3 1e-3 22

echo "===== $(date '+%F %T')  GPU 1 queue DONE =====" | tee -a $LOG
