#!/bin/bash
# Swap-meta ablation queue — GPU 0
# Tasks: blood, osteosarcoma, cellBT474 × seeds 42, 40, 22  (9 runs)
# LR mirror: each parameter set keeps its original LR; only slots swap.

set -e
cd /data2/li/workspace/SAMed
mkdir -p logs

PY=/home/li/anaconda/envs/yolo/bin/python
LOG=logs/swap_gpu0.log

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
        --gpu_id 0 \
        --num_classes 1 \
        --dice_param 0.8 \
        --rank 4 \
        --seed "$seed" \
        --swap_meta >> $LOG 2>&1
}

# blood: original (non-meta=LoRA)=5e-3, (meta=prompt)=1e-3 → swap: base=1e-3 (prompt), prompt=5e-3 (LoRA)
run blood        /data2/li/workspace/data/blood-cell/train/Images                                              1e-3 5e-3 42
run blood        /data2/li/workspace/data/blood-cell/train/Images                                              1e-3 5e-3 40
run blood        /data2/li/workspace/data/blood-cell/train/Images                                              1e-3 5e-3 22

# osteosarcoma: original 1e-3 / 1e-3 → swap 1e-3 / 1e-3
run osteosarcoma /data2/li/workspace/data/CellPose_datasets/bone_osteosarcoma_cell_dataset/train/Images        1e-3 1e-3 42
run osteosarcoma /data2/li/workspace/data/CellPose_datasets/bone_osteosarcoma_cell_dataset/train/Images        1e-3 1e-3 40
run osteosarcoma /data2/li/workspace/data/CellPose_datasets/bone_osteosarcoma_cell_dataset/train/Images        1e-3 1e-3 22

# cellBT474: original 5e-3 / 1e-3 → swap 1e-3 / 5e-3
run cellBT474    /data2/li/workspace/data/LiveCell_datasets/BT474/train/Images                                 1e-3 5e-3 42
run cellBT474    /data2/li/workspace/data/LiveCell_datasets/BT474/train/Images                                 1e-3 5e-3 40
run cellBT474    /data2/li/workspace/data/LiveCell_datasets/BT474/train/Images                                 1e-3 5e-3 22

echo "===== $(date '+%F %T')  GPU 0 queue DONE =====" | tee -a $LOG
