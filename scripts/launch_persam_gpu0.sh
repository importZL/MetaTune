#!/bin/bash
# PerSAM-F full sweep — 8 tasks × 3 seeds = 24 runs, sequential on GPU 0.
# Settings: topk=32, train_epoch=1000 (chosen via smoke test on blood).

set -e
cd /data2/li/workspace/SAMed
mkdir -p logs

PY=/home/li/anaconda/envs/yolo/bin/python
LOG=logs/persam_gpu0.log

run() {
    local ds=$1 num=$2 root=$3 test=$4 seed=$5
    echo "===== $(date '+%F %T')  ${ds}  seed=${seed}  num=${num} =====" | tee -a $LOG
    $PY baselines/persam_f_samed.py \
        --root_path "$root" \
        --test_path "$test" \
        --dataset "$ds" \
        --num_data "$num" \
        --seed "$seed" \
        --gpu_id 0 \
        --train_epoch 1000 \
        --topk 32 \
        --output ./output_baselines/persam_f >> $LOG 2>&1
}

# Tasks with num_data=4
for seed in 42 40 22; do
    run blood        4 /data2/li/workspace/data/blood-cell/train/Images                                              /data2/li/workspace/data/blood-cell/test/Images                                              $seed
    run osteosarcoma 4 /data2/li/workspace/data/CellPose_datasets/bone_osteosarcoma_cell_dataset/train/Images        /data2/li/workspace/data/CellPose_datasets/bone_osteosarcoma_cell_dataset/test/Images        $seed
    run cellBT474    4 /data2/li/workspace/data/LiveCell_datasets/BT474/train/Images                                 /data2/li/workspace/data/LiveCell_datasets/BT474/test/Images                                 $seed
    run cellHuh7     4 /data2/li/workspace/data/LiveCell_datasets/Huh7/train/Images                                  /data2/li/workspace/data/LiveCell_datasets/Huh7/test/Images                                  $seed
    run multimodal   4 /data2/li/workspace/data/multi-modal-bio/train/Images                                         /data2/li/workspace/data/multi-modal-bio/test/Images                                         $seed
    run cyto         4 /data2/li/workspace/data/CytoNuke/train/Images                                                /data2/li/workspace/data/CytoNuke/test/Images                                                $seed
done

# Tasks with num_data=10
for seed in 42 40 22; do
    run fluocellRed 10 /data2/li/workspace/data/fluocell_v2/red/train/Images                                         /data2/li/workspace/data/fluocell_v2/red/test/Images                                         $seed
    run sartorius   10 /data2/li/workspace/data/sartorius/train/Images                                               /data2/li/workspace/data/sartorius/test/Images                                               $seed
done

echo "===== $(date '+%F %T')  PerSAM-F sweep DONE =====" | tee -a $LOG
