#!/bin/bash
# YOLOv7+SAM-bilevel sweep on CytoNuke (4-shot) + FluoRed (10-shot), 3 seeds.
# Uses the cell_count-pretrained YOLOv7-seg backbone (consistent with cell_count ablations).

set -e
cd /data2/li/workspace/yolov7-sam
mkdir -p /data2/li/workspace/SAMed/logs

PY=/home/li/anaconda/envs/yolo/bin/python
LOG=/data2/li/workspace/SAMed/logs/yolosam_sweep.log
WEIGHTS=/data/li/seg_baselines/yolov7-segmentation/runs/train-seg/cell_count-yolov7-seg/weights/best.pt
SAM_CKPT=/data1/li/Auto_SAMed/checkpoints/sam_vit_b_01ec64.pth

run() {
    local ds=$1 n=$2 seed=$3 yaml_root=$4 gpu=$5
    local name="${ds}${n}_yolosam_s${seed}"
    echo "===== $(date '+%F %T')  ${ds}  n=${n}  seed=${seed}  gpu=${gpu} =====" | tee -a $LOG
    WANDB_MODE=disabled $PY -W ignore segment/train.py \
        --data "${yaml_root}/data_n${n}_s${seed}.yaml" \
        --weights "$WEIGHTS" \
        --cfg yolov7-seg.yaml \
        --epochs 20 --batch 1 --imgsz 256 \
        --hyp data/hyp.scratch.custom.yaml \
        --name "$name" \
        --project yolosam_runs \
        --wandb_mode disabled \
        --sam_ckpt "$SAM_CKPT" \
        --device $gpu --noplots >> $LOG 2>&1
}

# CytoNuke: 4-shot
for seed in 42 40 22; do
    run cyto 4 $seed /data2/li/workspace/data/yolo_CytoNuke 1
done
# fluored: 10-shot
for seed in 42 40 22; do
    run fluored 10 $seed /data2/li/workspace/data/yolo_fluored 1
done

echo "===== $(date '+%F %T')  YOLOv7+SAM sweep DONE =====" | tee -a $LOG
