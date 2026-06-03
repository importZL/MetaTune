#!/bin/bash
# YOLOv7+SAM-bilevel on FULL train data (CytoNuke=43, fluored=138), 3 seeds.
# Higher data than the 4/10-shot setting; tests whether more data closes the
# gap to Cellpose-SAM. Re-uses the cell_count-pretrained backbone.

set -e
cd /data2/li/workspace/yolov7-sam
mkdir -p /data2/li/workspace/SAMed/logs

PY=/home/li/anaconda/envs/yolo/bin/python
LOG=/data2/li/workspace/SAMed/logs/yolosam_fulldata.log
WEIGHTS=/data/li/seg_baselines/yolov7-segmentation/runs/train-seg/cell_count-yolov7-seg/weights/best.pt
SAM_CKPT=/data1/li/Auto_SAMed/checkpoints/sam_vit_b_01ec64.pth

run() {
    local ds=$1 yaml_p=$2 seed=$3 gpu=$4
    local name="${ds}_full_yolosam_s${seed}"
    echo "===== $(date '+%F %T')  ${ds}  full  seed=${seed}  gpu=${gpu} =====" | tee -a $LOG
    WANDB_MODE=disabled $PY -W ignore segment/train.py \
        --data "$yaml_p" \
        --weights "$WEIGHTS" \
        --cfg yolov7-seg.yaml \
        --epochs 20 --batch 1 --imgsz 256 \
        --hyp data/hyp.scratch.custom.yaml \
        --seed $seed \
        --name "$name" \
        --project yolosam_full \
        --wandb_mode disabled \
        --sam_ckpt "$SAM_CKPT" \
        --device $gpu --noplots >> $LOG 2>&1
}

for seed in 42 40 22; do
    run cyto    /data2/li/workspace/data/yolo_CytoNuke/data_full.yaml $seed 1
    run fluored /data2/li/workspace/data/yolo_fluored/data_full.yaml  $seed 1
done

echo "===== $(date '+%F %T')  YOLOv7+SAM full-data sweep DONE =====" | tee -a $LOG
