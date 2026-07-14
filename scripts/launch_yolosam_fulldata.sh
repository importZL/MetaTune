#!/bin/bash
# YOLOv7+SAM-bilevel on FULL train data (CytoNuke=43, fluored=138), 3 seeds.
# Higher data than the 4/10-shot setting; tests whether more data closes the
# gap to Cellpose-SAM. Re-uses the cell_count-pretrained backbone.

set -e
REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
DATA_ROOT=${DATA_ROOT:?Set DATA_ROOT to the dataset parent directory}
YOLOSAM_ROOT=${YOLOSAM_ROOT:-${REPO_ROOT}/third_party/yolov7_sam}
cd "$YOLOSAM_ROOT"
mkdir -p ${REPO_ROOT}/logs

PY=${PYTHON:-python}
LOG=${REPO_ROOT}/logs/yolosam_fulldata.log
WEIGHTS=${YOLO_WEIGHTS:?Set YOLO_WEIGHTS to the pretrained detector checkpoint}
SAM_CKPT=${SAM_CKPT:-${REPO_ROOT}/checkpoints/sam_vit_b_01ec64.pth}

run() {
    local ds=$1 yaml_p=$2 seed=$3 gpu=$4
    local name="${ds}_full_yolosam_s${seed}"
    echo "===== $(date '+%F %T')  ${ds}  full  seed=${seed}  gpu=${gpu} =====" | tee -a $LOG
    WANDB_MODE=disabled $PY -W ignore segment/train.py \
        --data "$yaml_p" \
        --weights "$WEIGHTS" \
        --cfg models/segment/yolov7-seg.yaml \
        --epochs 20 --batch 1 --imgsz 256 \
        --hyp data/hyp.scratch.custom.yaml \
        --seed $seed \
        --name "$name" \
        --project "${REPO_ROOT}/output_baselines/yolosam_full" \
        --wandb_mode disabled \
        --sam_ckpt "$SAM_CKPT" \
        --device $gpu --noplots >> $LOG 2>&1
}

for seed in 42 40 22; do
    run cyto    ${DATA_ROOT}/yolo_CytoNuke/data_full.yaml $seed 1
    run fluored ${DATA_ROOT}/yolo_fluored/data_full.yaml  $seed 1
done

echo "===== $(date '+%F %T')  YOLOv7+SAM full-data sweep DONE =====" | tee -a $LOG
