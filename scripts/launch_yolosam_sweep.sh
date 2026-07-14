#!/bin/bash
# YOLOv7+SAM-bilevel sweep on CytoNuke (4-shot) + FluoRed (10-shot), 3 seeds.
# Uses the cell_count-pretrained YOLOv7-seg backbone (consistent with cell_count ablations).

set -e
REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
DATA_ROOT=${DATA_ROOT:?Set DATA_ROOT to the dataset parent directory}
YOLOSAM_ROOT=${YOLOSAM_ROOT:-${REPO_ROOT}/third_party/yolov7_sam}
cd "$YOLOSAM_ROOT"
mkdir -p ${REPO_ROOT}/logs

PY=${PYTHON:-python}
LOG=${REPO_ROOT}/logs/yolosam_sweep.log
WEIGHTS=${YOLO_WEIGHTS:?Set YOLO_WEIGHTS to the pretrained detector checkpoint}
SAM_CKPT=${SAM_CKPT:-${REPO_ROOT}/checkpoints/sam_vit_b_01ec64.pth}

run() {
    local ds=$1 n=$2 seed=$3 yaml_root=$4 gpu=$5
    local name="${ds}${n}_yolosam_s${seed}"
    echo "===== $(date '+%F %T')  ${ds}  n=${n}  seed=${seed}  gpu=${gpu} =====" | tee -a $LOG
    WANDB_MODE=disabled $PY -W ignore segment/train.py \
        --data "${yaml_root}/data_n${n}_s${seed}.yaml" \
        --weights "$WEIGHTS" \
        --cfg models/segment/yolov7-seg.yaml \
        --epochs 20 --batch 1 --imgsz 256 \
        --hyp data/hyp.scratch.custom.yaml \
        --name "$name" \
        --seed "$seed" \
        --project "${REPO_ROOT}/output_baselines/yolosam_runs" \
        --wandb_mode disabled \
        --sam_ckpt "$SAM_CKPT" \
        --device $gpu --noplots >> $LOG 2>&1
}

# CytoNuke: 4-shot
for seed in 42 40 22; do
    run cyto 4 $seed ${DATA_ROOT}/yolo_CytoNuke 1
done
# fluored: 10-shot
for seed in 42 40 22; do
    run fluored 10 $seed ${DATA_ROOT}/yolo_fluored 1
done

echo "===== $(date '+%F %T')  YOLOv7+SAM sweep DONE =====" | tee -a $LOG
