#!/bin/bash
# Cellpose sweep: 2 datasets x 3 seeds x 2 models (cpsam + cyto3) = 12 runs.

set -e
REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
DATA_ROOT=${DATA_ROOT:?Set DATA_ROOT to the dataset parent directory}
cd "$REPO_ROOT"
mkdir -p logs

PY=${PYTHON:-python}
LOG=logs/cellpose_gpu0.log

run() {
    local ds=$1 num=$2 trim=$3 trma=$4 teim=$5 tema=$6 seed=$7 pre=$8
    echo "===== $(date '+%F %T')  ${ds}  seed=${seed}  num=${num}  pretrained=${pre} =====" | tee -a $LOG
    $PY baselines/cellpose_samed.py \
        --train_imgs "$trim" --train_masks "$trma" \
        --test_imgs "$teim" --test_masks "$tema" \
        --dataset "$ds" --num_data "$num" --seed "$seed" --gpu_id 0 \
        --pretrained_model "$pre" --n_epochs 100 \
        --output ./output_baselines/cellpose >> $LOG 2>&1
}

CYTO_TRAIN_IMG=${DATA_ROOT}/CytoNuke/train/Images
CYTO_TRAIN_MSK=${DATA_ROOT}/CytoNuke/train/Masks_instance
CYTO_TEST_IMG=${DATA_ROOT}/CytoNuke/test/Images
CYTO_TEST_MSK=${DATA_ROOT}/CytoNuke/test/Masks_instance

FLUO_TRAIN_IMG=${DATA_ROOT}/fluocell_v2/red/train/Images
FLUO_TRAIN_MSK=${DATA_ROOT}/fluocell_v2/red/train/Masks_instance
FLUO_TEST_IMG=${DATA_ROOT}/fluocell_v2/red/test/Images
FLUO_TEST_MSK=${DATA_ROOT}/fluocell_v2/red/test/Masks_instance

for pre in cpsam cyto3; do
    for seed in 42 40 22; do
        run cyto         4 "$CYTO_TRAIN_IMG" "$CYTO_TRAIN_MSK" "$CYTO_TEST_IMG" "$CYTO_TEST_MSK" $seed $pre
        run fluocellRed 10 "$FLUO_TRAIN_IMG" "$FLUO_TRAIN_MSK" "$FLUO_TEST_IMG" "$FLUO_TEST_MSK" $seed $pre
    done
done

echo "===== $(date '+%F %T')  Cellpose sweep DONE =====" | tee -a $LOG
