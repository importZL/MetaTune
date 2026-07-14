#!/bin/bash
# Cellpose remaining runs (3 cpsam done: cyto42, cyto40, fluored42).
# Remaining: cpsam: cyto22, fluored40, fluored22  + cyto3: all 6.

set -e
REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
DATA_ROOT=${DATA_ROOT:?Set DATA_ROOT to the dataset parent directory}
cd "$REPO_ROOT"
mkdir -p logs
PY=${PYTHON:-python}
LOG=logs/cellpose_remaining.log

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

CYTO_TR_IMG=${DATA_ROOT}/CytoNuke/train/Images
CYTO_TR_MSK=${DATA_ROOT}/CytoNuke/train/Masks_instance
CYTO_TE_IMG=${DATA_ROOT}/CytoNuke/test/Images
CYTO_TE_MSK=${DATA_ROOT}/CytoNuke/test/Masks_instance
FL_TR_IMG=${DATA_ROOT}/fluocell_v2/red/train/Images
FL_TR_MSK=${DATA_ROOT}/fluocell_v2/red/train/Masks_instance
FL_TE_IMG=${DATA_ROOT}/fluocell_v2/red/test/Images
FL_TE_MSK=${DATA_ROOT}/fluocell_v2/red/test/Masks_instance

# Remaining cpsam runs
run cyto         4 "$CYTO_TR_IMG" "$CYTO_TR_MSK" "$CYTO_TE_IMG" "$CYTO_TE_MSK" 22 cpsam
run fluocellRed 10 "$FL_TR_IMG"   "$FL_TR_MSK"   "$FL_TE_IMG"   "$FL_TE_MSK"   40 cpsam
run fluocellRed 10 "$FL_TR_IMG"   "$FL_TR_MSK"   "$FL_TE_IMG"   "$FL_TE_MSK"   22 cpsam

# All cyto3 runs
for seed in 42 40 22; do
    run cyto         4 "$CYTO_TR_IMG" "$CYTO_TR_MSK" "$CYTO_TE_IMG" "$CYTO_TE_MSK" $seed cyto3
    run fluocellRed 10 "$FL_TR_IMG"   "$FL_TR_MSK"   "$FL_TE_IMG"   "$FL_TE_MSK"   $seed cyto3
done

echo "===== $(date '+%F %T')  Cellpose remaining DONE =====" | tee -a $LOG
