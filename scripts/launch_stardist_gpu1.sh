#!/bin/bash
# StarDist sweep: 2 datasets x 3 seeds = 6 runs on GPU 1.

set -e
REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
DATA_ROOT=${DATA_ROOT:?Set DATA_ROOT to the dataset parent directory}
cd "$REPO_ROOT"
mkdir -p logs
PY=${PYTHON:-python}
LOG=logs/stardist_gpu1.log

run() {
    local ds=$1 num=$2 trim=$3 trma=$4 teim=$5 tema=$6 seed=$7
    echo "===== $(date '+%F %T')  ${ds}  seed=${seed}  num=${num} =====" | tee -a $LOG
    $PY baselines/stardist_samed.py \
        --train_imgs "$trim" --train_masks "$trma" \
        --test_imgs "$teim" --test_masks "$tema" \
        --dataset "$ds" --num_data "$num" --seed "$seed" --gpu_id 1 \
        --n_epochs 100 --steps_per_epoch 50 \
        --output ./output_baselines/stardist >> $LOG 2>&1
}

CYTO_TR_IMG=${DATA_ROOT}/CytoNuke/train/Images
CYTO_TR_MSK=${DATA_ROOT}/CytoNuke/train/Masks_instance
CYTO_TE_IMG=${DATA_ROOT}/CytoNuke/test/Images
CYTO_TE_MSK=${DATA_ROOT}/CytoNuke/test/Masks_instance
FL_TR_IMG=${DATA_ROOT}/fluocell_v2/red/train/Images
FL_TR_MSK=${DATA_ROOT}/fluocell_v2/red/train/Masks_instance
FL_TE_IMG=${DATA_ROOT}/fluocell_v2/red/test/Images
FL_TE_MSK=${DATA_ROOT}/fluocell_v2/red/test/Masks_instance

for seed in 42 40 22; do
    run cyto         4 "$CYTO_TR_IMG" "$CYTO_TR_MSK" "$CYTO_TE_IMG" "$CYTO_TE_MSK" $seed
    run fluocellRed 10 "$FL_TR_IMG"   "$FL_TR_MSK"   "$FL_TE_IMG"   "$FL_TE_MSK"   $seed
done

echo "===== $(date '+%F %T')  StarDist sweep DONE =====" | tee -a $LOG
