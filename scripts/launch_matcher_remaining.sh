#!/bin/bash
# Matcher remaining runs (Sartorius dropped per decision).
# Already done: blood/osteo/cellBT474 all 3 seeds; cyto/multimodal/cellHuh7 seeds 42+40.
# Remaining: cellHuh7-22, multimodal-22, cyto-22, fluored x3.

set -e
REPO_ROOT=${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}
DATA_ROOT=${DATA_ROOT:?Set DATA_ROOT to the dataset parent directory}
cd "$REPO_ROOT"
mkdir -p logs
PY=${PYTHON:-python}
LOG=logs/matcher_remaining.log

run() {
    local ds=$1 num=$2 root=$3 test=$4 seed=$5
    echo "===== $(date '+%F %T')  ${ds}  seed=${seed}  num=${num} =====" | tee -a $LOG
    $PY baselines/matcher_samed.py \
        --root_path "$root" --test_path "$test" --dataset "$ds" --num_data "$num" \
        --seed "$seed" --gpu_id 0 --points_per_side 32 \
        --output ./output_baselines/matcher >> $LOG 2>&1
}

# Remaining seed=22 small tasks
run cellHuh7   4 ${DATA_ROOT}/LiveCell_datasets/Huh7/train/Images   ${DATA_ROOT}/LiveCell_datasets/Huh7/test/Images   22
run multimodal 4 ${DATA_ROOT}/multi-modal-bio/train/Images          ${DATA_ROOT}/multi-modal-bio/test/Images          22
run cyto       4 ${DATA_ROOT}/CytoNuke/train/Images                 ${DATA_ROOT}/CytoNuke/test/Images                 22

# FluoRed x3 seeds (num_data=10)
for seed in 42 40 22; do
    run fluocellRed 10 ${DATA_ROOT}/fluocell_v2/red/train/Images ${DATA_ROOT}/fluocell_v2/red/test/Images $seed
done

echo "===== $(date '+%F %T')  Matcher remaining DONE =====" | tee -a $LOG
