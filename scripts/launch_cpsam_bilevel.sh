#!/bin/bash
# cpsam + bilevel meta-finetuning sweep: 2 datasets x 3 seeds = 6 runs.
# Hypothesis: bilevel meta-tuning on top of cpsam (Cellpose-SAM) yields gains.

set -e
mkdir -p /data2/li/workspace/SAMed/logs
PY=/home/li/anaconda/envs/yolo/bin/python
LOG=/data2/li/workspace/SAMed/logs/cpsam_bilevel.log

run() {
    local ds=$1 num=$2 trim=$3 trma=$4 teim=$5 tema=$6 seed=$7
    echo "===== $(date '+%F %T')  ${ds}  seed=${seed}  num=${num} =====" | tee -a $LOG
    cd /tmp  # avoid the local segment_anything shadowing cellpose's
    $PY /data2/li/workspace/SAMed/baselines/cellpose_cpsam_bilevel.py \
        --train_imgs "$trim" --train_masks "$trma" \
        --test_imgs "$teim" --test_masks "$tema" \
        --dataset "$ds" --num_data "$num" --seed "$seed" --gpu_id 1 \
        --pretrained_model cpsam --n_epochs 50 \
        --lr_main 1e-5 --lr_meta 1e-3 \
        --meta_set out_diam_pos \
        --output /data2/li/workspace/SAMed/output_baselines/cpsam_bilevel >> $LOG 2>&1
}

CYTO_TR_IMG=/data2/li/workspace/data/CytoNuke/train/Images
CYTO_TR_MSK=/data2/li/workspace/data/CytoNuke/train/Masks_instance
CYTO_TE_IMG=/data2/li/workspace/data/CytoNuke/test/Images
CYTO_TE_MSK=/data2/li/workspace/data/CytoNuke/test/Masks_instance
FL_TR_IMG=/data2/li/workspace/data/fluocell_v2/red/train/Images
FL_TR_MSK=/data2/li/workspace/data/fluocell_v2/red/train/Masks_instance
FL_TE_IMG=/data2/li/workspace/data/fluocell_v2/red/test/Images
FL_TE_MSK=/data2/li/workspace/data/fluocell_v2/red/test/Masks_instance

for seed in 42 40 22; do
    run cyto         4 "$CYTO_TR_IMG" "$CYTO_TR_MSK" "$CYTO_TE_IMG" "$CYTO_TE_MSK" $seed
    run fluocellRed 10 "$FL_TR_IMG"   "$FL_TR_MSK"   "$FL_TE_IMG"   "$FL_TE_MSK"   $seed
done

echo "===== $(date '+%F %T')  cpsam+bilevel sweep DONE =====" | tee -a $LOG
