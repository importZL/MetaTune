#!/bin/bash
# Inference for swap-meta checkpoints.
# Usage: ./infer_swap.sh <gpu_id> <task1> [task2] ...
# Example: ./infer_swap.sh 0 blood osteosarcoma cellBT474

set -e
cd /data2/li/workspace/SAMed

GPU=$1
shift
PY=/home/li/anaconda/envs/yolo/bin/python
LOG=logs/infer_swap_gpu${GPU}.log
mkdir -p logs

# dataset code → test-set Images dir
declare -A TEST_PATH
TEST_PATH[blood]=/data2/li/workspace/data/blood-cell/test/Images
TEST_PATH[osteosarcoma]=/data2/li/workspace/data/CellPose_datasets/bone_osteosarcoma_cell_dataset/test/Images
TEST_PATH[cellBT474]=/data2/li/workspace/data/LiveCell_datasets/BT474/test/Images
TEST_PATH[cellHuh7]=/data2/li/workspace/data/LiveCell_datasets/Huh7/test/Images
TEST_PATH[multimodal]=/data2/li/workspace/data/multi-modal-bio/test/Images
TEST_PATH[cyto]=/data2/li/workspace/data/CytoNuke/test/Images

# dataset code → directory prefix used in output_swap/
# (cellBT474 → cellBT4744, cellHuh7 → cellHuh74, others append num_data directly)
declare -A DIRPREFIX
DIRPREFIX[blood]=blood4
DIRPREFIX[osteosarcoma]=osteosarcoma4
DIRPREFIX[cellBT474]=cellBT4744
DIRPREFIX[cellHuh7]=cellHuh74
DIRPREFIX[multimodal]=multimodal4
DIRPREFIX[cyto]=cyto4

for ds in "$@"; do
    prefix=${DIRPREFIX[$ds]}
    test_path=${TEST_PATH[$ds]}
    for run_dir in output_swap/${prefix}_swap_meta_img256_*; do
        # Skip already-renamed dirs (they end with _NNNN where NNNN is the test dice)
        if [[ "$run_dir" =~ _[0-9]{4}$ ]] && [[ ! "$run_dir" =~ -[0-9]{6}$ ]]; then
            echo "SKIP already-scored: $run_dir" | tee -a $LOG
            continue
        fi
        ckpt="$run_dir/best.pth"
        if [ ! -f "$ckpt" ]; then
            echo "MISSING ckpt: $ckpt" | tee -a $LOG
            continue
        fi
        echo "===== $(date '+%F %T')  infer  $ds  $run_dir =====" | tee -a $LOG
        out=$($PY -W ignore inference.py \
            --volume_path "$test_path" \
            --lora_ckpt "$ckpt" \
            --gpu_id "$GPU" \
            --module sam_lora_mask_decoder \
            --dataset "$ds" \
            --num_classes 1 2>&1)
        echo "$out" >> $LOG
        # Extract Test dice score line: "Test dice score: 0.7440"
        dice=$(echo "$out" | grep -oE "Test dice score: [0-9.]+" | awk '{print $NF}')
        if [ -n "$dice" ]; then
            # Rename dir to append _NNNN (Dice × 10000), matching the paper's convention
            suffix=$(printf "%04d" $(echo "$dice * 10000" | bc | cut -d. -f1))
            newdir="${run_dir}_${suffix}"
            mv "$run_dir" "$newdir"
            echo "  -> Dice $dice  renamed to $newdir" | tee -a $LOG
        else
            echo "  !! no dice score parsed for $run_dir" | tee -a $LOG
        fi
    done
done

echo "===== $(date '+%F %T')  GPU ${GPU} inference DONE =====" | tee -a $LOG
