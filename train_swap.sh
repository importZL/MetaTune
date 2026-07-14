# Swap-meta ablation for Reviewer Comment #1.
# Inverts the assignment: prompt embedding = lower-level (non-meta, trained on D1);
# LoRA + decoder = upper-level (meta, trained on D2).
#
# LR mirror convention: each parameter set keeps its original LR; only the slots swap.
# In swap mode: base_lr trains the prompt embedding; prompt_base_lr trains LoRA+decoder.
# So swap base_lr := original prompt_base_lr, and swap prompt_base_lr := original base_lr.
#
# Per-task swap LRs (mirror of train.sh configs):
#   blood:        base_lr=1e-3   prompt_base_lr=5e-3
#   osteosarcoma: base_lr=1e-3   prompt_base_lr=1e-3
#   cellBT474:    base_lr=1e-3   prompt_base_lr=5e-3
#   cellHuh7:     base_lr=1e-3   prompt_base_lr=5e-3
#   multimodal:   base_lr=5e-3   prompt_base_lr=1e-3
#   cyto:         base_lr=5e-3   prompt_base_lr=1e-3
#
# Run with 3 seeds per task (e.g., 42, 40, 22) to match the existing 3-replicate ablation pattern.
# Override DATASET, TRAIN_IMAGES, BASE_LR, PROMPT_LR, and SEED through the environment.

DATASET=${DATASET:-blood}
DATA_ROOT=${DATA_ROOT:?Set DATA_ROOT to the dataset parent directory}
BASE_LR=${BASE_LR:-1e-3}
PROMPT_LR=${PROMPT_LR:-5e-3}
SEED=${SEED:-42}

${PYTHON:-python} -W ignore train.py \
    --root_path "${TRAIN_IMAGES:?Set TRAIN_IMAGES to the dataset train/Images directory}" \
    --output ./output_swap \
    --module sam_lora_mask_decoder \
    --max_epochs 100 \
    --num_data 4 \
    --wandb_mode disabled \
    --batch_size 1 \
    --dataset ${DATASET} \
    --exp_type swap_meta \
    --base_lr ${BASE_LR} \
    --prompt_base_lr ${PROMPT_LR} \
    --gpu_id 1 \
    --num_classes 1 \
    --dice_param 0.8 \
    --rank 4 \
    --seed ${SEED} \
    --swap_meta
