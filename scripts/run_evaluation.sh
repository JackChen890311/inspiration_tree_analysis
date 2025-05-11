#!/bin/bash

# If evaluate for instree, please un-comment the --multiseed option
DATASET_NAME="v2_sub_clip"
# For experiments in test folder
MASKED_IMG_DIRNAME="dog_backpack_1_image"
# For experiments of dataset
MASKED_IMG_DIRNAME=$DATASET_NAME
EXP_NAME="20250511_mask_with_origin"
EXP_SEED="0"

BASE_DIR="/home/jack/Code/Research/instree_analysis/experiment_image"
SCRIPT_DIR="/home/jack/Code/Research/instree_analysis"
EXP_IMG_DIR="${BASE_DIR}/${DATASET_NAME}/${EXP_NAME}"
ORIGIN_IMG_DIR="${BASE_DIR}/${DATASET_NAME}/${MASKED_IMG_DIRNAME}_masked"
OUTPUT_DIR="${BASE_DIR}/scores/${DATASET_NAME}"

# Generate images
cd "$SCRIPT_DIR" || exit
python utils/image_generator.py \
    --dataset_name "$DATASET_NAME" \
    --exp_name "$EXP_NAME" \
    --exp_seed "$EXP_SEED" \
    # --multiseed

# Evaluate with different models
for MODEL in clip dino
do
    echo "Start evaluating with model: $MODEL"
    python tools/evaluate_experiment.py \
        --exp_img_dir "$EXP_IMG_DIR" \
        --origin_img_dir "$ORIGIN_IMG_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --model_type "$MODEL"
done
