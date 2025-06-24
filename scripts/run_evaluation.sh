#!/bin/bash

# Please check the following variables
DATASET_NAME="v5_sub_clip"
if [ "$DATASET_NAME" == "test" ]; then
    # For experiments in test folder
    MASKED_IMG_DIRNAME="dog_backpack_1_image"
else
    # For experiments of dataset
    MASKED_IMG_DIRNAME=$DATASET_NAME
fi
EXP_NAME="20250527_v2_0524"
IS_MULTISEED=true
if [ "$IS_MULTISEED" = true ]; then
    # For experiments of multiple seeds
    EXP_SEED="-1"
else
    # For experiments of single seed
    EXP_SEED="0"
fi
# ======================================

BASE_DIR="/home/jack/Code/Research/instree_analysis/experiment_image"
SCRIPT_DIR="/home/jack/Code/Research/instree_analysis"
EXP_IMG_DIR="${BASE_DIR}/${DATASET_NAME}/${EXP_NAME}"
ORIGIN_IMG_DIR="${BASE_DIR}/${DATASET_NAME}/${MASKED_IMG_DIRNAME}_masked"
OUTPUT_DIR="${BASE_DIR}/scores/${DATASET_NAME}"

# Generate images
cd "$SCRIPT_DIR" || exit
if [ "$EXP_SEED" == "-1" ]; then
    python utils/image_generator.py \
        --dataset_name "$DATASET_NAME" \
        --exp_name "$EXP_NAME" \
        --exp_seed "$EXP_SEED" \
        --multiseed
else
    python utils/image_generator.py \
        --dataset_name "$DATASET_NAME" \
        --exp_name "$EXP_NAME" \
        --exp_seed "$EXP_SEED"
fi

echo "Start evaluating..."
python tools/evaluate_experiment.py \
    --exp_img_dir "$EXP_IMG_DIR" \
    --origin_img_dir "$ORIGIN_IMG_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --model_type "clip"
