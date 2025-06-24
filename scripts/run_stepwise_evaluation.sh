#!/bin/bash

# Please check the following variables
DATASET_NAME="v2_sub_clip"
if [ "$DATASET_NAME" == "test" ]; then
    # For experiments in test folder
    MASKED_IMG_DIRNAME="dog_backpack_1_image"
else
    # For experiments of dataset
    MASKED_IMG_DIRNAME=$DATASET_NAME
fi
EXP_NAME="20250524_last_rand_exp"
IS_MULTISEED=false
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
OUTPUT_DIR="${BASE_DIR}/scores/${DATASET_NAME}/stepwise/${EXP_NAME}"

# Generate images
cd "$SCRIPT_DIR" || exit

for STEP in 100 200 300 400 500 600 700 800 900 1000
do
    echo "Start evaluating with step: $STEP"

    if [ "$EXP_SEED" == "-1" ]; then
        python utils/image_generator.py \
            --dataset_name "$DATASET_NAME" \
            --exp_name "$EXP_NAME" \
            --exp_seed "$EXP_SEED" \
            --out_path "$BASE_DIR/temp" \
            --emb_name "embeds/learned_embeds-steps-$STEP.bin" \
            --multiseed
    else
        python utils/image_generator.py \
            --dataset_name "$DATASET_NAME" \
            --exp_name "$EXP_NAME" \
            --exp_seed "$EXP_SEED" \
            --out_path "$BASE_DIR/temp" \
            --emb_name "embeds/learned_embeds-steps-$STEP.bin"
    fi

    echo "Start evaluating..."
    python tools/evaluate_experiment.py \
        --exp_img_dir "$BASE_DIR/temp/$DATASET_NAME/$EXP_NAME"\
        --origin_img_dir "$ORIGIN_IMG_DIR" \
        --output_dir "$OUTPUT_DIR" \
        --model_type "clip" \
        --out_sub_name "$STEP"
    
    rm -r "$BASE_DIR/temp"
done