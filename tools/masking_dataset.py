import os
import cv2
import argparse
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description="Apply mask to images and save them.")
    dataset_name = "test"
    image_folder_name = "20250505_instree_1_image"
    # image_folder_name = dataset_name
    parser.add_argument("--original_folder", type=str, default=f"/home/jack/Code/Research/instree_analysis/experiment_image/{dataset_name}/{image_folder_name}", help="Path to the original images folder.")
    parser.add_argument("--mask_folder", type=str, default=f"/home/jack/Code/Research/instree_analysis/experiment_image/{dataset_name}/{image_folder_name}_mask", help="Path to the mask images folder.")
    parser.add_argument("--output_folder", type=str, default=f"/home/jack/Code/Research/instree_analysis/experiment_image/{dataset_name}/{image_folder_name}_masked", help="Path to save the masked images.")
    return parser.parse_args()

def apply_mask_and_save(args):
    original_folder = args.original_folder
    mask_folder = args.mask_folder
    output_folder = args.output_folder

    for root, _, files in os.walk(original_folder):
        for file in files:
            original_path = os.path.join(root, file)
            relative_path = os.path.relpath(original_path, original_folder)
            mask_path = os.path.join(mask_folder, relative_path)
            output_path = os.path.join(output_folder, relative_path)

            if not os.path.exists(mask_path):
                print(f"Mask not found for: {relative_path}")
                continue

            os.makedirs(os.path.dirname(output_path), exist_ok=True)

            # Read the original image and the mask
            image = cv2.imread(original_path, cv2.IMREAD_UNCHANGED)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

            if image is None or mask is None:
                print(f"Error reading image or mask for: {relative_path}")
                continue

            # If mask is not binary, make it binary
            _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)

            bg_color = 0  # black background
            bg_color = 127 # gray background
            # bg_color = 255  # white background

            # Apply mask
            if len(image.shape) == 3:  # Color image
                masked_image = cv2.bitwise_and(image, image, mask=binary_mask)
                background = np.full_like(image, bg_color)
                mask_inv = cv2.bitwise_not(binary_mask)
                white_background = cv2.bitwise_and(background, background, mask=mask_inv)
                masked_image = cv2.add(masked_image, white_background)
            else:  # Grayscale image
                masked_image = cv2.bitwise_and(image, binary_mask)
                background = np.full_like(image, bg_color)
                mask_inv = cv2.bitwise_not(binary_mask)
                white_background = cv2.bitwise_and(background, mask_inv)
                masked_image = cv2.add(masked_image, white_background)

            # Save the masked image
            cv2.imwrite(output_path, masked_image)
            print(f"Saved: {output_path}")

if __name__ == "__main__":
    args = parse_args()
    apply_mask_and_save(args)
