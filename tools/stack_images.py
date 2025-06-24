import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(description='Stack image pairs from a folder into a grid and save to output.')
    parser.add_argument('--input_folder', type=str, help='Path to the input folder containing 80 images.')
    parser.add_argument('--output_folder', type=str, help='Path to save the output stacked image.')
    return parser.parse_args()

def load_image_pairs(input_folder):
    return [os.path.join(input_folder, name) for name in os.listdir(input_folder)]


def stack_images_and_save_it(pair, output_path):
    sub_img = {}
    for title, paths in pair.items():
        assert len(paths) == 40, f"Expected 40 images, got {len(paths)} for '{title}'"

        # Load all images and ensure same size
        imgs = [Image.open(p).convert('RGB') for p in paths]
        w, h = imgs[0].size
        imgs = [img.resize((w, h)) for img in imgs]  # ensure consistent size if needed

        # Stack into 10 rows of 4 images
        rows = []
        for i in range(10):
            row_imgs = imgs[i*4:(i+1)*4]
            row_np = np.hstack([np.array(img) for img in row_imgs])
            rows.append(row_np)

        sub_img[title] = np.vstack(rows)

    # Plot in two subfigures
    num_sub_fig = len(sub_img)
    fig, ax = plt.subplots(1, num_sub_fig, figsize=(10, 10))
    title_map = {
        "v1": "Vl",
        "v2": "Vr",
        "v1v2": "Vl Vr"
    }
    if num_sub_fig == 1:
        # ax 是單一物件
        k, v = next(iter(sub_img.items()))
        ax.imshow(v)
        ax.set_title(title_map[k])
        ax.axis('off')
    else:
        for idx, (k, v) in enumerate(sub_img.items()):
            ax[idx].imshow(v)
            ax[idx].set_title(title_map[k])
            ax[idx].axis('off')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.1)
    plt.close(fig)

def main():
    args = parse_args()
    for concept in os.listdir(args.input_folder):
        if not os.path.isdir(os.path.join(args.input_folder, concept)):
            continue
        concept_path = os.path.join(args.input_folder, concept)
        output_dir = os.path.join(args.output_folder, concept)
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"final_samples.png")
        aspect = ["v1", "v2", "v1v2"]
        image_path = {}
        for k in aspect:
            image_path[k] = load_image_pairs(os.path.join(concept_path, k))
        stack_images_and_save_it(image_path, output_path)
        print(f"Saved stacked image to {output_path}")

if __name__ == '__main__':
    main()
