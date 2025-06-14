import os
import argparse
import imageio
import cv2
import numpy as np

def add_title(image, text, font=cv2.FONT_HERSHEY_SIMPLEX, font_scale=1, thickness=2):
    """ Adds a title to the image at the top with a black background. """
    height, width, _ = image.shape
    title_height = 50  # Space for title
    title_image = np.full((title_height, width, 3), (0, 0, 0), dtype=np.uint8)  # Black background
    cv2.putText(title_image, text, (10, 35), font, font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return np.vstack((title_image, image))  # Stack the title on top of the image

def make_gif(in_path, out_path_name, duration, sep_method):
    images = []
    for filename in sorted(os.listdir(in_path), key=sep_method):
        if filename.endswith(".png") or filename.endswith(".jpg"):
            img_path = os.path.join(in_path, filename)
            image = cv2.imread(img_path)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB for imageio
            image_with_title = add_title(image, f"Training Step {sep_method(filename)}")
            images.append(image_with_title)

    imageio.mimsave(out_path_name, images, duration=duration)

def train_step_sep_method(filename):
    return filename.split('.')[0].split('_')[1]

def attn_step_sep_method(filename):
    return int(filename.split('_')[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, help="Path to image folder")
    parser.add_argument("--output", type=str, help="File name to output folder")
    parser.add_argument("--duration", type=float, default=0.5, help="Duration of each frame in seconds")
    parser.add_argument("--sep_method", type=str, choices=["train", "attn"], default="train",
                        help="Method to separate training steps")
    args = parser.parse_args()

    sep_method_mapping = {
        "train": train_step_sep_method,
        "attn": attn_step_sep_method
    }

    make_gif(args.input, args.output, args.duration, sep_method_mapping[args.sep_method])
