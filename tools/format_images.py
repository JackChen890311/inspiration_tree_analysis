import cv2
import os
import argparse

def center_crop(image_folder):
    for image_path in os.listdir(image_folder):
        image = cv2.imread(os.path.join(image_folder + image_path))
        # center crop
        h, w, _ = image.shape
        if h > w:
            image = image[h//2 - w//2:h//2 + w//2, :, :]
        else:
            image = image[:, w//2 - h//2:w//2 + h//2, :]

        # save image
        cv2.imwrite(os.path.join(image_folder + image_path), image)

def resize(image_folder, size):
    for image_path in os.listdir(image_folder):
        image = cv2.imread(os.path.join(image_folder + image_path))
        image = cv2.resize(image, (size, size))
        cv2.imwrite(os.path.join(image_folder + image_path), image)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="Path to image folder")
    parser.add_argument("--resize", action="store_true", help="Resize images to 512x512")
    args = parser.parse_args()

    center_crop(args.path)
    resize(args.path, 512)
