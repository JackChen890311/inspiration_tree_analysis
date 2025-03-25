import cv2
import os
import argparse


def center_crop(image):
    """
    Center crop an image to a square.
    Parameters: image (np.ndarray): Input image of shape (H, W, C).
    Returns: np.ndarray: Center cropped image.
    """
    h, w, _ = image.shape
    if h > w:
        image = image[h//2 - w//2:h//2 + w//2, :, :]
    else:
        image = image[:, w//2 - h//2:w//2 + h//2, :]
    return image


def resize(image, size = 512):
    """
    Resize an image to a square.
    Parameters: image (np.ndarray): Input image of shape (H, W, C).
                size (int): Size of the resized image.
    Returns: np.ndarray: Resized image.
    """
    image = cv2.resize(image, (size, size))
    return image


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, help="Path to image folder")
    parser.add_argument("--resize", action="store_true", help="Resize images to 512x512")
    args = parser.parse_args()

    for image_path in os.listdir(args.path):
        image = cv2.imread(os.path.join(args.path + image_path))
        image = center_crop(image)
        if args.resize:
            image = resize(image, 512)
        cv2.imwrite(os.path.join(args.path + image_path), image)
