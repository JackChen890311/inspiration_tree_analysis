import numpy as np
import cv2
from PIL import Image
import torch

def otsu_thresholding(image: np.ndarray) -> np.ndarray:
    """
    Apply Otsu's thresholding to a color image by converting it to grayscale.
    Parameters: image (np.ndarray): Input color image of shape (H, W, C). 
    Returns: np.ndarray: Binary thresholded image.
    """
    grayscale = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(grayscale, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return np.expand_dims(binary_image, axis=2)

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    image = np.array(image)
    return image

def gaussian_noise(size):
    """
    Generate Gaussian noise image of given size.
    Parameters:
        size (int): Size of the image (size x size).
    Returns:
        np.ndarray: Gaussian noise image of shape (size, size, 3).
    """
    noise = np.random.normal(0, 1, (size, size, 3)).astype(np.float32)
    noise = (noise - noise.min()) / (noise.max() - noise.min()) * 255
    return noise.astype(np.uint8)

def resize_and_center_crop(img, size):
    """
    Resize and center crop an image to the specified size.
    Parameters:
        img (np.ndarray): Input image.
        size (int): Desired size for the output image.
    Returns:
        np.ndarray: Resized and cropped image.
    """
    h, w, _ = img.shape
    scale = size / min(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized_img = cv2.resize(img, (new_w, new_h))
    
    start_x = (new_w - size) // 2
    start_y = (new_h - size) // 2
    cropped_img = resized_img[start_y:start_y + size, start_x:start_x + size]
    
    return cropped_img

def resize(img, size):
    """
    Resize an image to the specified size.
    Parameters:
        img (np.ndarray): Input image.
        size (int): Desired size for the output image.
    Returns:
        np.ndarray: Resized image.
    """
    return cv2.resize(img, (size, size))

def horizontal_filp(img):
    """
    Flip an image horizontally.
    Parameters:
        img (np.ndarray): Input image.
    Returns:
        np.ndarray: Horizontally flipped image.
    """
    return cv2.flip(img, 1)

def normalize_image(image):
    """
    Normalize an image to the range [0, 1].
    Parameters:
        image (np.ndarray): Input image.
    Returns:
        np.ndarray: Normalized image.
    """
    return (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-6)