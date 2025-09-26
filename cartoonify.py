import sys
import os

# --- Utility: Ensure required libraries are installed ---
def install_and_import(package, import_name=None):
    """
    Installs the given package via pip if not already installed, then imports it.
    :param package: Name for pip install
    :param import_name: Name to import (if different)
    """
    try:
        __import__(import_name or package)
    except ImportError:
        print(f"Installing {package}... Please wait.")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"{package} installed successfully.")

# Ensure OpenCV and NumPy are available
install_and_import("opencv-python", "cv2")
install_and_import("numpy")

import cv2
import numpy as np

# --- Cartoonify Functions ---
def read_image(image_path):
    """
    Reads an image from the given file path.
    :param image_path: Path to the image file
    :return: Image as a NumPy array
    """
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"Could not read the image: {image_path}")
    return img

def convert_to_grayscale(img):
    """
    Converts a BGR image to grayscale.
    :param img: Input color image
    :return: Grayscale image
    """
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def apply_edge_detection(gray_img):
    """
    Detects edges using adaptive thresholding after median blur.
    :param gray_img: Grayscale image
    :return: Edge mask
    """
    blurred = cv2.medianBlur(gray_img, 5)
    edges = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        blockSize=9,
        C=9
    )
    return edges

def apply_bilateral_filter(img):
    """
    Smooths the image while preserving edges using bilateral filtering.
    :param img: Input color image
    :return: Smoothed image
    """
    return cv2.bilateralFilter(img, d=9, sigmaColor=250, sigmaSpace=250)

def cartoonify_image(img):
    """
    Converts a normal image to a cartoon-style image.
    :param img: Input color image
    :return: Cartoonified image
    """
    gray = convert_to_grayscale(img)
    edges = apply_edge_detection(gray)
    color = apply_bilateral_filter(img)
    cartoon = cv2.bitwise_and(color, color, mask=edges)
    return cartoon

def display_and_save(original, cartoon, output_path):
    """
    Displays the original and cartoonified images side by side and saves the cartoon image.
    :param original: Original image
    :param cartoon: Cartoonified image
    :param output_path: Path to save the cartoonified image
    """
    # Resize images for side-by-side display if too large
    max_width = 800
    scale = min(1.0, max_width / max(original.shape[1], cartoon.shape[1]))
    if scale < 1.0:
        original = cv2.resize(original, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        cartoon = cv2.resize(cartoon, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    combined = np.hstack((original, cartoon))
    cv2.imshow('Original (Left) vs Cartoonified (Right)', combined)
    print("Press any key in the image window to close...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(output_path, cartoon)
    print(f"\nCartoonified image saved as: {output_path}\n")

def main():
    print("""
==============================
   Cartoonify an Image Tool
==============================
""")
    image_path = input("Enter the path to your image file (.jpg, .png, etc.): ").strip('"')
    if not os.path.isfile(image_path):
        print("\n[Error] File does not exist. Please check the path and try again.\n")
        return
    try:
        img = read_image(image_path)
        cartoon = cartoonify_image(img)
        base, ext = os.path.splitext(image_path)
        output_path = base + "_cartoonified" + ext
        display_and_save(img, cartoon, output_path)
    except Exception as e:
        print(f"\n[Error] {e}\n")

if __name__ == "__main__":
    main()
