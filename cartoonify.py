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
import tkinter as tk
from tkinter import filedialog, messagebox

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

def color_quantization(img, k=8):
    """
    Reduce the number of colors in the image using k-means clustering for a cartoon effect.
    :param img: Input color image
    :param k: Number of color clusters
    :return: Quantized image
    """
    data = np.float32(img).reshape((-1, 3))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.001)
    _, labels, centers = cv2.kmeans(data, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    centers = np.uint8(centers)
    quantized = centers[labels.flatten()]
    return quantized.reshape(img.shape)

def enhance_edges(img):
    """
    Create strong cartoon edges by combining adaptive threshold and Canny edge detection.
    :param img: Input color image
    :return: Edge mask (inverted for cartoon overlay)
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.medianBlur(gray, 7)
    # Adaptive threshold for bold edges
    edges_adapt = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY,
        blockSize=9,
        C=2
    )
    # Canny for fine details
    edges_canny = cv2.Canny(blurred, 100, 200)
    # Combine both
    edges = cv2.bitwise_or(edges_adapt, edges_canny)
    # Invert for mask
    edges = cv2.bitwise_not(edges)
    return edges

def smooth_image(img):
    """
    Apply bilateral filter multiple times for extra smooth, paint-like regions.
    :param img: Input color image
    :return: Smoothed image
    """
    temp = img.copy()
    for _ in range(2):
        temp = cv2.bilateralFilter(temp, d=9, sigmaColor=200, sigmaSpace=200)
    return temp

def cartoonify_image(img):
    """
    Full cartoonification pipeline: color quantization, smoothing, edge enhancement, and overlay.
    :param img: Input color image
    :return: Cartoonified image
    """
    quantized = color_quantization(img, k=8)
    smoothed = smooth_image(quantized)
    edges = enhance_edges(img)
    # Convert edges to 3 channels for overlay
    edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    cartoon = cv2.bitwise_and(smoothed, edges_colored)
    return cartoon

def display_and_save(original, cartoon, output_path):
    """
    Display original and cartoonified images side by side and save the cartoon image.
    :param original: Original image
    :param cartoon: Cartoonified image
    :param output_path: Path to save the cartoonified image
    """
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

# --- GUI Section ---
def select_and_cartoonify():
    file_path = filedialog.askopenfilename(
        title="Select an Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp *.tiff")]
    )
    if not file_path:
        return
    try:
        img = read_image(file_path)
        cartoon = cartoonify_image(img)
        base, ext = os.path.splitext(file_path)
        output_path = base + "_cartoonified" + ext
        display_and_save(img, cartoon, output_path)
        messagebox.showinfo("Success", f"Cartoonified image saved as:\n{output_path}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

def main():
    root = tk.Tk()
    root.title("Cartoonify an Image - Fun Cartoon Effect!")
    root.geometry("370x160")
    root.resizable(False, False)
    label = tk.Label(root, text="Cartoonify any photo!\nClick below to select an image.", font=("Arial", 12), pady=20)
    label.pack()
    btn = tk.Button(root, text="Select Image", font=("Arial", 12, "bold"), command=select_and_cartoonify, bg="#4CAF50", fg="white", padx=20, pady=10)
    btn.pack()
    root.mainloop()

if __name__ == "__main__":
    main()
