#!/usr/bin/env python3
"""
rule_based_classifier.py - Classify microscope images as amorphous or crystalline using rule-based classical image processing.
"""

import numpy as np
import os
from skimage import io, color, exposure, feature, filters
import matplotlib.pyplot as plt

# Hardcoded image path (modify as needed)
image_paths = ["12.tif", "14.tif", "15.tif", "16.tif", "17.tif", "18.tif", "crystal1.tif", "crystal2.png"]

def extract_features(image_path):
    global color_var, edge_density
    image = io.imread(image_path)

    if image.ndim == 3 and image.shape[2] == 4:
        image = image[:, :, :3]

    if image.ndim == 3:
        gray = color.rgb2gray(image)
    else:
        gray = image / 255.0 if image.max() > 1 else image

    # === Background Masking ===
    mask = gray > 0.05

    # === Illumination Normalization ===
    background = filters.gaussian(gray, sigma=25)
    gray_norm = gray - background
    gray_norm = np.clip(gray_norm, 0, 1)

    # Apply histogram equalization
    gray_eq = exposure.equalize_hist(gray_norm)

    # Feature 1: Color variance (only over masked regions)
    if image.ndim == 3:
        color_var = np.var(image[mask])
    else:
        color_var = 0

    # Feature 2: Edge density (only in masked area)
    edges = feature.canny(gray_eq, sigma=1.0)
    edge_density = np.sum(edges & mask) / np.sum(mask)

    # Scoring based on updated thresholds
    score = 0
    if color_var > 500:
        score += 1
    if edge_density < 0.1:
        score += 1

    label = "crystalline" if score >= 1 else "amorphous"

    return label, image, edges

def visualize(image_path, image, edges, label):
    global color_var, edge_density
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.suptitle(
        f"Classification: {label} | Color Var: {color_var:.2f}, Edge Density: {edge_density:.4f}",
        fontsize=14
    )

    # Original image
    axes[0].imshow(image, cmap='gray')
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    # Edges
    edge_overlay = image.copy()
    if edge_overlay.ndim == 2:
        edge_overlay = np.stack([edge_overlay]*3, axis=2)
    edge_overlay[edges] = [255, 0, 0]
    axes[1].imshow(edge_overlay)
    axes[1].set_title("Edges (Red Overlay)")
    axes[1].axis('off')

    # Histogram
    axes[2].set_title("Color Histogram")
    if image.ndim == 3:
        for i, color_name in enumerate(['r', 'g', 'b']):
            hist, bins = np.histogram(image[:, :, i].ravel(), bins=256, range=(0, 255))
            hist = hist / hist.sum()
            axes[2].plot(bins[:-1], hist, color=color_name)
    else:
        hist, bins = np.histogram(image.ravel(), bins=256, range=(0, 255))
        hist = hist / hist.sum()
        axes[2].plot(bins[:-1], hist, color='black')

    plt.tight_layout()
    plt.show()

def main():
    for image_path in image_paths:
        label, image, edges = extract_features(image_path)
        print(f"{os.path.basename(image_path)} => Classified as: {label}")
        visualize(image_path, image, edges, label)

if __name__ == "__main__":
    main()
