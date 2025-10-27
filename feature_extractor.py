"""
Feature Extraction Module
Provides feature extraction functionality with multiple algorithm flavors.
"""
from typing import List
import torch
import numpy as np


def feature_extractor(feature_extraction_args, aligned_faces: List[torch.Tensor]):
    """
    Extracts LBP features from aligned face tensors.

    Args:
        feature_extraction_args: Namespace with feature extraction flags (lbp, facenet)
        aligned_faces: List of aligned face tensors from face detection

    Returns:
        List of LBP feature vectors (histograms)
    """
    descriptors = []

    for face_tensor in aligned_faces:
        # Convert tensor to grayscale numpy array
        face_np = face_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)

        # Convert RGB to grayscale
        if len(face_np.shape) == 3:
            gray = 0.299 * face_np[:, :, 0] + 0.587 * face_np[:, :, 1] + 0.114 * face_np[:, :, 2]
            gray = gray.astype(np.uint8)
        else:
            gray = face_np

        if feature_extraction_args.lbp:
            # Compute uniform LBP descriptor
            lbp_descriptor = compute_lbp_descriptor(gray, grid_x=8, grid_y=8, radius=1)
            descriptors.append(lbp_descriptor)
        elif feature_extraction_args.facenet:
            facenet_descriptor = feature_extraction_facenet(gray)
            descriptors.append(facenet_descriptor)

    return descriptors


def compute_lbp_descriptor(image: np.ndarray, grid_x: int = 8, grid_y: int = 8, radius: int = 1):
    """
    Computes uniform LBP descriptor for an image.

    Args:
        image: Grayscale image as numpy array
        grid_x: Number of grid cells horizontally
        grid_y: Number of grid cells vertically
        radius: Radius for LBP computation

    Returns:
        Concatenated histogram feature vector
    """
    height, width = image.shape

    # Compute LBP image
    lbp_image = compute_uniform_lbp(image, radius, n_points=8)

    # Divide into grid and compute histograms
    cell_height = height // grid_y
    cell_width = width // grid_x

    histograms = []

    for i in range(grid_y):
        for j in range(grid_x):
            # Extract cell
            y_start = i * cell_height
            y_end = (i + 1) * cell_height if i < grid_y - 1 else height
            x_start = j * cell_width
            x_end = (j + 1) * cell_width if j < grid_x - 1 else width

            cell = lbp_image[y_start:y_end, x_start:x_end]

            # Compute histogram for this cell (59 bins for uniform LBP)
            hist, _ = np.histogram(cell, bins=59, range=(0, 59))

            # Normalize histogram
            hist = hist.astype(float)
            hist_sum = np.sum(hist)
            if hist_sum > 0:
                hist = hist / hist_sum

            histograms.append(hist)

    # Concatenate all histograms
    descriptor = np.concatenate(histograms)

    return descriptor


def compute_uniform_lbp(image: np.ndarray, radius: int, n_points: int):
    """
    Computes uniform Local Binary Pattern.

    Args:
        image: Grayscale image
        radius: Radius of circle
        n_points: Number of sampling points (8 for standard LBP)

    Returns:
        LBP image where uniform patterns are mapped to [0, 58] and non-uniform to 59
    """
    height, width = image.shape
    lbp_image = np.zeros((height, width), dtype=np.uint8)

    # Precompute uniform pattern lookup table
    uniform_map = get_uniform_pattern_mapping(n_points)

    # Compute sampling points on circle
    angles = 2 * np.pi * np.arange(n_points) / n_points
    sample_points = np.array([
        -radius * np.sin(angles),  # y coordinates
        radius * np.cos(angles)    # x coordinates
    ])

    # Process each pixel
    for y in range(radius, height - radius):
        for x in range(radius, width - radius):
            center = image[y, x]

            # Sample neighbors using bilinear interpolation
            binary_pattern = 0
            for i in range(n_points):
                # Calculate neighbor coordinates
                ny = y + sample_points[0, i]
                nx = x + sample_points[1, i]

                # Bilinear interpolation
                neighbor_value = bilinear_interpolate(image, nx, ny)

                # Compare with center and build binary pattern
                if neighbor_value >= center:
                    binary_pattern |= (1 << i)

            # Map to uniform pattern
            lbp_image[y, x] = uniform_map[binary_pattern]

    return lbp_image


def bilinear_interpolate(image: np.ndarray, x: float, y: float) -> float:
    """
    Performs bilinear interpolation at fractional coordinates.

    Args:
        image: Input image
        x: X coordinate (can be fractional)
        y: Y coordinate (can be fractional)

    Returns:
        Interpolated pixel value
    """
    x0 = int(np.floor(x))
    x1 = x0 + 1
    y0 = int(np.floor(y))
    y1 = y0 + 1

    # Get the four surrounding pixels
    Ia = image[y0, x0]
    Ib = image[y1, x0]
    Ic = image[y0, x1]
    Id = image[y1, x1]

    # Compute weights
    wa = (x1 - x) * (y1 - y)
    wb = (x1 - x) * (y - y0)
    wc = (x - x0) * (y1 - y)
    wd = (x - x0) * (y - y0)

    return wa * Ia + wb * Ib + wc * Ic + wd * Id


def get_uniform_pattern_mapping(n_points: int) -> np.ndarray:
    """
    Creates mapping from binary patterns to uniform LBP codes.

    Uniform patterns have at most 2 bitwise transitions (0->1 or 1->0).
    These are mapped to [0, n_points * (n_points - 1) + 2], non-uniform to last bin.

    Args:
        n_points: Number of sampling points

    Returns:
        Lookup table mapping binary patterns to uniform codes
    """
    # Total possible patterns
    n_patterns = 2 ** n_points
    mapping = np.zeros(n_patterns, dtype=np.uint8)

    uniform_code = 0

    for i in range(n_patterns):
        # Count transitions in binary pattern
        transitions = 0
        binary_str = format(i, f'0{n_points}b')

        for j in range(n_points):
            if binary_str[j] != binary_str[(j + 1) % n_points]:
                transitions += 1

        # Uniform patterns have at most 2 transitions
        if transitions <= 2:
            mapping[i] = uniform_code
            uniform_code += 1
        else:
            # Non-uniform pattern mapped to last bin
            mapping[i] = n_points * (n_points - 1) + 3

    return mapping



def feature_extraction_facenet(aligned_faces):
    """FaceNet feature extraction stub."""
    #TODO: Add face_recognition implementation here. see: https://pypi.org/project/face-recognition/
    print("Running the facenet flavor")
    return aligned_faces