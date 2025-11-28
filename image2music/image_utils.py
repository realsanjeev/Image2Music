import cv2
import numpy as np
from typing import Union

def load_image(image: Union[str, np.ndarray], size: tuple = (26, 26)) -> np.ndarray:
    """
    Load an image and convert it to HSV format with specified size.

    Parameters
    ----------
    image : str or np.ndarray
        File path to image or raw image array (in BGR).
    size : tuple
        Size to resize the image to (width, height).

    Returns
    -------
    np.ndarray
        HSV image resized to specified shape.
    """
    if isinstance(image, str):
        img = cv2.imread(image)
        if img is None:
            raise FileNotFoundError(f"Image not found: {image}")
    elif isinstance(image, np.ndarray):
        img = image
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")

    # Resize and convert to HSV
    resized_img = cv2.resize(img, size)
    hsv_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2HSV)
    return hsv_img


def sample_grid(hsv_img: np.ndarray, step: int = 4) -> np.ndarray:
    """
    Sample pixels in a regular grid pattern.
    
    Parameters
    ----------
    hsv_img : np.ndarray
        HSV image of shape (H, W, 3)
    step : int
        Step size for grid sampling
        
    Returns
    -------
    np.ndarray
        Sampled pixels of shape (N, 3) where N is number of samples
    """
    h, w = hsv_img.shape[:2]
    samples = []
    for i in range(0, h, step):
        for j in range(0, w, step):
            samples.append(hsv_img[i, j])
    return np.array(samples)


def sample_spiral(hsv_img: np.ndarray, num_samples: int = 50) -> np.ndarray:
    """
    Sample pixels following a spiral pattern from center outward.
    
    Parameters
    ----------
    hsv_img : np.ndarray
        HSV image of shape (H, W, 3)
    num_samples : int
        Number of pixels to sample
        
    Returns
    -------
    np.ndarray
        Sampled pixels of shape (N, 3)
    """
    h, w = hsv_img.shape[:2]
    cy, cx = h // 2, w // 2
    
    # Generate spiral coordinates
    coords = []
    x, y = 0, 0
    dx, dy = 0, -1
    
    for _ in range(max(h, w) ** 2):
        if (-w//2 < x <= w//2) and (-h//2 < y <= h//2):
            coords.append((cy + y, cx + x))
        
        if x == y or (x < 0 and x == -y) or (x > 0 and x == 1-y):
            dx, dy = -dy, dx
        x, y = x + dx, y + dy
        
        if len(coords) >= num_samples:
            break
    
    # Sample pixels at spiral coordinates
    samples = []
    for y, x in coords[:num_samples]:
        if 0 <= y < h and 0 <= x < w:
            samples.append(hsv_img[y, x])
    
    return np.array(samples)


def sample_edges(hsv_img: np.ndarray, num_samples: int = 50) -> np.ndarray:
    """
    Sample pixels based on edge detection (more samples at edges).
    
    Parameters
    ----------
    hsv_img : np.ndarray
        HSV image of shape (H, W, 3)
    num_samples : int
        Number of pixels to sample
        
    Returns
    -------
    np.ndarray
        Sampled pixels of shape (N, 3)
    """
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    
    # Get edge coordinates
    edge_coords = np.argwhere(edges > 0)
    
    if len(edge_coords) == 0:
        # No edges found, fall back to grid sampling
        return sample_grid(hsv_img, step=max(1, hsv_img.shape[0] // int(np.sqrt(num_samples))))
    
    # Sample from edge coordinates
    if len(edge_coords) > num_samples:
        indices = np.random.choice(len(edge_coords), num_samples, replace=False)
        edge_coords = edge_coords[indices]
    
    samples = [hsv_img[y, x] for y, x in edge_coords]
    return np.array(samples)


def sample_weighted(hsv_img: np.ndarray, num_samples: int = 50) -> np.ndarray:
    """
    Sample pixels weighted by brightness and saturation.
    
    Parameters
    ----------
    hsv_img : np.ndarray
        HSV image of shape (H, W, 3)
    num_samples : int
        Number of pixels to sample
        
    Returns
    -------
    np.ndarray
        Sampled pixels of shape (N, 3)
    """
    h, w = hsv_img.shape[:2]
    
    # Calculate weights based on saturation and value
    s_channel = hsv_img[:, :, 1].astype(float)
    v_channel = hsv_img[:, :, 2].astype(float)
    weights = (s_channel / 255.0) * (v_channel / 255.0)
    weights = weights.flatten()
    
    # Normalize weights
    weights = weights / (weights.sum() + 1e-10)
    
    # Sample based on weights
    total_pixels = h * w
    indices = np.random.choice(total_pixels, size=min(num_samples, total_pixels), 
                               replace=False, p=weights)
    
    # Convert flat indices to 2D coordinates
    samples = []
    for idx in indices:
        y, x = divmod(idx, w)
        samples.append(hsv_img[y, x])
    
    return np.array(samples)


def extract_pixel_data(
    hsv_img: np.ndarray, 
    sampling_strategy: str = 'all',
    step: int = 4,
    num_samples: int = 50
) -> dict:
    """
    Extract Hue, Saturation, and Value channels from an HSV image with optional sampling.

    Parameters
    ----------
    hsv_img : np.ndarray
        HSV image of shape (H, W, 3)
    sampling_strategy : str
        Sampling strategy: 'all', 'grid', 'spiral', 'edges', 'weighted'
    step : int
        Step size for grid sampling
    num_samples : int
        Number of samples for non-grid strategies

    Returns
    -------
    dict
        Dictionary containing 'hue', 'saturation', and 'value' 1D arrays.
    """
    if sampling_strategy == 'grid':
        samples = sample_grid(hsv_img, step)
    elif sampling_strategy == 'spiral':
        samples = sample_spiral(hsv_img, num_samples)
    elif sampling_strategy == 'edges':
        samples = sample_edges(hsv_img, num_samples)
    elif sampling_strategy == 'weighted':
        samples = sample_weighted(hsv_img, num_samples)
    else:  # 'all'
        h, s, v = cv2.split(hsv_img)
        return {
            'hue': h.flatten(),
            'saturation': s.flatten(),
            'value': v.flatten()
        }
    
    # Extract channels from samples
    return {
        'hue': samples[:, 0],
        'saturation': samples[:, 1],
        'value': samples[:, 2]
    }


def extract_hues(hsv_img: np.ndarray) -> np.ndarray:
    """
    Extract and flatten the hue channel from an HSV image.
    (Deprecated: Use extract_pixel_data instead)
    """
    return extract_pixel_data(hsv_img)['hue']
