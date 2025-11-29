import cv2
import numpy as np
from typing import Union

def lab_to_lch(lab_img: np.ndarray) -> np.ndarray:
    """
    Convert LAB image to LCH (Lightness, Chroma, Hue) color space.
    
    Parameters
    ----------
    lab_img : np.ndarray
        LAB image from OpenCV (L: 0-255, A: 0-255, B: 0-255)
        
    Returns
    -------
    np.ndarray
        LCH image (L: 0-100, C: 0-100+, H: 0-360)
    """
    # OpenCV LAB ranges:
    # L: 0-255 (should be 0-100, so divide by 2.55)
    # A: 0-255 (should be -128 to 127, so subtract 128)
    # B: 0-255 (should be -128 to 127, so subtract 128)
    
    L = lab_img[:, :, 0] / 2.55  # Convert to 0-100
    A = lab_img[:, :, 1].astype(float) - 128  # Convert to -128 to 127
    B = lab_img[:, :, 2].astype(float) - 128  # Convert to -128 to 127
    
    # Calculate Chroma and Hue
    C = np.sqrt(A**2 + B**2)  # Chroma
    H = np.arctan2(B, A) * 180 / np.pi  # Hue in degrees
    H = (H + 360) % 360  # Ensure 0-360 range
    
    # Stack into LCH image
    lch_img = np.stack([L, C, H], axis=2)
    return lch_img


def load_image(image: Union[str, np.ndarray], size: tuple = (26, 26), color_space: str = 'hsv') -> np.ndarray:
    """
    Load an image and convert it to specified color space with specified size.

    Parameters
    ----------
    image : str or np.ndarray
        File path to image or raw image array (in BGR).
    size : tuple
        Size to resize the image to (width, height).
    color_space : str
        Color space to convert to: 'hsv', 'lab', or 'lch'

    Returns
    -------
    np.ndarray
        Image in specified color space, resized to specified shape.
    """
    if isinstance(image, str):
        img = cv2.imread(image)
        if img is None:
            raise FileNotFoundError(f"Image not found: {image}")
    elif isinstance(image, np.ndarray):
        img = image
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")

    # Resize
    resized_img = cv2.resize(img, size)
    
    # Convert to requested color space
    if color_space == 'hsv':
        converted_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2HSV)
    elif color_space == 'lab':
        converted_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2LAB)
    elif color_space == 'lch':
        lab_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2LAB)
        converted_img = lab_to_lch(lab_img)
    else:
        raise ValueError(f"Unsupported color space: {color_space}. Choose 'hsv', 'lab', or 'lch'.")
    
    return converted_img


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
    img: np.ndarray, 
    sampling_strategy: str = 'all',
    step: int = 4,
    num_samples: int = 50,
    color_space: str = 'hsv'
) -> dict:
    """
    Extract color channels from an image with optional sampling.

    Parameters
    ----------
    img : np.ndarray
        Image in specified color space (H, W, 3)
    sampling_strategy : str
        Sampling strategy: 'all', 'grid', 'spiral', 'edges', 'weighted'
    step : int
        Step size for grid sampling
    num_samples : int
        Number of samples for non-grid strategies
    color_space : str
        Color space of input image: 'hsv', 'lab', 'lch'

    Returns
    -------
    dict
        Dictionary containing color channel arrays with appropriate names.
        HSV: 'hue', 'saturation', 'value'
        LAB: 'lightness', 'a', 'b'
        LCH: 'lightness', 'chroma', 'hue'
    """
    if sampling_strategy == 'grid':
        samples = sample_grid(img, step)
    elif sampling_strategy == 'spiral':
        samples = sample_spiral(img, num_samples)
    elif sampling_strategy == 'edges':
        samples = sample_edges(img, num_samples)
    elif sampling_strategy == 'weighted':
        samples = sample_weighted(img, num_samples)
    else:  # 'all'
        ch1, ch2, ch3 = cv2.split(img)
        samples = None
    
    # Extract channels based on color space
    if samples is not None:
        ch1, ch2, ch3 = samples[:, 0], samples[:, 1], samples[:, 2]
    else:
        ch1, ch2, ch3 = ch1.flatten(), ch2.flatten(), ch3.flatten()
    
    # Return with appropriate channel names
    if color_space == 'hsv':
        return {
            'hue': ch1,
            'saturation': ch2,
            'value': ch3
        }
    elif color_space == 'lab':
        return {
            'lightness': ch1 / 2.55,  # Convert to 0-100
            'a': ch2.astype(float) - 128,  # Convert to -128 to 127
            'b': ch3.astype(float) - 128   # Convert to -128 to 127
        }
    elif color_space == 'lch':
        return {
            'lightness': ch1,  # Already 0-100
            'chroma': ch2,     # Already computed
            'hue': ch3         # Already 0-360
        }
    else:
        raise ValueError(f"Unsupported color space: {color_space}")


def extract_hues(hsv_img: np.ndarray) -> np.ndarray:
    """
    Extract and flatten the hue channel from an HSV image.
    (Deprecated: Use extract_pixel_data instead)
    """
    return extract_pixel_data(hsv_img)['hue']


def analyze_image_properties(img: np.ndarray, color_space: str = 'lch') -> dict:
    """
    Analyze global image properties to suggest musical parameters.
    
    Returns
    -------
    dict
        Dictionary with suggested 'bpm' and 'scale'.
    """
    # Ensure image is in LCH for analysis (easier to reason about)
    if color_space != 'lch':
        # This is a simplification; ideally we'd convert. 
        # But for now let's assume the input img matches the color_space arg.
        pass

    # Calculate average lightness/brightness
    if color_space == 'lch':
        l_channel = img[:, :, 0]
        avg_lightness = np.mean(l_channel)
    elif color_space == 'lab':
        l_channel = img[:, :, 0]
        avg_lightness = np.mean(l_channel)
    elif color_space == 'hsv':
        v_channel = img[:, :, 2]
        avg_lightness = np.mean(v_channel) / 2.55 # Normalize 0-100
    else:
        avg_lightness = 50.0

    # Map Lightness to BPM (Dark=60, Bright=180)
    # Linear mapping: BPM = 60 + (Lightness/100 * 120)
    bpm = int(60 + (avg_lightness / 100.0 * 120))
    bpm = max(60, min(180, bpm)) # Clamp

    # Calculate Color Temperature (Warm vs Cool) to suggest Scale
    # Warm (Red/Yellow) -> Major
    # Cool (Blue/Green) -> Minor
    
    scale = 'MAJOR' # Default
    
    if color_space == 'lch':
        h_channel = img[:, :, 2]
        # LCH Hue: 0=Red, 90=Yellow, 180=Green, 270=Blue
        # Warm: 0-135, 315-360
        # Cool: 135-315
        
        # We need to handle the circular nature of hue
        # Let's count pixels in warm vs cool ranges
        warm_mask = (h_channel < 135) | (h_channel > 315)
        cool_mask = (h_channel >= 135) & (h_channel <= 315)
        
        warm_pixels = np.sum(warm_mask)
        cool_pixels = np.sum(cool_mask)
        
        if cool_pixels > warm_pixels:
            scale = 'MINOR'
            
    elif color_space == 'hsv':
        h_channel = img[:, :, 0]
        # HSV Hue (0-180 in OpenCV usually, but here we might have 0-360 depending on loading)
        # Assuming 0-180 for OpenCV standard, or 0-360 if we converted.
        # Let's assume standard 0-360 for logic
        
        # Warm: 0-60 (Red-Yellow), 300-360 (Magenta-Red)
        # Cool: 60-300 (Green-Cyan-Blue-Purple)
        
        warm_mask = (h_channel < 60) | (h_channel > 300)
        cool_mask = (h_channel >= 60) & (h_channel <= 300)
        
        if np.sum(cool_mask) > np.sum(warm_mask):
            scale = 'MINOR'

    return {
        'bpm': bpm,
        'scale': scale
    }


def analyze_texture(img: np.ndarray, color_space: str = 'lch') -> float:
    """
    Analyze image texture (complexity) using local variance.
    
    Returns
    -------
    float
        Texture score (0.0 - 1.0). Higher = more complex/noisy.
    """
    # Use Lightness/Value channel
    if color_space == 'lch':
        channel = img[:, :, 0]
    elif color_space == 'lab':
        channel = img[:, :, 0]
    elif color_space == 'hsv':
        channel = img[:, :, 2]
    else:
        channel = img[:, :, 0] # Fallback
        
    # Calculate standard deviation of the channel
    # High std dev = high contrast/texture
    std_dev = np.std(channel)
    
    # Normalize (heuristic: std dev of 50 is very high for 0-100 range)
    texture_score = std_dev / 50.0
    return min(1.0, max(0.0, texture_score))
