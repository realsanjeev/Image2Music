from typing import List, Sequence, Tuple
import pandas as pd
import numpy as np
import cv2
from sklearn.cluster import KMeans

from .logger import get_logger

logger = get_logger(__name__)


def extract_dominant_colors_kmeans(
    image_path: str,
    n_clusters: int,
    color_space: str = 'lch',
    sample_size: int = 10000
) -> Tuple[np.ndarray, KMeans]:
    """
    Extract dominant colors using K-means clustering.
    
    Parameters
    ----------
    image_path : str
        Path to the image file
    n_clusters : int
        Number of clusters (dominant colors) to extract
    color_space : str
        Color space for clustering: 'hsv', 'lab', 'lch'
    sample_size : int
        Maximum number of pixels to use for clustering (for performance)
        
    Returns
    -------
    cluster_centers : np.ndarray
        Cluster centers in color space (n_clusters, 3)
    kmeans : KMeans
        Fitted K-means model for prediction
    """
    from .image_utils import load_image, lab_to_lch
    
    # Load full-resolution image in specified color space
    # Use larger size for better color representation
    img = load_image(image_path, size=(100, 100), color_space=color_space)
    
    # Flatten to (N, 3) array
    pixels = img.reshape(-1, 3)
    
    # Subsample for performance if needed
    if len(pixels) > sample_size:
        indices = np.random.choice(len(pixels), sample_size, replace=False)
        pixels = pixels[indices]
    
    logger.info("Clustering %d pixels into %d dominant colors...", len(pixels), n_clusters)
    
    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(pixels)
    
    logger.info("Extracted %d dominant colors", n_clusters)
    
    return kmeans.cluster_centers_, kmeans


def map_clusters_to_frequencies(
    cluster_centers: np.ndarray,
    scale_freqs: List[float],
    color_space: str = 'lch'
) -> dict:
    """
    Map cluster centers to scale frequencies.
    
    Sort clusters by hue and assign to scale degrees in order.
    
    Parameters
    ----------
    cluster_centers : np.ndarray
        Cluster centers (n_clusters, 3)
    scale_freqs : List[float]
        Scale frequencies
    color_space : str
        Color space of cluster centers
        
    Returns
    -------
    dict
        Mapping from cluster index to frequency
    """
    if color_space == 'lch':
        # Sort by hue (channel 2)
        hues = cluster_centers[:, 2]
        sorted_indices = np.argsort(hues)
    elif color_space == 'hsv':
        # Sort by hue (channel 0)
        hues = cluster_centers[:, 0]
        sorted_indices = np.argsort(hues)
    elif color_space == 'lab':
        # Compute hue from a/b
        a = cluster_centers[:, 1]
        b = cluster_centers[:, 2]
        hues = np.arctan2(b, a) * 180 / np.pi
        hues = (hues + 360) % 360
        sorted_indices = np.argsort(hues)
    else:
        sorted_indices = np.arange(len(cluster_centers))
    
    # Map sorted clusters to scale frequencies
    cluster_to_freq = {}
    for i, cluster_idx in enumerate(sorted_indices):
        cluster_to_freq[cluster_idx] = scale_freqs[i % len(scale_freqs)]
    
    logger.debug("Mapped %d clusters to %d scale frequencies", len(cluster_centers), len(scale_freqs))
    
    return cluster_to_freq


def assign_pixels_to_clusters(
    pixel_data: dict,
    kmeans: KMeans,
    cluster_to_freq: dict,
    color_space: str = 'lch'
) -> np.ndarray:
    """
    Assign each pixel to its nearest cluster and return frequencies.
    
    Parameters
    ----------
    pixel_data : dict
        Pixel data dictionary with color channels
    kmeans : KMeans
        Fitted K-means model
    cluster_to_freq : dict
        Mapping from cluster index to frequency
    color_space : str
        Color space
        
    Returns
    -------
    np.ndarray
        Frequencies for each pixel
    """
    # Reconstruct pixel array from dict
    if color_space == 'hsv':
        pixels = np.column_stack([
            pixel_data['hue'],
            pixel_data['saturation'],
            pixel_data['value']
        ])
    elif color_space == 'lch':
        pixels = np.column_stack([
            pixel_data['lightness'],
            pixel_data['chroma'],
            pixel_data['hue']
        ])
    elif color_space == 'lab':
        pixels = np.column_stack([
            pixel_data['lightness'],
            pixel_data['a'],
            pixel_data['b']
        ])
    else:
        raise ValueError(f"Unsupported color space: {color_space}")
    
    # Predict cluster for each pixel
    cluster_labels = kmeans.predict(pixels)
    
    # Map to frequencies
    frequencies = np.array([cluster_to_freq[label] for label in cluster_labels])
    
    logger.debug("Assigned %d pixels to clusters", len(frequencies))
    
    return frequencies


def hue2freq(hue: float, scale_freqs: Sequence[float], color_space: str = 'hsv') -> float:
    """
    Map a hue value to a frequency in the given musical scale.

    Parameters
    ----------
    hue : float
        Hue value (HSV: 0-179, LCH: 0-360, LAB: computed angle)
    scale_freqs : Sequence[float]
        Sequence of frequencies for the scale.
    color_space : str
        Color space: 'hsv', 'lab', 'lch'

    Returns
    -------
    float
        The frequency corresponding to the hue value.
    """
    if not scale_freqs:
        raise ValueError("scale_freqs must not be empty")
    
    # Determine hue range based on color space
    if color_space == 'hsv':
        max_hue = 180  # OpenCV HSV hue is 0-179
        if not 0 <= hue <= 255:
            logger.warning(f"Hue value {hue} out of range, returning default frequency")
            return scale_freqs[0]
    elif color_space in ['lch', 'lab']:
        max_hue = 360  # LCH/LAB hue is 0-360 degrees
        if not 0 <= hue <= 360:
            logger.warning(f"Hue value {hue} out of range [0, 360], returning default frequency")
            return scale_freqs[0]
    else:
        raise ValueError(f"Unsupported color space: {color_space}")
    
    num_notes = len(scale_freqs)
    index = int(hue / max_hue * num_notes)
    index = min(index, num_notes - 1)
    
    return scale_freqs[index]

def hues_to_frequencies(hues: Sequence[int], scale_freqs: List[float]) -> np.ndarray:
    """
    Convert a sequence of hue values into an array of frequencies.

    Parameters
    ----------
    hues : Sequence[int]
        Sequence of hue values (0-255).
    scale_freqs : Sequence[float]
        Frequencies for the chosen musical scale.

    Returns
    -------
    np.ndarray
        Array of mapped frequencies.
    """
    if not scale_freqs:
        raise ValueError("scale_freqs must not be empty")
    logger.debug("Mapping %d hues to frequencies...", len(hues))
    freqs = [hue2freq(h, scale_freqs) for h in hues]
    freqs_array = np.array(freqs, dtype=float)
    logger.info("Converted hues to frequencies array of shape %s", freqs_array.shape)
    return freqs_array

def map_to_amplitude(value: float, color_space: str = 'hsv') -> float:
    """
    Map color channel to amplitude (0.1-1.0).
    
    HSV: Saturation (0-255)
    LAB/LCH: Lightness (0-100)
    """
    if color_space == 'hsv':
        # Saturation 0-255
        norm = value / 255.0
    elif color_space in ['lab', 'lch']:
        # Lightness 0-100
        norm = value / 100.0
    else:
        norm = 0.5
    
    return 0.1 + (norm * 0.9)

def map_to_duration(value: float, base_duration: float, color_space: str = 'hsv') -> float:
    """
    Map color channel to duration multiplier (0.5x - 2.0x).
    
    HSV: Value (0-255)
    LCH: Chroma (0-100+)
    LAB: Computed chroma from a/b
    """
    if color_space == 'hsv':
        # Value 0-255
        norm = value / 255.0
    elif color_space == 'lch':
        # Chroma 0-100+ (can exceed, so clamp)
        norm = min(value / 100.0, 1.0)
    elif color_space == 'lab':
        # Compute chroma from a/b (already done in extract_pixel_data)
        # Assume value is already normalized
        norm = min(value / 100.0, 1.0)
    else:
        norm = 0.5
    
    multiplier = 0.5 + (norm * 1.5)
    return base_duration * multiplier

def hues_dataframe(
    pixel_data: dict, 
    scale_freqs: List[float], 
    base_duration: float = 0.1, 
    color_space: str = 'hsv',
    use_kmeans: bool = False,
    image_path: str = None
) -> pd.DataFrame:
    """
    Create a pandas DataFrame with pixel data and mapped musical properties.

    Parameters
    ----------
    pixel_data : dict
        Dictionary with color channel arrays (channel names depend on color_space).
    scale_freqs : List[float]
        Frequencies for the chosen musical scale.
    base_duration : float
        Base duration for notes.
    color_space : str
        Color space: 'hsv', 'lab', 'lch'
    use_kmeans : bool
        Use K-means clustering for perceptual pitch mapping
    image_path : str
        Path to image file (required if use_kmeans=True)

    Returns
    -------
    pd.DataFrame
        DataFrame with musical properties.
    """
    if not scale_freqs:
        raise ValueError("scale_freqs must not be empty")
    
    # Extract appropriate channels based on color space
    if color_space == 'hsv':
        hues = pixel_data['hue']
        amp_channel = pixel_data['saturation']
        dur_channel = pixel_data['value']
    elif color_space == 'lch':
        hues = pixel_data['hue']
        amp_channel = pixel_data['lightness']
        dur_channel = pixel_data['chroma']
    elif color_space == 'lab':
        # Compute hue from a/b
        a = pixel_data['a']
        b = pixel_data['b']
        hues = np.arctan2(b, a) * 180 / np.pi
        hues = (hues + 360) % 360
        amp_channel = pixel_data['lightness']
        # Compute chroma from a/b
        dur_channel = np.sqrt(a**2 + b**2)
    else:
        raise ValueError(f"Unsupported color space: {color_space}")
    
    logger.debug("Creating DataFrame for %d pixels", len(hues))
    df = pd.DataFrame({
        "hue": hues,
        "amp_channel": amp_channel,
        "dur_channel": dur_channel
    })
    
    # Frequency mapping: K-means or linear
    if use_kmeans:
        if image_path is None:
            raise ValueError("image_path required for K-means clustering")
        
        logger.info("Using K-means clustering for pitch mapping...")
        
        # Extract dominant colors
        cluster_centers, kmeans = extract_dominant_colors_kmeans(
            image_path, len(scale_freqs), color_space
        )
        
        # Map clusters to frequencies
        cluster_to_freq = map_clusters_to_frequencies(
            cluster_centers, scale_freqs, color_space
        )
        
        # Assign pixels to clusters
        frequencies = assign_pixels_to_clusters(
            pixel_data, kmeans, cluster_to_freq, color_space
        )
        
        df["frequency"] = frequencies
    else:
        # Original linear mapping
        df["frequency"] = df["hue"].apply(lambda h: hue2freq(h, scale_freqs, color_space))
    
    df["amplitude"] = df["amp_channel"].apply(lambda v: map_to_amplitude(v, color_space))
    df["duration"] = df["dur_channel"].apply(lambda v: map_to_duration(v, base_duration, color_space))
    
    logger.info("Generated DataFrame with %d rows", len(df))
    return df


def smooth_parameters(df: pd.DataFrame, window_size: int = 3) -> pd.DataFrame:
    """
    Apply moving average smoothing to musical parameters.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with 'frequency', 'amplitude', 'duration' columns
    window_size : int
        Window size for moving average (must be odd)
        
    Returns
    -------
    pd.DataFrame
        DataFrame with smoothed parameters
    """
    if window_size <= 1:
        return df
    
    df = df.copy()
    
    # Apply rolling mean
    df['frequency'] = df['frequency'].rolling(window=window_size, center=True, min_periods=1).mean()
    df['amplitude'] = df['amplitude'].rolling(window=window_size, center=True, min_periods=1).mean()
    df['duration'] = df['duration'].rolling(window=window_size, center=True, min_periods=1).mean()
    
    logger.info("Applied smoothing with window size %d", window_size)
    return df


def add_phrase_boundaries(df: pd.DataFrame, phrase_length: int = 8, rest_duration: float = 0.2) -> pd.DataFrame:
    """
    Insert rest notes (silent pauses) at phrase boundaries.
    
    Parameters
    ----------
    df : pd.DataFrame
        DataFrame with musical parameters
    phrase_length : int
        Number of notes per phrase (rest inserted after each phrase)
    rest_duration : float
        Duration of rest in seconds
        
    Returns
    -------
    pd.DataFrame
        DataFrame with rest notes inserted
    """
    if phrase_length <= 0:
        return df
    
    rows = []
    for i, row in enumerate(df.iterrows()):
        rows.append(row[1])
        
        # Insert rest after each phrase (except the last)
        if (i + 1) % phrase_length == 0 and (i + 1) < len(df):
            rest_row = row[1].copy()
            rest_row['frequency'] = 0.0  # Silence
            rest_row['amplitude'] = 0.0
            rest_row['duration'] = rest_duration
            rows.append(rest_row)
    
    result_df = pd.DataFrame(rows).reset_index(drop=True)
    logger.info("Added phrase boundaries every %d notes", phrase_length)
    return result_df
