from typing import List, Sequence
import pandas as pd
import numpy as np

from .logger import get_logger

logger = get_logger(__name__)


def hue2freq(hue: int, scale_freqs: Sequence[float]) -> float:
    """
    Map a hue value to a frequency in the given musical scale.

    Parameters
    ----------
    hue : int
        Hue value (0–255) from an HSV image.
    scale_freqs : Sequence[float]
        Sequence of frequencies for the scale (e.g., Harmonic Minor).

    Returns
    -------
    float
        The frequency corresponding to the hue value.
    """
    # thresholds = [26, 52, 78, 104, 128, 154, 180]
    # thresholds = [25, 50, 75, 101, 126, 151, 179]  # 7 thresholds for 0-179 hue range

    if not scale_freqs:
        raise ValueError("scale_freqs must not be empty")
    if not 0 <= hue <= 255:
        logger.warning(f"Hue value {hue} out of range [0, 255], returning default frequency")
        return scale_freqs[0]
    
    # Dynamic mapping based on the number of notes in the scale
    # We map the hue range [0, 180) to indices [0, len(scale_freqs))
    # Note: OpenCV hues are typically 0-179.
    
    num_notes = len(scale_freqs)
    
    # Calculate index proportionally
    # If hue is 179 and num_notes is 7: 179 / 180 * 7 = 6.96 -> 6
    index = int(hue / 180 * num_notes)
    
    # Clamp index to be safe (in case hue >= 180)
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

def map_saturation_to_amplitude(saturation: int) -> float:
    """
    Map saturation (0-255) to amplitude (0.1-1.0).
    """
    # Normalize 0-255 to 0-1
    norm = saturation / 255.0
    # Map to 0.1 - 1.0 range
    return 0.1 + (norm * 0.9)

def map_value_to_duration(value: int, base_duration: float) -> float:
    """
    Map value (0-255) to duration multiplier (0.5x - 2.0x).
    """
    # Normalize 0-255 to 0-1
    norm = value / 255.0
    # Map to 0.5 - 2.0 range
    multiplier = 0.5 + (norm * 1.5)
    return base_duration * multiplier

def hues_dataframe(pixel_data: dict, scale_freqs: List[float], base_duration: float = 0.1) -> pd.DataFrame:
    """
    Create a pandas DataFrame with pixel data and mapped musical properties.

    Parameters
    ----------
    pixel_data : dict
        Dictionary with 'hue', 'saturation', 'value' arrays.
    scale_freqs : List[float]
        Frequencies for the chosen musical scale.
    base_duration : float
        Base duration for notes.

    Returns
    -------
    pd.DataFrame
        DataFrame with musical properties.
    """
    if not scale_freqs:
        raise ValueError("scale_freqs must not be empty")
        
    hues = pixel_data['hue']
    sats = pixel_data['saturation']
    vals = pixel_data['value']
    
    logger.debug("Creating DataFrame for %d pixels", len(hues))
    df = pd.DataFrame({
        "hue": hues,
        "saturation": sats,
        "value": vals
    })
    
    df["frequency"] = df["hue"].apply(lambda h: hue2freq(h, scale_freqs))
    df["amplitude"] = df["saturation"].apply(map_saturation_to_amplitude)
    df["duration"] = df["value"].apply(lambda v: map_value_to_duration(v, base_duration))
    
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
