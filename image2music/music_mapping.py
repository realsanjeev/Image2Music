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

def hues_dataframe(hues: Sequence[int], scale_freqs: List[float]) -> pd.DataFrame:
    """
    Create a pandas DataFrame with hues and their corresponding frequencies.

    Parameters
    ----------
    hues : Sequence[int]
        List or array of hue values.
    scale_freqs : List[float]
        Frequencies for the chosen musical scale.

    Returns
    -------
    pd.DataFrame
        DataFrame with 'hues' and 'frequencies' columns.
    """
    if not scale_freqs:
        raise ValueError("scale_freqs must not be empty")
    logger.debug("Creating DataFrame for %d hues", len(hues))
    df = pd.DataFrame({"hues": hues})
    df["frequencies"] = df["hues"].apply(lambda h: hue2freq(h, scale_freqs))
    logger.info("Generated DataFrame with %d rows", len(df))
    return df
