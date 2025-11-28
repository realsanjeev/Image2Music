from typing import List, Tuple
import numpy as np

from .logger import get_logger

logger = get_logger(__name__)

def get_piano_notes() -> dict:
    """
    Generate a mapping from piano note names (A0 to C8) to their frequencies in Hz.

    Returns
    -------
    dict
        Dictionary mapping note names to frequencies.
    """
    chromatic_notes = ['C', 'c', 'D', 'd', 'E', 'F', 'f', 'G', 'g', 'A', 'a', 'B']
    base_freq = 440.0  # Frequency of A4
    keys = [note + str(oct) for oct in range(0, 9) for note in chromatic_notes]

    # actual piano range (A0 to C8)
    keys = keys[keys.index('A0'):keys.index('C8') + 1]

    # A4 is the 49th key on an 88-key piano, index = 48 (0-based)
    note_freqs = {
        key: base_freq * 2 ** ((i + 1 - 49) / 12) for i, key in enumerate(keys)
    }
    note_freqs[''] = 0.0  # Fallback for invalid notes
    return note_freqs

def make_scale(octave: int, key: str, scale: str, make_harmony: str = "U0") -> Tuple[List[float], List[float]]:
    """
    Generate frequencies and harmonies for a specified musical scale.

    Parameters
    ----------
    octave : int
        The octave number for the base key (e.g., 4 for 'C4').
    key : str
        The starting key of the scale (e.g., 'C', 'D', etc.). Use 'c' for C# (Db).
    scale : str
        The type of scale to generate. E.g., 'MAJOR', 'MINOR'.
    make_harmony : str
        Harmony interval to apply. E.g., 'P5', 'M3'.

    Returns
    -------
    Tuple[List[float], List[float]]
        Frequencies for the scale notes, and their harmonies.
    """
    note_freqs = get_piano_notes()

    # Define tones (white keys uppercase, black keys lowercase)
    chromatic_notes = ['C', 'c', 'D', 'd', 'E', 'F', 'f', 'G', 'g', 'A', 'a', 'B']
    if key not in chromatic_notes:
        logger.error("Invalid key: %s. Must be in chromatic scale.", key)
        return [], []
    
    # Rotate chromatic notes to start from the given key
    index = chromatic_notes.index(key)
    rotated_notes = chromatic_notes[index:] + chromatic_notes[:index]

    # Scale intervals defined as semitone offsets from the root
    scales_dict = {
        'HARMONIC_MINOR': [0, 2, 3, 5, 7, 8, 11],
        'MAJOR': [0, 2, 4, 5, 7, 9, 11],
        'MINOR': [0, 2, 3, 5, 7, 8, 10],
        'PENTATONIC': [0, 2, 4, 7, 9],
        'BLUES': [0, 2, 3, 4, 5, 7, 9, 10, 11],
    }
    if scale not in scales_dict:
        logger.error("Invalid scale: %s", scale)
        return [], []

    harmony_select = {
        "U0": 1,
        "M2": 9/8,
        "m3": 6/5,
        "M3": 5/4,
        "P4": 4/3,
        "P5": 3/2,
        "M6": 5/3,
        "m7": 9/5,
        "O8": 2
    }
    harmony_val = harmony_select.get(make_harmony, 1)

    freqs, harmony = [], []
    
    # Calculate the index where the chromatic scale wraps around
    # rotated_notes = chromatic_notes[index:] + chromatic_notes[:index]
    # The first part has length len(chromatic_notes) - index.
    # Any index >= wrap_index comes from the second part (chromatic_notes[:index]),
    # which means it belongs to the next octave.
    wrap_index = len(chromatic_notes) - index

    for i in scales_dict[scale]:
        current_octave = octave
        if i >= wrap_index:
            current_octave += 1
            
        note_name = rotated_notes[i] + str(current_octave)
        freq_to_add = note_freqs.get(note_name)
        if freq_to_add is not None:
            freqs.append(freq_to_add)
            harmony.append(freq_to_add * harmony_val)
        else:
            logger.warning("Note not found: %s", note_name)

    logger.info("Generated %d-note scale (%s) starting from %s%d",
                len(freqs), scale, key, octave)
    return freqs, harmony
