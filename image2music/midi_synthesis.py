import numpy as np
import pandas as pd
from typing import Union

import librosa
import music21 as m21
from music21 import stream, note, tempo
from tqdm.auto import tqdm

from .logger import get_logger

logger = get_logger(__name__)


def frequencies_to_midi_stream(
    pixel_df: pd.DataFrame,
    bpm: int = 120,
    filter_repeats: bool = True
) -> stream.Stream:
    """
    Convert a DataFrame of pixel frequencies into a MIDI stream.

    Parameters
    ----------
    pixel_df : pd.DataFrame
        DataFrame with 'frequency', 'duration', and 'amplitude' columns.
    bpm : int
        Beats per minute for the MIDI file.
    filter_repeats : bool
        Whether to remove consecutive duplicate notes.

    Returns
    -------
    stream.Stream
        A music21 Stream containing the MIDI notes.
    """
    required_cols = ["frequency", "duration", "amplitude"]
    for col in required_cols:
        if col not in pixel_df.columns:
            raise ValueError(f"pixel_df must contain a '{col}' column.")
    
    logger.info("Converting frequencies to musical notes...")
    pixel_df = pixel_df.copy()
    pixel_df["notes"] = pixel_df["frequency"].apply(librosa.hz_to_note)
    
    # Map amplitude (0.1-1.0) to velocity (0-127)
    pixel_df["velocity"] = (pixel_df["amplitude"] * 127).astype(int)

    logger.info("Creating MIDI stream with %d notes...", len(pixel_df))
    midi_stream = stream.Stream()
    midi_stream.append(tempo.MetronomeMark(number=bpm))

    # Pre-calculate seconds per beat
    seconds_per_beat = 60.0 / bpm

    for _, row in tqdm(pixel_df.iterrows(), total=len(pixel_df), desc="Adding notes"):
        pitch = row["notes"].replace('♯', "#")
        midi_note = note.Note(pitch)
        
        # Duration to quarterLength
        midi_note.quarterLength = row["duration"] / seconds_per_beat
        
        # Velocity
        midi_note.volume.velocity = row["velocity"]
        
        midi_stream.append(midi_note)

    return midi_stream

def save_midi(midi_stream: stream.Stream, file_path: Union[str, bytes]) -> None:
    """
    Save a music21 Stream to a MIDI file.

    Parameters
    ----------
    midi_stream : stream.Stream
        The music21 stream to save.
    file_path : str or bytes
        Path to save the MIDI file.
    """
    midi_stream.write("midi", fp=file_path)
    logger.info("Saved MIDI file: %s", file_path)
