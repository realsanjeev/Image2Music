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
    logger.info("Converting frequencies to musical notes...")
    pixel_df = pixel_df.copy()
    
    # Pre-calculate seconds per beat
    seconds_per_beat = 60.0 / bpm

    logger.info("Creating MIDI stream with %d events...", len(pixel_df))
    midi_stream = stream.Stream()
    midi_stream.append(tempo.MetronomeMark(number=bpm))

    for _, row in tqdm(pixel_df.iterrows(), total=len(pixel_df), desc="Adding notes"):
        freq = row["frequency"]
        duration = row["duration"]
        amplitude = row["amplitude"]
        
        # Calculate velocity (60-127)
        velocity = int(60 + amplitude * 67)
        quarter_length = duration / seconds_per_beat
        
        # Handle Chords (List of frequencies)
        if isinstance(freq, list):
            # Filter out 0 or invalid frequencies
            valid_freqs = [f for f in freq if f > 0]
            if not valid_freqs:
                continue
                
            # Convert Hz to Note Names
            pitches = [librosa.hz_to_note(f).replace('♯', '#') for f in valid_freqs]
            
            # Create Chord
            m21_chord = m21.chord.Chord(pitches)
            m21_chord.quarterLength = quarter_length
            m21_chord.volume.velocity = velocity
            midi_stream.append(m21_chord)
            
        # Handle Single Note (Float/Int)
        elif freq > 0:
            pitch = librosa.hz_to_note(freq).replace('♯', '#')
            m21_note = note.Note(pitch)
            m21_note.quarterLength = quarter_length
            m21_note.volume.velocity = velocity
            midi_stream.append(m21_note)
            
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
