# image2music/audio_synthesis.py

import numpy as np
from scipy.io import wavfile
from typing import Sequence

from .logger import get_logger

logger = get_logger(__name__)


def generate_sine_wave(frequency: float, duration: float, sample_rate: int = 44100, amplitude: float = 0.5) -> np.ndarray:
    """
    Generate a sine wave for a given frequency and duration.
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = amplitude * np.sin(2 * np.pi * frequency * t)
    return wave


def apply_envelope(wave: np.ndarray, fade_duration: float, sample_rate: int) -> np.ndarray:
    """
    Apply a linear fade-in and fade-out envelope to the waveform.
    """
    fade_samples = int(fade_duration * sample_rate)
    if fade_samples * 2 > len(wave):
        fade_samples = len(wave) // 2
    
    # Fade in
    fade_in = np.linspace(0, 1, fade_samples)
    wave[:fade_samples] *= fade_in
    
    # Fade out
    fade_out = np.linspace(1, 0, fade_samples)
    wave[-fade_samples:] *= fade_out
    
    return wave


def generate_song(frequencies: Sequence[float], duration: float, sample_rate: int = 44100, use_octaves: bool = True) -> np.ndarray:
    """
    Generate a song waveform from a list of frequencies.
    """
    song_parts = []
    octaves = np.array([0.5, 1, 2]) if use_octaves else np.array([1])
    
    # 10ms fade to prevent clicks
    fade_duration = 0.01 

    for freq in frequencies:
        octave = np.random.choice(octaves)
        note_wave = generate_sine_wave(freq * octave, duration, sample_rate)
        note_wave = apply_envelope(note_wave, fade_duration, sample_rate)
        song_parts.append(note_wave)

    song = np.concatenate(song_parts)
    logger.info("Generated song with %d notes", len(frequencies))
    return song


def save_wav(file_path: str, data: np.ndarray, sample_rate: int = 44100) -> None:
    """
    Save waveform as a WAV file.
    """
    wavfile.write(file_path, rate=sample_rate, data=data.astype(np.float32))
    logger.info("Saved WAV file: %s", file_path)