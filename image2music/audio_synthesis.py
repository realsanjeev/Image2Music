# image2music/audio_synthesis.py

import numpy as np
from scipy.io import wavfile
from typing import Sequence

from .logger import get_logger

logger = get_logger(__name__)


INSTRUMENTS = {
    'sine': [1.0],
    'organ': [1.0, 0.5, 0.25, 0.125, 0.06],
    'woodwind': [1.0, 0.0, 0.5, 0.0, 0.25],  # Odd harmonics
    'brass': [1.0, 0.8, 0.6, 0.5, 0.4, 0.3],
    'rich': [1.0, 0.5, 0.33, 0.25, 0.2, 0.16], # Sawtooth-like approximation
    'square': [1.0, 0.0, 0.33, 0.0, 0.2, 0.0, 0.14] # Square wave approximation
}


def generate_sine_wave(frequency: float, duration: float, sample_rate: int = 44100, amplitude: float = 0.5) -> np.ndarray:
    """
    Generate a sine wave for a given frequency and duration.
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    wave = amplitude * np.sin(2 * np.pi * frequency * t)
    return wave


def generate_harmonic_wave(
    frequency: float, 
    duration: float, 
    sample_rate: int, 
    amplitude: float,
    harmonics: Sequence[float]
) -> np.ndarray:
    """
    Generate a wave using additive synthesis (sum of harmonics).
    """
    t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
    mixed_wave = np.zeros_like(t)
    
    for i, h_amp in enumerate(harmonics):
        if h_amp > 0:
            harmonic_freq = frequency * (i + 1)
            # Avoid aliasing: don't generate frequencies above Nyquist limit
            if harmonic_freq < sample_rate / 2:
                mixed_wave += h_amp * np.sin(2 * np.pi * harmonic_freq * t)
    
    # Normalize to prevent clipping, then scale by target amplitude
    max_val = np.max(np.abs(mixed_wave))
    if max_val > 0:
        mixed_wave = mixed_wave / max_val * amplitude
        
    return mixed_wave


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


def generate_song(
    frequencies: Sequence[float], 
    amplitudes: Sequence[float],
    durations: Sequence[float],
    sample_rate: int = 44100, 
    use_octaves: bool = True,
    instrument: str = 'rich'
) -> np.ndarray:
    """
    Generate a song waveform from lists of frequencies, amplitudes, and durations.
    """
    song_parts = []
    octaves = np.array([0.5, 1, 2]) if use_octaves else np.array([1])
    
    # 10ms fade to prevent clicks
    fade_duration = 0.01 
    
    # Get harmonic profile
    harmonics = INSTRUMENTS.get(instrument, INSTRUMENTS['rich'])
    logger.info("Using instrument '%s' with harmonics: %s", instrument, harmonics)

    for freq, amp, dur in zip(frequencies, amplitudes, durations):
        # Handle rest notes (frequency = 0)
        if freq == 0 or amp == 0:
            # Generate silence
            num_samples = int(dur * sample_rate)
            note_wave = np.zeros(num_samples)
        else:
            octave = np.random.choice(octaves)
            note_wave = generate_harmonic_wave(
                freq * octave, 
                dur, 
                sample_rate, 
                amplitude=amp,
                harmonics=harmonics
            )
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