# image2music/audio_synthesis.py

import numpy as np
from typing import Sequence, NamedTuple
from scipy.io import wavfile
from pedalboard import Pedalboard, Reverb, Delay, Chorus, HighpassFilter

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


class ADSR(NamedTuple):
    attack: float = 0.05   # Seconds
    decay: float = 0.1     # Seconds
    sustain: float = 0.7   # Amplitude fraction (0.0-1.0)
    release: float = 0.1   # Seconds


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


def apply_adsr(wave: np.ndarray, duration: float, sample_rate: int, adsr: ADSR) -> np.ndarray:
    """
    Apply ADSR envelope to the waveform.
    """
    total_samples = len(wave)
    attack_samples = int(adsr.attack * sample_rate)
    decay_samples = int(adsr.decay * sample_rate)
    release_samples = int(adsr.release * sample_rate)
    
    # Check if total duration is enough for full ADSR
    if attack_samples + decay_samples + release_samples > total_samples:
        # Scale down proportionally
        scale = total_samples / (attack_samples + decay_samples + release_samples)
        attack_samples = int(attack_samples * scale)
        decay_samples = int(decay_samples * scale)
        release_samples = int(release_samples * scale)
    
    sustain_samples = total_samples - (attack_samples + decay_samples + release_samples)
    if sustain_samples < 0:
        sustain_samples = 0
        # Re-adjust release to fit
        release_samples = total_samples - (attack_samples + decay_samples)
    
    envelope = np.ones(total_samples)
    
    # Attack: 0 -> 1
    if attack_samples > 0:
        envelope[:attack_samples] = np.linspace(0, 1, attack_samples)
    
    # Decay: 1 -> Sustain
    if decay_samples > 0:
        start = attack_samples
        end = start + decay_samples
        envelope[start:end] = np.linspace(1, adsr.sustain, decay_samples)
    
    # Sustain: Hold (already 1s, need to scale to sustain level)
    if sustain_samples > 0:
        start = attack_samples + decay_samples
        end = start + sustain_samples
        envelope[start:end] = adsr.sustain
        
    # Release: Sustain -> 0
    if release_samples > 0:
        start = total_samples - release_samples
        envelope[start:] = np.linspace(adsr.sustain, 0, release_samples)
        
    return wave * envelope


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


def apply_effects(
    audio: np.ndarray, 
    sample_rate: int, 
    reverb_mix: float = 0.0, 
    delay_mix: float = 0.0,
    chorus_mix: float = 0.0
) -> np.ndarray:
    """
    Apply audio effects using pedalboard.
    """
    if reverb_mix <= 0 and delay_mix <= 0 and chorus_mix <= 0:
        return audio
        
    logger.info("Applying effects: Reverb=%.2f, Delay=%.2f, Chorus=%.2f", reverb_mix, delay_mix, chorus_mix)
    
    board = Pedalboard()
    if reverb_mix > 0:
        board.append(Reverb(room_size=0.5, wet_level=reverb_mix))
    if delay_mix > 0:
        board.append(Delay(delay_seconds=0.25, feedback=0.5, mix=delay_mix))
    if chorus_mix > 0:
        board.append(Chorus(mix=chorus_mix))
    
    # Run effects
    # Pedalboard expects float32
    effected = board(audio.astype(np.float32), sample_rate)
    return effected


def generate_song(
    frequencies: Sequence[float], 
    amplitudes: Sequence[float],
    durations: Sequence[float],
    sample_rate: int = 44100, 
    use_octaves: bool = True,
    instrument: str = 'rich',
    adsr: ADSR = None,
    reverb_mix: float = 0.0,
    delay_mix: float = 0.0,
    chorus_mix: float = 0.0
) -> np.ndarray:
    """
    Generate a song waveform from lists of frequencies, amplitudes, and durations.
    """
    song_parts = []
    octaves = np.array([0.5, 1, 2]) if use_octaves else np.array([1])
    
    # Default ADSR if not provided
    if adsr is None:
        adsr = ADSR(attack=0.01, decay=0.1, sustain=0.7, release=0.1)
    
    # Get harmonic profile
    harmonics = INSTRUMENTS.get(instrument, INSTRUMENTS['rich'])
    logger.info("Using instrument '%s' with harmonics: %s", instrument, harmonics)

    for freq, amp, dur in zip(frequencies, amplitudes, durations):
        # Handle rest notes (frequency = 0)
        # Check if freq is a list (chord) or single value
        if isinstance(freq, list):
            # Chord Mode
            chord_wave = np.zeros(int(dur * sample_rate))
            valid_notes = 0
            
            for f in freq:
                if f > 0:
                    octave = np.random.choice(octaves)
                    wave = generate_harmonic_wave(
                        f * octave, 
                        dur, 
                        sample_rate, 
                        amplitude=amp,
                        harmonics=harmonics
                    )
                    # Resize if needed (rounding errors)
                    if len(wave) > len(chord_wave):
                        wave = wave[:len(chord_wave)]
                    elif len(wave) < len(chord_wave):
                        chord_wave = chord_wave[:len(wave)]
                        
                    chord_wave += wave
                    valid_notes += 1
            
            # Normalize chord amplitude
            if valid_notes > 0:
                chord_wave /= valid_notes
            
            note_wave = apply_adsr(chord_wave, dur, sample_rate, adsr)
            
        elif freq == 0 or amp == 0:
            # Generate silence
            num_samples = int(dur * sample_rate)
            note_wave = np.zeros(num_samples)
        else:
            # Single Note Mode
            octave = np.random.choice(octaves)
            note_wave = generate_harmonic_wave(
                freq * octave, 
                dur, 
                sample_rate, 
                amplitude=amp,
                harmonics=harmonics
            )
            note_wave = apply_adsr(note_wave, dur, sample_rate, adsr)
        
        song_parts.append(note_wave)

    song = np.concatenate(song_parts)
    
    # Apply global effects
    song = apply_effects(song, sample_rate, reverb_mix, delay_mix, chorus_mix)
    
    logger.info("Generated song with %d notes", len(frequencies))
    return song


def save_wav(file_path: str, data: np.ndarray, sample_rate: int = 44100) -> None:
    """
    Save waveform as a WAV file.
    """
    wavfile.write(file_path, rate=sample_rate, data=data.astype(np.float32))
    logger.info("Saved WAV file: %s", file_path)