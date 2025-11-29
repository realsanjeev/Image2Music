# image2music/pipeline.py

from pathlib import Path
import numpy as np

from .logger import get_logger
from .image_utils import load_image, extract_pixel_data
from .scales import make_scale
from .music_mapping import hues_dataframe, smooth_parameters, add_phrase_boundaries
from .audio_synthesis import generate_song, save_wav
from .midi_synthesis import frequencies_to_midi_stream, save_midi

logger = get_logger(__name__)


def convert_image_to_music(
    image_path: str,
    output_path: str,
    scale_name: str = "HARMONIC_MINOR",
    key: str = "A",
    octave: int = 3,
    duration_per_note: float = 0.1,
    sample_rate: int = 22050,
    use_octaves: bool = True,
    midi_output_path: str = None,
    bpm: int = 120,
    sampling_strategy: str = "grid",
    grid_step: int = 4,
    num_samples: int = 50,
    smooth_window: int = 3,
    phrase_length: int = 8,
    color_space: str = "lch",
    use_kmeans: bool = False,
    instrument: str = "rich",
    attack: float = 0.01,
    decay: float = 0.1,
    sustain: float = 0.7,
    release: float = 0.1,
    reverb: float = 0.0,
    delay: float = 0.0,
    chorus: float = 0.0,
    quantize: bool = False,
    use_chords: bool = False,
    auto_bpm: bool = False,
    auto_scale: bool = False,
    multi_track: bool = False,
    use_drums: bool = False
) -> None:
    """
    Convert an image into music (WAV + optional MIDI) by mapping pixel properties to musical parameters.

    Parameters
    ----------
    image_path : str
        Path to the image file.
    output_path : str
        Path to save the generated WAV file.
    midi_output_path : str
        Path to save the generated MIDI file.
    sampling_strategy : str
        Pixel sampling strategy: 'all', 'grid', 'spiral', 'edges', 'weighted'
    grid_step : int
        Step size for grid sampling
    num_samples : int
        Number of samples for non-grid strategies
    smooth_window : int
        Window size for parameter smoothing (0 = no smoothing)
    phrase_length : int
        Insert rest every N notes (0 = no phrases)
    color_space : str
        Color space for image analysis: 'hsv', 'lab', 'lch' (default: 'lch')
    use_kmeans : bool
        Use K-means clustering for perceptual pitch mapping
    instrument : str
        Instrument timbre: 'sine', 'organ', 'woodwind', 'brass', 'rich', 'square'
    attack : float
        ADSR Attack time (seconds)
    decay : float
        ADSR Decay time (seconds)
    sustain : float
        ADSR Sustain level (0.0-1.0)
    release : float
        ADSR Release time (seconds)
    reverb : float
        Reverb mix (0.0-1.0)
    delay : float
        Delay mix (0.0-1.0)
    chorus : float
        Chorus mix (0.0-1.0)
    quantize : bool
        Quantize durations to musical grid (1/16th notes)
    use_chords : bool
        Generate chords (triads) instead of single notes
    auto_bpm : bool
        Automatically detect BPM from image brightness
    auto_scale : bool
        Automatically detect Scale from image color temperature
    multi_track : bool
        Generate separate Bass and Melody tracks and mix them
    use_drums : bool
        Generate a percussion track based on image texture
    """
    from .audio_synthesis import ADSR, generate_drum_sound
    from .image_utils import analyze_image_properties, analyze_texture
    
    logger.info("Loading image: %s", image_path)
    img = load_image(image_path, color_space=color_space)

    # Auto-Analysis
    if auto_bpm or auto_scale:
        logger.info("Analyzing image for musical properties...")
        props = analyze_image_properties(img, color_space=color_space)
        
        if auto_bpm:
            bpm = props['bpm']
            logger.info("Auto-detected BPM: %d (from brightness)", bpm)
            
        if auto_scale:
            scale_name = props['scale']
            logger.info("Auto-detected Scale: %s (from color temperature)", scale_name)

    # Helper to generate a track
    def generate_track(
        track_img, 
        track_octave, 
        track_instrument, 
        track_quantize_grid, 
        track_chords,
        is_bass=False
    ):
        logger.info("Extracting pixel data for track...")
        pixel_data = extract_pixel_data(
            track_img, 
            sampling_strategy=sampling_strategy,
            step=grid_step,
            num_samples=num_samples,
            color_space=color_space
        )
        
        scale_freqs, _ = make_scale(octave=track_octave, key=key.lower(), scale=scale_name)

        df = hues_dataframe(
            pixel_data, 
            scale_freqs, 
            base_duration=duration_per_note, 
            color_space=color_space,
            use_kmeans=use_kmeans,
            image_path=image_path,
            quantize=quantize,
            quantize_grid=track_quantize_grid,
            bpm=bpm,
            use_chords=track_chords
        )
        
        if smooth_window > 1:
            df = smooth_parameters(df, window_size=smooth_window)
        
        if phrase_length > 0:
            df = add_phrase_boundaries(df, phrase_length=phrase_length)

        # Rhythmic Bass Logic
        if is_bass and quantize:
            # Force bass to play on beats (simplification)
            # We'll insert rests to create a groove
            # Pattern: Note - Rest - Note - Rest
            for i in range(len(df)):
                if i % 2 != 0:
                    df.at[i, 'frequency'] = 0 # Rest
                    
        track_adsr = ADSR(attack=attack, decay=decay, sustain=sustain, release=release)
        
        return generate_song(
            frequencies=df["frequency"].tolist(),
            amplitudes=df["amplitude"].tolist(),
            durations=df["duration"].tolist(),
            sample_rate=sample_rate, 
            use_octaves=use_octaves,
            instrument=track_instrument,
            adsr=track_adsr,
            reverb_mix=reverb,
            delay_mix=delay,
            chorus_mix=chorus
        ), df

    final_df = None
    drum_audio = None

    if use_drums:
        logger.info("Generating Percussion Track...")
        texture = analyze_texture(img, color_space=color_space)
        logger.info("Image Texture Score: %.2f", texture)
        
        # Determine drum pattern based on texture
        # Simple: Kick on 1, 3. Snare on 2, 4.
        # Complex: 16th notes
        
        seconds_per_beat = 60.0 / bpm
        total_beats = int(num_samples * duration_per_note / seconds_per_beat) # Approx
        if total_beats < 4: total_beats = 16 # Min length
        
        drum_track_len = int(total_beats * seconds_per_beat * sample_rate)
        drum_audio = np.zeros(drum_track_len)
        
        # Basic Beat (Kick/Snare)
        for beat in range(total_beats):
            pos = int(beat * seconds_per_beat * sample_rate)
            if pos >= len(drum_audio): break
            
            # Kick on 0, 2 (1 and 3)
            if beat % 2 == 0:
                kick = generate_drum_sound('kick', sample_rate)
                end = min(pos + len(kick), len(drum_audio))
                drum_audio[pos:end] += kick[:end-pos]
            
            # Snare on 1, 3 (2 and 4)
            if beat % 2 != 0:
                snare = generate_drum_sound('snare', sample_rate)
                end = min(pos + len(snare), len(drum_audio))
                drum_audio[pos:end] += snare[:end-pos]
                
            # Hi-hats
            if texture > 0.3: # Add hi-hats for texture
                # 8th notes
                for sub in [0, 0.5]:
                    hh_pos = int((beat + sub) * seconds_per_beat * sample_rate)
                    if hh_pos >= len(drum_audio): break
                    hh = generate_drum_sound('hihat', sample_rate)
                    end = min(hh_pos + len(hh), len(drum_audio))
                    drum_audio[hh_pos:end] += hh[:end-hh_pos] * 0.5

    if multi_track:
        logger.info("Generating Multi-Track Arrangement (Bass + Melody)...")
        
        # 1. Bass Track (Bottom 20%, Low Octave, Slow Rhythm, Sine/Square)
        height = img.shape[0]
        bass_img = img[int(height*0.8):, :, :]
        logger.info("Generating Bass Track...")
        bass_audio, bass_df = generate_track(
            bass_img, 
            track_octave=2, 
            track_instrument='sine', 
            track_quantize_grid='1/4', 
            track_chords=False,
            is_bass=True
        )
        
        # 2. Melody Track (Full Image, High Octave, Fast Rhythm, Selected Instrument)
        logger.info("Generating Melody Track...")
        melody_audio, melody_df = generate_track(
            img, 
            track_octave=4, 
            track_instrument=instrument, 
            track_quantize_grid='1/16', 
            track_chords=use_chords
        )
        
        # Use Melody DF for MIDI output (simplification)
        final_df = melody_df
        
        # Mix
        logger.info("Mixing tracks...")
        max_len = max(len(bass_audio), len(melody_audio))
        if drum_audio is not None:
            max_len = max(max_len, len(drum_audio))
        
        # Pad with zeros
        bass_padded = np.zeros(max_len)
        bass_padded[:len(bass_audio)] = bass_audio
        
        melody_padded = np.zeros(max_len)
        melody_padded[:len(melody_audio)] = melody_audio
        
        # Mix (Bass slightly quieter)
        song = (bass_padded * 0.6) + melody_padded
        
        if drum_audio is not None:
            drum_padded = np.zeros(max_len)
            drum_padded[:len(drum_audio)] = drum_audio
            song += drum_padded * 0.8 # Add drums
        
        # Normalize
        max_val = np.max(np.abs(song))
        if max_val > 0:
            song = song / max_val
            
    else:
        # Single Track (Original Logic)
        song, final_df = generate_track(
            img, 
            track_octave=octave, 
            track_instrument=instrument, 
            track_quantize_grid='1/16', 
            track_chords=use_chords
        )
        
        if drum_audio is not None:
            max_len = max(len(song), len(drum_audio))
            song_padded = np.zeros(max_len)
            song_padded[:len(song)] = song
            
            drum_padded = np.zeros(max_len)
            drum_padded[:len(drum_audio)] = drum_audio
            
            song = song_padded + drum_padded * 0.8
            
            max_val = np.max(np.abs(song))
            if max_val > 0:
                song = song / max_val

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_wav(str(output_path), song, sample_rate)
    logger.info("Image-to-music conversion complete! Output saved to: %s", output_path)

    # MIDI generation
    if midi_output_path and final_df is not None:
        logger.info("Converting frequencies to MIDI notes...")
        midi_stream = frequencies_to_midi_stream(
            final_df, 
            bpm=bpm, 
            filter_repeats=False
        )
        save_midi(midi_stream, midi_output_path)
        logger.info("MIDI file saved to: %s", midi_output_path)
