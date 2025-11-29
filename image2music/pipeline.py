# image2music/pipeline.py

from pathlib import Path

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
    instrument: str = "rich"
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
    """
    logger.info("Loading image: %s", image_path)
    img = load_image(image_path, color_space=color_space)

    logger.info("Extracting pixel data with '%s' sampling in %s color space...", sampling_strategy, color_space.upper())
    pixel_data = extract_pixel_data(
        img, 
        sampling_strategy=sampling_strategy,
        step=grid_step,
        num_samples=num_samples,
        color_space=color_space
    )
    
    logger.info("Sampled %d pixels", len(pixel_data[list(pixel_data.keys())[0]]))

    logger.info("Generating scale: %s %s octave %d", key, scale_name, octave)
    scale_freqs, _ = make_scale(octave=octave, key=key.lower(), scale=scale_name)

    logger.info("Mapping pixels to musical properties...")
    df = hues_dataframe(
        pixel_data, 
        scale_freqs, 
        base_duration=duration_per_note, 
        color_space=color_space,
        use_kmeans=use_kmeans,
        image_path=image_path
    )
    
    # Apply smoothing if requested
    if smooth_window > 1:
        df = smooth_parameters(df, window_size=smooth_window)
    
    # Add phrase boundaries if requested
    if phrase_length > 0:
        df = add_phrase_boundaries(df, phrase_length=phrase_length)

    logger.info("Generating song waveform...")
    song = generate_song(
        frequencies=df["frequency"].tolist(),
        amplitudes=df["amplitude"].tolist(),
        durations=df["duration"].tolist(),
        sample_rate=sample_rate, 
        use_octaves=use_octaves,
        instrument=instrument
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    save_wav(str(output_path), song, sample_rate)
    logger.info("Image-to-music conversion complete! Output saved to: %s", output_path)

    # MIDI generation
    if midi_output_path:
        logger.info("Converting frequencies to MIDI notes...")
        midi_stream = frequencies_to_midi_stream(
            df, 
            bpm=bpm, 
            filter_repeats=False
        )
        save_midi(midi_stream, midi_output_path)
        logger.info("MIDI file saved to: %s", midi_output_path)
