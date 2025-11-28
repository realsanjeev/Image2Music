# image2music/pipeline.py

from pathlib import Path

from .logger import get_logger
from .image_utils import load_image, extract_pixel_data
from .scales import make_scale
from .music_mapping import hues_dataframe
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
    bpm: int = 120
) -> None:
    """
    Convert an image into music (WAV + optional MIDI) by mapping pixel hues to scale frequencies.

    Parameters
    ----------
    image_path : str
        Path to the image file.
    output_path : str
        Path to save the generated WAV file.
    midi_output_path : str
        Path to save the generated MIDI file.
    """
    logger.info("Loading image: %s", image_path)
    img = load_image(image_path)

    logger.info("Extracting pixel data (Hue, Saturation, Value)...")
    pixel_data = extract_pixel_data(img)

    logger.info("Generating scale: %s %s octave %d", key, scale_name, octave)
    scale_freqs, _ = make_scale(octave=octave, key=key.lower(), scale=scale_name)

    logger.info("Mapping pixels to musical properties...")
    df = hues_dataframe(pixel_data, scale_freqs, base_duration=duration_per_note)

    logger.info("Generating song waveform...")
    song = generate_song(
        frequencies=df["frequency"].tolist(),
        amplitudes=df["amplitude"].tolist(),
        durations=df["duration"].tolist(),
        sample_rate=sample_rate, 
        use_octaves=use_octaves
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
