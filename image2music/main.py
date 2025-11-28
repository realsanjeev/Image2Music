# image2music/main.py

import argparse
import sys
from pathlib import Path

from . import config
from .logger import get_logger
from .pipeline import convert_image_to_music

logger = get_logger(__name__)


def parse_args() -> argparse.Namespace:
    """
    Parse command-line arguments for the image-to-music converter.
    """
    parser = argparse.ArgumentParser(
        description="Convert an image into a music composition."
    )
    parser.add_argument(
        "image_path",
        type=str,
        help="Path to the input image file."
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Path to save the generated WAV file. Defaults to OUTPUT_DIR/<image_stem>.wav"
    )
    parser.add_argument(
        "--midi",
        type=str,
        default=None,
        help="Optional path to save the generated MIDI file."
    )
    parser.add_argument(
        "--bpm",
        type=int,
        default=120,
        help="Tempo in beats per minute for MIDI output."
    )
    parser.add_argument(
        "-s", "--scale",
        type=str,
        default="HARMONIC_MINOR",
        choices=["HARMONIC_MINOR", "MAJOR", "MINOR", "PENTATONIC", "BLUES"],
        help="Musical scale to use."
    )
    parser.add_argument(
        "-k", "--key",
        type=str,
        default="A",
        help="Key for the scale (e.g., C, D, E, F#, etc.)"
    )
    parser.add_argument(
        "-oc", "--octave",
        type=int,
        default=3,
        help="Octave to use for the notes."
    )
    parser.add_argument(
        "-d", "--duration",
        type=float,
        default=0.1,
        help="Duration of each note in seconds."
    )
    parser.add_argument(
        "-sr", "--sample_rate",
        type=int,
        default=config.SAMPLE_RATE,
        help="Sample rate for audio generation."
    )
    parser.add_argument(
        "--no-octaves",
        action="store_true",
        help="Disable random octave variations."
    )
    parser.add_argument(
        "--sampling",
        type=str,
        default="grid",
        choices=["all", "grid", "spiral", "edges", "weighted"],
        help="Pixel sampling strategy."
    )
    parser.add_argument(
        "--grid-step",
        type=int,
        default=4,
        help="Step size for grid sampling."
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=50,
        help="Number of samples for non-grid strategies."
    )
    parser.add_argument(
        "--smooth",
        type=int,
        default=3,
        help="Smoothing window size (0 = no smoothing)."
    )
    parser.add_argument(
        "--phrase-length",
        type=int,
        default=8,
        help="Insert rest every N notes (0 = no phrases)."
    )
    return parser.parse_args()


def main():
    args = parse_args()

    image_path = Path(args.image_path)
    if not image_path.exists():
        logger.error("Image not found: %s", image_path)
        sys.exit(1)

    output_path = Path(args.output) if args.output else config.OUTPUT_DIR / f"{image_path.stem}.wav"

    try:
        convert_image_to_music(
            image_path=str(image_path),
            output_path=str(output_path),
            scale_name=args.scale,
            key=args.key,
            octave=args.octave,
            duration_per_note=args.duration,
            sample_rate=args.sample_rate,
            use_octaves=not args.no_octaves,
            midi_output_path=args.midi,
            bpm=args.bpm,
            sampling_strategy=args.sampling,
            grid_step=args.grid_step,
            num_samples=args.num_samples,
            smooth_window=args.smooth,
            phrase_length=args.phrase_length
        )
        logger.info("Music generation complete! File saved to: %s", output_path)
    except Exception as e:
        logger.exception("Failed to convert image to music: %s", e)
        sys.exit(1)


if __name__ == "__main__":
    main()

