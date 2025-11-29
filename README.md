# Image2Music: Algorithmic Music Composer

Transform images into music using advanced audio synthesis, intelligent analysis, and multi-track arrangement.

## 🎵 Features

### Audio Engine
- **Additive Synthesis**: Rich timbres with harmonic generation (Sine, Organ, Woodwind, Brass, Rich Synth, Square Wave)
- **ADSR Envelopes**: Natural note articulation with Attack, Decay, Sustain, and Release
- **Audio Effects**: Reverb, Delay, and Chorus via `pedalboard`

### Musical Structure
- **Rhythm Quantization**: Snap note durations to musical grid (1/4, 1/8, 1/16 notes)
- **Chord Mode**: Generate triads (Root, 3rd, 5th) for richer harmony
- **Phrase Boundaries**: Automatic rest insertion for musical phrasing

### Intelligent Analysis
- **Auto-BPM**: Derive tempo from image brightness (Bright = Fast, Dark = Slow)
- **Auto-Scale**: Determine musical scale from color temperature (Warm = Major, Cool = Minor)

### Multi-Track Arrangement
- **Bass Track**: Generated from bottom 20% of image with low octaves and rhythmic patterns
- **Melody Track**: Generated from full image with higher octaves and faster rhythm
- **Percussion Track**: Synthesized drums (Kick, Snare, Hi-hat) based on image texture

### Output Formats
- **WAV**: High-quality audio files
- **MIDI**: Standard MIDI files for further editing in DAWs

## 🚀 Getting Started

### Installation

1. **Create a Virtual Environment**

   For **Linux and macOS**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

   For **Windows**:
   ```cmd
   python -m venv venv
   .venv\Scripts\activate
   ```

2. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## 📖 Usage

### Command Line Interface

#### Basic Usage
```bash
python -m image2music.main -o output.wav image.jpg
```

#### Full Band Experience (Recommended)
```bash
python -m image2music.main -o song.wav \
    --multi-track --drums \
    --auto-bpm --auto-scale \
    --quantize --chords \
    --reverb 0.3 --delay 0.2 \
    --midi song.midi \
    image.jpg
```

#### CLI Options

**Arrangement**:
- `--multi-track`: Generate Bass + Melody tracks
- `--drums`: Add percussion (Kick, Snare, Hi-hat)
- `--chords`: Generate triads instead of single notes

**Intelligence**:
- `--auto-bpm`: Auto-detect tempo from brightness
- `--auto-scale`: Auto-detect scale from color temperature

**Manual Overrides**:
- `--bpm <value>`: Set tempo (40-200, default: 120)
- `--scale <name>`: Set scale (MAJOR, MINOR, HARMONIC_MINOR, PENTATONIC_MAJOR, PENTATONIC_MINOR)
- `--key <note>`: Set key (C, C#, D, ..., B)
- `--instrument <name>`: Set instrument (rich, sine, square, organ, woodwind, brass)

**Effects**:
- `--reverb <0.0-1.0>`: Reverb mix level
- `--delay <0.0-1.0>`: Delay mix level
- `--chorus <0.0-1.0>`: Chorus mix level

**Other**:
- `--quantize`: Enable rhythm quantization
- `--midi <path>`: Generate MIDI file
- `-o <path>`: Output WAV file path

### Web Interface

Launch the web app for a visual, interactive experience:

```bash
python app.py
# Visit http://localhost:5000
```

**Features**:
- Drag & drop image upload
- Real-time parameter adjustment
- In-browser audio playback
- Download WAV/MIDI files
- Modern dark "Studio" theme

## 🎹 Examples

### Ambient Soundscape
```bash
python -m image2music.main -o ambient.wav \
    --instrument sine \
    --reverb 0.5 \
    --scale PENTATONIC_MINOR \
    image.jpg
```

### Upbeat Dance Track
```bash
python -m image2music.main -o dance.wav \
    --multi-track --drums \
    --bpm 140 --scale MAJOR \
    --quantize --chords \
    image.jpg
```

### Cinematic Score
```bash
python -m image2music.main -o cinematic.wav \
    --multi-track \
    --instrument brass \
    --reverb 0.4 --delay 0.3 \
    --auto-bpm --auto-scale \
    image.jpg
```

## 🔬 Technical Details

### Architecture
```
Image → [Analyzer] → [Composer] → [Synthesizer] → Audio/MIDI
         ↓             ↓             ↓
      Texture      Multi-Track    Drums + Bass + Melody
      BPM/Scale    Chords         Effects (Reverb/Delay)
```

### Color Space Analysis
- **LCH (Default)**: Perceptually uniform color space for natural musical mapping
- **LAB**: Alternative perceptual color space
- **HSV**: Hue-Saturation-Value for traditional color analysis

### Sampling Strategies
- **Grid**: Uniform sampling across the image
- **Spiral**: Radial sampling from center
- **Edges**: Focus on high-contrast regions
- **Weighted**: Importance-based sampling

## 📚 Notebooks

Explore the development process and experiments:
- [`music_generation_tone_matrix.ipynb`](./notebooks/music_generation_tone_matrix.ipynb): Music21 basics and tone matrix algorithm
- [`construct_music_from_images.ipynb`](./notebooks/construct_music_from_images.ipynb): Image-to-music pipeline development

## 🛠️ Dependencies

- **NumPy**: Numerical computing
- **Pandas**: Data manipulation
- **OpenCV**: Image processing
- **Music21**: MIDI generation and music theory
- **Librosa**: Audio analysis
- **Pedalboard**: Audio effects
- **scikit-learn**: K-means clustering for perceptual pitch mapping
- **Flask**: Web interface

## 📖 References

- [NumPy Documentation](https://numpy.org/doc/)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [OpenCV Documentation](https://opencv24-python-tutorials.readthedocs.io/en/latest/index.html)
- [Music21 Documentation](https://www.music21.org/music21docs/)
- [Librosa Documentation](https://librosa.org/doc/latest/index.html)
- [Pedalboard Documentation](https://pypi.org/project/pedalboard/)
- [Flask Documentation](https://flask.palletsprojects.com/)

## 🎯 Project Evolution

This project has evolved from a basic pixel-to-sound mapper into a full-featured algorithmic music composer:

1. **Phase 1**: Audio Engine (Additive Synthesis, ADSR, Effects)
2. **Phase 2**: Musical Structure (Quantization, Chords)
3. **Phase 3**: Intelligent Analysis (Auto-BPM, Auto-Scale, Multi-Track)
4. **Phase 4**: Rhythm Section (Percussion, Rhythmic Bass)
5. **Phase 5**: Web Interface (Flask App)

## 📝 License

This project is open source and available for educational and research purposes.