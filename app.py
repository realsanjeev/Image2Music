import os
import time
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename

from image2music.pipeline import convert_image_to_music
from image2music.logger import get_logger

app = Flask(__name__)
logger = get_logger(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
GENERATED_FOLDER = 'generated'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'webp'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['GENERATED_FOLDER'] = GENERATED_FOLDER

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GENERATED_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        # Add timestamp to avoid collisions
        timestamp = int(time.time())
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Get parameters from form
        bpm = int(request.form.get('bpm', 120))
        scale = request.form.get('scale', 'MAJOR').upper()
        key = request.form.get('key', 'A')
        instrument = request.form.get('instrument', 'rich')
        
        auto_bpm = request.form.get('auto_bpm') == 'true'
        auto_scale = request.form.get('auto_scale') == 'true'
        multi_track = request.form.get('multi_track') == 'true'
        use_drums = request.form.get('use_drums') == 'true'
        use_chords = request.form.get('use_chords') == 'true'
        quantize = request.form.get('quantize') == 'true'
        
        reverb = float(request.form.get('reverb', 0.0))
        delay = float(request.form.get('delay', 0.0))
        
        # Output filename
        output_filename = f"song_{timestamp}.wav"
        output_path = os.path.join(app.config['GENERATED_FOLDER'], output_filename)
        
        midi_filename = f"song_{timestamp}.midi"
        midi_path = os.path.join(app.config['GENERATED_FOLDER'], midi_filename)
        
        try:
            logger.info(f"Processing {filename}...")
            convert_image_to_music(
                image_path=filepath,
                output_path=output_path,
                midi_output_path=midi_path,
                bpm=bpm,
                scale_name=scale,
                key=key,
                instrument=instrument,
                auto_bpm=auto_bpm,
                auto_scale=auto_scale,
                multi_track=multi_track,
                use_drums=use_drums,
                use_chords=use_chords,
                quantize=quantize,
                reverb=reverb,
                delay=delay
            )
            
            return jsonify({
                'success': True,
                'audio_url': f"/generated/{output_filename}",
                'midi_url': f"/generated/{midi_filename}"
            })
            
        except Exception as e:
            logger.error(f"Generation failed: {e}", exc_info=True)
            return jsonify({'error': str(e)}), 500
            
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/generated/<path:filename>')
def serve_generated(filename):
    return send_from_directory(app.config['GENERATED_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True, port=5000)
