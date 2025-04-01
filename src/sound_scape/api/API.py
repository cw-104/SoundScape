from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os, uuid
from Base_Path import get_path_relative_base
from flask_cors import CORS
from sound_scape.api.Bindings import ModelBindings

app = Flask(__name__)
UPLOADS_FOLDER = get_path_relative_base("uploads")

model_bindings = ModelBindings()


# Configure allowed file extensions (adjust as needed)
ALLOWED_EXTENSIONS = {'mp3'}

@app.route('/upload', methods=['POST'])
def recieve_file():
    if not 'audio' in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['audio']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOADS_FOLDER, filename)
        file.save(file_path)
        # upload was successful process validity
        return confirm_upload(file_path)
    else:
        return jsonify({'error': 'Invalid file type'}), 400

def confirm_upload(file_path):
    # create unique id to send back to client
    id = str(uuid.uuid4())
    print("uploaded id: ", id)

    model_bindings.upload_file(id, file_path)
    return jsonify({'id': id}), 200
    
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/status', methods=['POST'])
def get_status():
    data = request.json
    if 'id' in data:
        file_id = data['id']
    
    if not file_id:
        return jsonify({'state': 'Error: No id provided'}), 400

    if not model_bindings.file_ids.exists(file_id):
        return jsonify({'state': 'Does not exist', 'error': 'Invalid ID'}), 404

    return jsonify(model_bindings.file_ids.get_status(file_id)), 200


@app.route('/results', methods=['POST'])
def get_results():
    if request.get_data() == b'':
        return jsonify({'status': 'Error: No id provided'}), 400
    id = request.get_data().decode('utf-8')

    if not model_bindings.file_ids.exists(id):
        return jsonify({
            'status': 'Does not exist',
            'error': 'Invalid ID'}), 404
    return jsonify(model_bindings.file_ids.get_results(id)), 200

@app.route('/resultsv2', methods=['POST'])
def get_results2():
    data = request.json

    if data is None or 'id' not in data:
        return jsonify({'status': 'Error: No id provided'}), 400

    id = data['id']

    if not model_bindings.file_ids.exists(id):
        print("invalid id")
        return jsonify({
            'status': 'Does not exist',
            'error': 'Invalid ID'
        }), 404
    
    results = model_bindings.file_ids.get_results(id)
    return jsonify(results), 200

import logging
from numba import config

def start_api():
    if not os.path.exists(UPLOADS_FOLDER):
        os.makedirs(UPLOADS_FOLDER)
        
    # Set the logging level for Numba to WARNING
    logging.getLogger('numba').setLevel(logging.WARNING)

    # Alternatively, you can set the logging level for all loggers
    logging.basicConfig(level=logging.WARNING)
    global app
    CORS(app, resources={r"/*": {"origins": ["https://projectsoundscape.net", "*"]}})
    app.run(debug=True, port=8080)
