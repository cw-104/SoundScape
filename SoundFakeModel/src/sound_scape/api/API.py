from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import tempfile
import os, uuid
from Base_Path import get_path_relative_base
from flask_cors import CORS
from sound_scape.api.Bindings import ModelBindings
import time

app = Flask(__name__)
model_bindings = ModelBindings()

ALLOWED_EXTENSIONS = {'mp3'}

@app.route('/upload', methods=['POST'])
def recieve_file():
    if 'audio' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['audio']

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)

        # Create a temporary file and save the uploaded data
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
        file.save(temp_file.name)
        temp_path = temp_file.name  # Get the temporary file path
        temp_file.close()  # Close file so it can be accessed later

        return confirm_upload(temp_path)
    
    return jsonify({'error': 'Invalid file type'}), 400

def confirm_upload(file_path):
    id = str(uuid.uuid4())  # Generate a unique ID
    print("uploaded id: ", id)

    # Upload the file and schedule it for later deletion
    model_bindings.upload_file(id, file_path)

    return jsonify({'id': id}), 200

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/status', methods=['POST'])
def get_status():
    if request.get_data() == b'':
        return jsonify({'status': 'Error: No id provided'}), 400
    
    id = request.get_data().decode('utf-8')

    if not model_bindings.file_ids.exists(id):
        return jsonify({'status': 'Does not exist', 'error': 'Invalid ID'}), 404
    
    return jsonify(model_bindings.file_ids.get_status(id)), 200

@app.route('/results', methods=['POST'])
def get_results():
    data = request.get_json()
    file_id = data.get("id")

    for _ in range(15):  # Retry for up to 45 seconds
        if model_bindings.file_ids.exists(file_id):
            status = model_bindings.file_ids.get_status(file_id)

            if status.get("state") == "finished":
                results = model_bindings.file_ids.get_results(file_id)
                
                # Remove file only AFTER processing completes
                file_path = model_bindings.file_ids.get_path(file_id)
                try:
                    os.remove(file_path)
                except Exception as e:
                    print(f"Error deleting file {file_path}: {e}")

                return jsonify(results), 200

            print(f"DEBUG: File {file_id} is still processing... Retrying...")

        else:
            print(f"DEBUG: File {file_id} is not found yet. Waiting...")

        time.sleep(3)  # Wait 3 seconds before checking again

    return jsonify({'status': 'error: processing took too long'}), 500

def start_api():
    global app
    CORS(app)
    app.run(debug=True, port=8080)
