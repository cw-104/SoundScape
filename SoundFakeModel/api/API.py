from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
from Base_Path import get_path_relative_base
from backend.Evaluate import init_whisper_specrnet, eval_file

app = Flask(__name__)
UPLOADS_FOLDER = get_path_relative_base("uploads")
whisper_model = None
rawgat_model = None
config = None
device = None

# Configure allowed file extensions (adjust as needed)
ALLOWED_EXTENSIONS = {'mp3', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/upload', methods=['POST'])
def recieve_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    print(f"file: {file}")

    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(UPLOADS_FOLDER, filename)  
        file.save(file_path)
        # upload was successful process validity
        return process_file(file_path)
    else:
        return jsonify({'error': 'Invalid file type'}), 400

def process_file(path):
    print(">>> do processing with uploaded file")
    pred, label, str_out = eval_file(path, whisper_model, config, device)
    result = jsonify({
        'prediction': pred,
        'label': label,
        'output': str_out
    }), 200
    return result

def start_api():
    global whisper_model, rawgat_model, config, device
    whisper_model, rawgat_model, config, device = init_whisper_specrnet()
    app.run(debug=True, port=8080)


if __name__ == '__main__':
    app.run(debug=True,port=8080)