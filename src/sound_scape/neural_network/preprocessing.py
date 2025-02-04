from sound_scape.backend.Isolate import isolate_file

def pre_isolate(audio_file_path):
    """
    Pre-process the audio file to isolate the voice,
    make training more efficient
    """
    # Load the audio file
    audio, sr = librosa.load(audio_file_path, sr=16000)
    # Isolate the voice
    isolated_path = isolate_file(audio_file_path, mp3=True)
    return isolated_path

def process_model_x_times(audio_file_path, model):
    """
    process a file multiple times to test result randomness/variance
    """

