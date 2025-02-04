from tqdm import tqdm
from backend.Models import rawgat, whisper_specrnet
from neural_network.preprocessing import PreprocessedData, REAL_LABLE, FAKE_LABLE
import os
from Base_Path import get_path_relative_base

def unproc_rows_from_dir(directory, label):
    directory = os.path.abspath(directory)
    new_rows = []
    for file in tqdm(os.listdir(directory)):
        file = os.path.join(directory, file)
        if file.endswith(".mp3"):
            new_rows.append(preprocessed_data.create_unprocessed_row(file, label))

preprocessed_data = PreprocessedData(get_path_relative_base("test.csv"))
preprocessed_data.add_entries(unproc_rows_from_dir("../../AuthenticSoundFiles", REAL_LABLE))
preprocessed_data.add_entries(unproc_rows_from_dir("../../DeepfakeSoundFiles", REAL_LABLE))

