from backend.Models import rawgat, whisper_specrnet
from neural_network.preprocessing import PreprocessedData, Classification
import os
from Base_Path import get_path_relative_base
model = rawgat()

print('testing')
path = os.path.abspath("../../AuthenticSoundFiles/ArianaGrandeAuthentic.mp3")

csv_file_path = get_path_relative_base("test.csv")
preprocessed_data = PreprocessedData(csv_file_path)
preprocessed_data.add_data(path, "", Classification.REAL, model.evaluate(path))
