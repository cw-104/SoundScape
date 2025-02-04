from Base_Path import get_path_relative_base
from neural_network.neural import AudioClassifier, AudioDataset, LabeledAudioFile
import os
from backend.Models import rawgat, whisper_specrnet
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from neural_network.training import train_model

def get_labeled_files_from_dir(directory, label):
    directory = os.path.abspath(directory)
    labeled_files = []
    for file in os.listdir(directory):
        file = os.path.join(directory, file)
        if file.endswith(".mp3"):
            labeled_files.append(LabeledAudioFile(file, label))
    return labeled_files



def init_dataloader(l_audio_files):
    if os.path.exists(dataset_path):
        dataset = AudioDataset.load_from_file(dataset_path)
    else:
        # Create the dataset
        rawgat_model = rawgat()
        whisper_specrnet_model = whisper_specrnet()
        dataset = AudioDataset.create_dataset(l_audio_files, rawgat_model, whisper_specrnet_model, save_path=dataset_path)
        
    return DataLoader(dataset, batch_size=16, shuffle=True, collate_fn=AudioDataset.custom_collate_fn), AudioClassifier(n_model_features=dataset.n_model_features, n_rms=dataset.n_rms, n_magnitude_spectrum= dataset.n_magnitude_spectrum)


dataset_path = get_path_relative_base("datasets/features.pt")
REAL_LABLE = 1
FAKE_LABLE = -1
# labeled_audio_files = get_labeled_files_from_dir("../../AuthenticSoundFiles", REAL_LABLE) + get_labeled_files_from_dir("../../DeepfakeSoundFiles", FAKE_LABLE)
labeled_audio_files = get_labeled_files_from_dir("../../DeepfakeSoundFiles", REAL_LABLE)
dataloader, model = init_dataloader(labeled_audio_files)
# clear memory
labeled_audio_files = None

criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)  # Adam optimizer


train_model(model, dataloader, criterion, optimizer, num_epochs=50)