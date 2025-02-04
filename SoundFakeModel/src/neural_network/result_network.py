import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from neural_network.features import extract_features
from tqdm import tqdm
from queue import Queue
import os, csv, threading, ast
import numpy as np


import torch
import torch.nn as nn

import torch
import torch.nn as nn

class SimpleFNN(nn.Module):
    def __init__(self, model_input_size, audio_input_size, hidden_size=128):
        super(SimpleFNN, self).__init__()
        
        # Layers for model features
        self.model_fc1 = nn.Linear(model_input_size, hidden_size)
        self.model_fc2 = nn.Linear(hidden_size, hidden_size)
        
        # Layers for audio features
        self.audio_fc1 = nn.Linear(audio_input_size, hidden_size)
        self.audio_fc2 = nn.Linear(hidden_size, hidden_size)
        
        # Final layers
        self.fc_final = nn.Linear(hidden_size * 2, 2) # 2 output classes
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, model_features, audio_features):
        # Process model features
        x1 = self.model_fc1(model_features)
        x1 = self.relu(x1)
        x1 = self.model_fc2(x1)

        # Process audio features
        x2 = self.audio_fc1(audio_features)
        x2 = self.relu(x2)
        x2 = self.audio_fc2(x2)

        # Concatenate outputs from both branches
        x = torch.cat((x1, x2), dim=1)
        x = self.fc_final(x)
        return self.softmax(x)



class LabeledAudioFile():
    def __init__(self, audio_file_path, label):
        self.file_path = audio_file_path
        self.label = label
            

    def get_label(self):
            return self.label


class AudioDataset(Dataset):
    def __init__(self, model_features, audio_features, labels, file_list=None):
        self.model_features = model_features
        self.audio_features = audio_features
        self.labels = labels
        self.num_model_features = len(self.model_features)
        self.num_audio_features = len(self.audio_features)

    def __len__(self):
        return len(self.audio_features)
    

    def __getitem__(self, idx):
        return self.model_features[idx], self.audio_features[idx], self.labels[idx]

    @staticmethod
    def _save_to_file(dataset, file_path):
        """Save the dataset to a file."""
        data = []
        for i in range(len(dataset)):
            model_features, audio_features, label = dataset[i]
            data.append((model_features, audio_features, label))  

        torch.save(data, file_path)  # Save to a .pt 
    

    @staticmethod
    def load_from_file(file_path):
        """Load the dataset from a file."""
        data = torch.load(file_path)  # Load the data from the file
        model_features, audio_features, labels = zip(*data)  # Unzip the data into features and labels
        model_features = [torch.tensor(f, dtype=torch.float32) for f in model_features]  # Convert features to tensors
        audio_features = [torch.tensor(f, dtype=torch.float32) for f in audio_features]  # Convert features to tensors

        labels = [torch.tensor(l, dtype=torch.long) for l in labels]  # Convert labels to tensors

        return AudioDataset(model_features=model_features, audio_features=audio_features, labels=labels)


    @staticmethod
    def custom_collate_fn(batch):
        model_features, audio_features, labels = zip(*batch)
        # Extract model features and stack them
        model_features = {
            'rawgat_confidence': torch.stack([f['rawgat_confidence'] for f in model_features]),
            'rawgat_label': torch.stack([f['rawgat_label'] for f in model_features]),
            'whisper_confidence': torch.stack([f['whisper_confidence'] for f in model_features]),
            'whisper_label': torch.stack([f['whisper_label'] for f in model_features])
        }

        # Extract audio features and stack them
        audio_features = {
            'rms': torch.stack([f['rms'] for f in audio_features]),
            'magnitude_spectrum': pad_sequence([f['magnitude_spectrum'] for f in audio_features], batch_first=True),
            'spectral_centroid': torch.stack([f['spectral_centroid'] for f in audio_features])
        }

        # Convert labels to a tensor
        labels = torch.tensor(labels)

        return model_features, audio_features, labels


    @staticmethod
    def create_dataset(labeledAudioFiles : [LabeledAudioFile], rawgat_model, whisper_specrnet_model, save_path=None, n_threads=4):
        if save_path and os.path.exists(save_path):
            raise FileExistsError(f"{save_path} already exists.")
        model_features_list = []
        audio_features_list = []
        labels = []
        file_paths = []


        audio_file_queue = Queue()
        for audio_file in labeledAudioFiles:
            audio_file_queue.put(audio_file)
        # remove labeledAudioFiles from memory
        n_files = len(labeledAudioFiles)
        labeledAudioFiles = None
        
        threads = []
        lock = threading.Lock()
        

        def extract_features_thread():
            while not audio_file_queue.empty():
                audio_file = audio_file_queue.get()
                model_features, audio_features = extract_features(audio_file.file_path, rawgat_model, whisper_specrnet_model)
                with lock:
                    model_features_list.append(model_features)
                    audio_features_list.append(audio_features)
                    labels.append(audio_file.get_label())
                    file_paths.append(audio_file.file_path)
                    pbar.update(1)
                
        with tqdm(total=n_files) as pbar:
            for i in range(n_threads):
                threads.append(threading.Thread(target=extract_features_thread))
                threads[-1].start()
             # Wait for all threads to complete
            for thread in threads:
                thread.join()

        dataset = AudioDataset(audio_features=audio_features_list, model_features=model_features_list, labels=labels)
        if save_path is not None:
            AudioDataset._save_to_file(dataset, save_path)

        return dataset
