import os
import csv
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import torchaudio
import torch
class SVDDDataset:
    def __init__(self, train_audio_dir, train_txt, dev_audio_dir, dev_txt):
        self.train_folder = os.path.expanduser(train_audio_dir)
        self.train_txt = train_txt
        self.dev_folder = os.path.expanduser(dev_audio_dir)
        self.dev_txt = dev_txt

        self.dev_dataset, self.csv_dev_dataset = self.create_dataset(dev_txt, dev_audio_dir)
        self.train_dataset, self.csv_train_dataset = self.create_dataset(train_txt, train_audio_dir)
    
    def create_dataset(self, label_txt_path, audio_files_path):
        dataset = []
        csv_dataset = []
        with open(label_txt_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split(" ")
                audio_path = os.path.join(audio_files_path, parts[2] + ".flac")
                label = 1 if parts[-1].strip().lower() == "deepfake" else 0
                # not all paths in txt exist as they are from other datasets I did not download
                singer_id = parts[3]
                if os.path.exists(audio_path):  # Check if file exists
                    dataset.append((audio_path, label))
                    csv_dataset.append({
                        "file_path": audio_path,
                        "speaker_id": singer_id,
                        "label": label
                    })

                # else:
                #     print(f"Warning: File not found {audio_path}")
        return dataset, csv_dataset
    def save_to_csv(self, _csv_folder_path, overwrite=False):
        self.csv_folder_path = os.path.expanduser(_csv_folder_path)
        if not os.path.exists(self.csv_folder_path):
            if not os.mkdir(self.csv_folder_path):
                raise RuntimeError(f"cannot not create final level folder at {self.csv_folder_path}")
        self.csv_dev_fname = "SVDD_Dev.csv"
        self.csv_train_fname = "SVDD_Train.csv"
        self.csv_dev_path = os.path.join(self.csv_folder_path, self.csv_dev_fname)
        self.csv_train_path = os.path.join(self.csv_folder_path, self.csv_train_fname)
        self.csv_headers = ['file_path', 'speaker_id', 'label']
        
        # helper method to write dataset to path
        def write_csv(path, dataset):
            with open(path, mode='w', newline='') as file:
                writer = csv.DictWriter(file, fieldnames=self.csv_headers)
                writer.writeheader()
                for dict in dataset:
                    writer.writerow(dict)
        # write dev csv
        if os.path.exists(self.csv_dev_path):
            print(f"Dev csv exists: {'Overwriting' if overwrite else 'not overwriting, set overwite=True or delete csv to rewrite'}")
        else:
            write_csv(self.csv_dev_path, self.csv_dev_dataset)
            print(f"Wrote {self.csv_dev_fname} to {self.csv_dev_path}")

        # write train csv
        if os.path.exists(self.csv_train_path):
            print(f"Train csv exists: {'Overwriting' if overwrite else 'not overwriting, set overwite=True or delete csv to rewrite'}")
        else:
            write_csv(self.csv_train_path, self.csv_train_dataset)
            print(f"Wrote {self.csv_train_fname} to {self.csv_train_path}")


class SVDDSpeechBrainDataset(Dataset):
    def __init__(self, csv_path, transform=None, target_length=80000):
        self.data = pd.read_csv(csv_path)
        self.transform = transform
        self.target_length = target_length  # Define your target length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        file_path = row['file_path']
        label = torch.tensor(row['label'], dtype=torch.long)

        # Load audio file
        waveform, sample_rate = torchaudio.load(file_path)
        print(f"Loaded waveform: {file_path}, shape: {waveform.shape}, sample_rate: {sample_rate}")

        # Apply transformations if provided
        if self.transform:
            waveform = self.transform(waveform)

        return {
            "signal": waveform,
            "label": label
        }
