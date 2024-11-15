import os
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import torchaudio  # Assuming torchaudio is used for Resample
import ffmpeg
import io

from evalfile import load_audio_ffmpeg, pad

class FolderDataset(Dataset):
    def __init__(self, folderpath, transform=None):
        self.folderpath = folderpath
        self.filepaths = [os.path.join(folderpath, fname) for fname in os.listdir(folderpath) if fname.endswith(('.wav', '.mp3', '.flac'))]
        self.transform = transform
    
    def __len__(self):
        return len(self.filepaths)
    
    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        waveform, sample_rate = load_audio_ffmpeg(filepath)
        
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
        
        if self.transform:
            waveform = self.transform(waveform[0])  # Process only the first channel
        
        # Set a dummy label (0: spoof, 1: bonafide) and metadata
        return waveform, torch.tensor([1]), {'filename': os.path.basename(filepath)}

def evaluate_folder(folderPath, model, device):
    transform = transforms.Compose([
        lambda x: pad(x),  
        lambda x: torch.Tensor(x)
    ])
    dataset = FolderDataset(folderPath, transform=transform)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Perform evaluation
    model.eval()
    print("✅: Real | ❌: Deepfake")
    print("--------------------------------------------------\n")
    # total count of files
    results = []

    with torch.no_grad():
        for batch_x, batch_y, batch_meta in data_loader:
            batch_x = batch_x.to(device)
            output = model(batch_x, Freq_aug=False)
            score = output[:, 1].item()  


            # res = "✅" if score > min_real_score else "❌"
            print(f"{batch_meta['filename'][0]}")
            print(f"Score: {score}")
            results.append((batch_meta['filename'][0], score))

    return results
            