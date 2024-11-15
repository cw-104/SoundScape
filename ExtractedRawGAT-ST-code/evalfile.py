import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import ffmpeg
import io

class SingleFileDataset(Dataset):
    def __init__(self, filepath, transform=None):
        self.filepath = filepath
        self.transform = transform
    
    def __len__(self):
        return 1
    
    def __getitem__(self, idx):
        waveform, sample_rate = load_audio_ffmpeg(self.filepath)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
        
        if self.transform:
            waveform = self.transform(waveform[0])  # Only get the first channel
        
        # Set a dummy label (0: spoof, 1: bonafide) and metadata for compatibility
        return waveform, torch.tensor([1]), {'filename': self.filepath.split('/')[-1]}
    

def evaluate_single_file(filepath, model, device):
    # Define the transformation and dataset
    transform = transforms.Compose([
        lambda x: pad(x),  
        lambda x: torch.Tensor(x)
    ])
    dataset = SingleFileDataset(filepath, transform=transform)
    data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Perform evaluation
    model.eval()
    with torch.no_grad():
        for batch_x, batch_y, batch_meta in data_loader:
            batch_x = batch_x.to(device)
            output = model(batch_x, Freq_aug=False)
            score = output[:, 1].item()  
            print(f"File: {batch_meta['filename']} - Score: {score}")
            print("Bonafide" if score > 0.5 else "Spoof")


def pad(x, max_len=64600):
    x_len = x.shape[0]
    if x_len >= max_len:
        return x[:max_len]
    num_repeats = int(max_len / x_len) + 1
    padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
    return padded_x

    
def load_audio_ffmpeg(filepath):
    # Use ffmpeg-python to load FLAC file and convert it to raw waveform (WAV format)
    out, _ = (
        ffmpeg.input(filepath)
        .output('pipe:', format='wav')
        .run(capture_stdout=True, capture_stderr=True)
    )
    # Convert the byte data to a NumPy array
    waveform = np.frombuffer(out, np.int16).astype(np.float32) / 32768.0
    waveform = torch.Tensor(waveform).unsqueeze(0)  # Add the channel dimension if it's mono
    sample_rate = 16000  # Assuming the resampling was done correctly
    return waveform, sample_rate



