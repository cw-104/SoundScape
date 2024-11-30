import os
import numpy as np
import torch
import torch.nn as nn
import torchaudio
import torchaudio.transforms as T
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from SVDDDataset import SVDDDataset
from Evaluate import get_best_device

# Parameters
n_mels = 13
training_folder = "/Users/christiankilduff/Deepfake_Detection_Resources/Datasets/SVDD/train"
dev_folder = "/Users/christiankilduff/Deepfake_Detection_Resources/Datasets/SVDD/dev"

train_features_dir = "logmel_train"
dev_save = "logmel_dev"

# Initialize the dataset
SVDD_data = SVDDDataset(
    train_txt=os.path.join(training_folder, "train.txt"),
    train_audio_dir=os.path.join(training_folder, "train_set"),
    dev_audio_dir=os.path.join(dev_folder, "dev_set"),
    dev_txt=os.path.join(dev_folder, "dev.txt")
)

# Function to extract log Mel spectrogram
def extract_log_mel(audio_file, n_fft=400, hop_length=160):
    waveform, sample_rate = torchaudio.load(audio_file, normalize=True)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # Convert to mono
    mel_transform = T.MelSpectrogram(sample_rate=sample_rate, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels)
    mel_spectrogram = mel_transform(waveform)
    log_mel = mel_spectrogram.log()
    log_mel_flat = log_mel.squeeze(0)  # Remove the channel dimension
    return log_mel_flat

# Function to preprocess and save features with periodic progress print
def preprocess_and_save_features(audio_files, feature_dir, print_interval=822):
    os.makedirs(feature_dir, exist_ok=True)
    
    total_files = len(audio_files)  # Total number of files
    for idx, (audio_file, label) in enumerate(audio_files):
        feature_filename = f"{os.path.basename(audio_file)}.npy"
        feature_file = os.path.join(feature_dir, feature_filename)
        
        if not os.path.exists(feature_file):
            log_mel = extract_log_mel(audio_file)
            np.save(feature_file, log_mel.numpy())

        # print progress per interval of files processed        
        if (idx + 1) % print_interval == 0 or (idx + 1) == total_files:
            print(f"Loaded {idx + 1}/{total_files} {((idx + 1) / total_files) * 100:.2f}% files.")

# Preprocess and save features for train and dev data
preprocess_and_save_features(SVDD_data.train_dataset, train_features_dir)
preprocess_and_save_features(SVDD_data.dev_dataset, dev_save)

print()
print("loaded logmel features")

# Custom Dataset Class to use pre-saved features
class SVDDDatasetWithFeatures(Dataset):
    def __init__(self, data, feature_dir, transform=None):
        self.data = data
        self.feature_dir = feature_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        audio_file, label = self.data[idx]
        feature_file = os.path.join(self.feature_dir, f"{os.path.basename(audio_file)}.npy")
        
        # Load precomputed feature if it exists
        if os.path.exists(feature_file):
            log_mel = torch.tensor(np.load(feature_file), dtype=torch.float32)
            # print(f"Loaded feature for {audio_file}")
        else:
            raise FileNotFoundError(f"Feature for {audio_file} not found.")
        
        if self.transform:
            log_mel = self.transform(log_mel)
        
        return log_mel, label



# Define the ECAPA-TDNN model
class ECAPA_TDNN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ECAPA_TDNN, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 512, kernel_size=5, stride=1, padding=2)
        self.tdnn1 = nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1)
        self.tdnn2 = nn.Conv1d(512, 512, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, output_dim)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.tdnn1(x))
        x = torch.relu(self.tdnn2(x))
        x = torch.mean(x, dim=-1)  # Global average pooling
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Custom collate function for DataLoader
def collate_fn(batch):
    seq_lengths = [log_mel.shape[-1] for log_mel, label in batch]
    q3_median = np.percentile(seq_lengths, 20)  # 75th percentile

    padded_batch = []
    for log_mel, label in batch:
        seq_length = log_mel.shape[-1]
        if seq_length < q3_median:
            padding_size = int(q3_median - seq_length)
            padding = torch.zeros(log_mel.shape[0], padding_size)
            padded_log_mel = torch.cat((log_mel, padding), dim=-1)
        else:
            padded_log_mel = log_mel[:, :int(q3_median)]
        
        padded_log_mel = padded_log_mel.unsqueeze(0)  # Add batch dimension
        padded_batch.append((padded_log_mel, label))

    spectrograms, labels = zip(*padded_batch)
    spectrograms = torch.cat(spectrograms, dim=0)
    labels = torch.tensor(labels)
    
    return spectrograms, labels


print("")
# Prepare DataLoaders
train_loader = DataLoader(SVDDDatasetWithFeatures(SVDD_data.train_dataset, train_features_dir), 
                          batch_size=1, shuffle=True, collate_fn=collate_fn, pin_memory=True)
valid_loader = DataLoader(SVDDDatasetWithFeatures(SVDD_data.dev_dataset, dev_save), 
                          batch_size=1, shuffle=False, collate_fn=collate_fn, pin_memory=True)


# Get the best device (GPU or MPS for Apple Silicon)
device = get_best_device()
model = ECAPA_TDNN(input_dim=n_mels, output_dim=2).to(device)  # Example: 2 classes (real/fake)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-3)

# Training loop
for epoch in range(10):
    model.train()
    running_loss = 0
    for mfcc, label in train_loader:
        mfcc, label = mfcc.to(device), label.to(device)
        
        optimizer.zero_grad()
        output = model(mfcc)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f'Epoch {epoch+1}, Loss: {running_loss / len(train_loader)}')

    # Validation loop
    model.eval()
    with torch.no_grad():
        valid_loss = 0
        for mfcc, label in valid_loader:
            mfcc, label = mfcc.to(device), label.to(device)
            output = model(mfcc)
            loss = criterion(output, label)
            valid_loss += loss.item()

        print(f'Validation Loss: {valid_loss / len(valid_loader)}')
