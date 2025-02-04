from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from tqdm import tqdm
import threading, os, torch, pickle
from queue import Queue

from neural_network.features import extract_features

class AudioClassifier(nn.Module):
    def __init__(self, n_model_features, n_rms, n_magnitude_spectrum,  hidden_size=128):
        super(AudioClassifier, self).__init__()


        print()
        print(f'sizes: {n_model_features}, {n_audio_features}')
        print()

        n_model_features = 2
        self.fc1 = nn.Linear(n_model_features, hidden_size)
        self.fc2 = nn.Linear(n_model_features, hidden_size)
        
        # Branch for the second input set (rms, magnitude_spectrum)
        self.fc3 = nn.Linear(n_rms, hidden_size)
        self.fc3 = nn.Linear(n_magnitude_spectrum, hidden_size)
        
        # Combined layer
        self.fc_combined = nn.Linear(hidden_size * 3, hidden_size)
        self.fc_out = nn.Linear(hidden_size, 2)

    def forward(self, model_features, audio_features):
        # Process the first input set
        print()
        print(model_features[0].shape)
        x1 = F.relu(self.fc1(model_features[0])) 
        x2 = F.relu(self.fc2(model_features[1])) 
        
        # Process the second input set
        # (rms, magnitude_spectrum, whisper_confidence)
        x3 = F.relu(self.fc3(audio_features[0]))
        x4 = F.relu(self.fc3(audio_features[1]))    
        
        # Combine the outputs
        x_combined = torch.cat((x1, x2, x3, x4), dim=1)
        x_combined = F.relu(self.fc_combined(x_combined))
        
        # Final output
        output = self.fc_out(x_combined)
        return output


class LabeledAudioFile():
    def __init__(self, audio_file_path, label):
        self.file_path = audio_file_path
        self.label = label



class AudioDataset(Dataset):
    def __init__(self, model_features, audio_features, labels):
        self.model_features = model_features  # List of tuples [(label1, conf1), (label2, conf2)]
        self.audio_features = audio_features  # List of [rms, magnitude_spectrum, whisper_confidence]
        self.labels = labels # dataset classification labels
        self.n_model_features = model_features[0].shape[1]
        # self.n_model_features = 2
        self.n_magnitude_spectrum = audio_features[0].shape[0]
        self.n_rms = audio_features[0].shape[1]
        print(f'audio features: {audio_features[0].shape[0]}')
        print(f'audio features: {audio_features[0].shape[1]}')

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return (self.model_features[idx], self.audio_features[idx]), self.labels[idx]
    
    
    @staticmethod
    def create_dataset(labeled_audio_files, rawgat_model, whisper_specrnet_model, n_threads=6, save_path=None, batch_size=2):
        if save_path is not None and os.path.exists(save_path):
            raise ValueError("Dataset already exists at the specified path.")

        model_features = []
        audio_features = []
        labels = []
        
        n_files = len(labeled_audio_files)
        audio_files_queue = Queue()
        for labeled_audio_file in labeled_audio_files:
            audio_files_queue.put(labeled_audio_file)
        # clear memory
        labeled_audio_files = None
        
        lock = threading.Lock()
        
        def work(batch_size=batch_size, n_files=n_files):
            while not audio_files_queue.empty():
                model_features_tensor = [] 
                audio_features_tensor = []
                files = []
                with lock:
                    for _ in range(batch_size):
                        if audio_files_queue.empty():
                            break
                        files.append(audio_files_queue.get())

                
                for f in files:
                    mft, aft = extract_features(f.file_path, rawgat_model, whisper_specrnet_model)
                    model_features_tensor.append(mft)
                    audio_features_tensor.append(aft)
                    
                # Append features and label
                with lock:
                    model_features.extend(model_features_tensor)
                    audio_features.extend(audio_features_tensor)
                    labels.extend([f.label for f in files])
                    pbar.update(len(files))  

        with tqdm(total=n_files) as pbar:
            threads = []
            for _ in range(n_threads):
                thread = threading.Thread(target=work)
                thread.start()
                threads.append(thread)
            
            for thread in threads:
                thread.join()
            print('joined threads')

        dataset = AudioDataset(model_features, audio_features, labels)
        if save_path is not None:
            dataset.save_to_file(save_path)
        return dataset
    
    def save_to_file(self, file_path):
        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))
        with open(file_path, 'wb') as f:
            pickle.dump({
                'model_features': self.model_features,
                'audio_features': self.audio_features,
                'labels': self.labels
            }, f)

    @staticmethod
    def load_from_file(file_path):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            return AudioDataset(data['model_features'], data['audio_features'], data['labels'])
    
    

    @staticmethod
    def custom_collate_fn(batch):
        # Unzip the batch into model features, audio features, and labels
        # model_features, audio_features, labels = zip(*batch)
        features, labels = zip(*batch)
        model_features_tensor, audio_features = zip(*features)
        
        
        # Pad audio features
        # Assuming audio_features is a list of tensors or lists of varying lengths
        audio_features_tensor = pad_sequence([torch.tensor(aft, dtype=torch.float32) for aft in audio_features], batch_first=True)
        
        # Convert labels to tensors
        labels_tensor = torch.tensor(labels, dtype=torch.long)  # Assuming labels are integers
        
        return (model_features_tensor, audio_features_tensor), labels_tensor
