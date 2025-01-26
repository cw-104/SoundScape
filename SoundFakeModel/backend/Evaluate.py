import os, io, ffmpeg, yaml, torch

from tqdm import tqdm
import numpy as np
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from enum import IntEnum
from backend.Results import DfResultHandler
from Base_Path import get_path_relative_base
from backend.whisper_eval import evaluate_nn
from backend.whisper_specrnet import WhisperSpecRNet, set_seed

class Models(IntEnum):
    SOUNDSCAPE = 0    
    RAWGAT = 1

def get_best_device():
    # Check if CUDA devices are available.
    if torch.cuda.is_available():
        return "cuda"
    
    # If not, check for MPS (Metal Performance Shaders) support on Apple Silicon Macs with macOS 12.0 or later.
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_built() and torch.backends.mps.is_available():
        return "mps"

    else:
        # Default to CPU if neither CUDA nor MPS is available
        return "cpu"

def file_paths_in_folder(folderpath):
    return [os.path.join(folderpath, fname) for fname in os.listdir(folderpath) if fname.endswith(('.wav', '.mp3', '.flac'))]

    
import librosa
import torch

def load_audio_librosa(filepath, target_sr=16000):
    waveform, sample_rate = librosa.load(filepath, sr=target_sr, mono=True)
    waveform = torch.tensor(waveform).unsqueeze(0)  # Add channel dimension
    return waveform, sample_rate

def pad(x, max_len=64600):
        x_len = x.shape[0]
        if x_len >= max_len:
            return x[:max_len]
        num_repeats = int(max_len / x_len) + 1
        padded_x = np.tile(x, (1, num_repeats))[:, :max_len][0]
        return padded_x
class DeepfakeClassificationModel:

    def __init__(self, modeltype=Models.RAWGAT, result_handler=DfResultHandler(-1.72, "DeepFake", "Real", 10, .90)):
        '''
        Initializes the model
        '''
        self.result_handler=result_handler
        self.device = get_best_device()
        if modeltype == Models.RAWGAT:
            from backend.RawGATmodel import RawGAT_ST
            model_path = "pretrained_models/RawGAT/RawGAT.pth"
            with open("pretrained_models/RawGAT/model_config_RawGAT_ST.yaml", 'r') as f_yaml:
                config = yaml.safe_load(f_yaml)  
        
            # Extract only the model-related part of the configuration
            model_config = config['model']
            # Instantiate the model with the correct configuration
            self.model = RawGAT_ST(model_config, self.device)
            self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
            self.model = self.model.to(self.device)
            print(f"Loaded model from {model_path}")
            


        
            




    def evaluate_file(self, file_path):
        '''
        Method: 
        Evaluates a single audio file for deepfake classification
        
        ------------
        Parameters:
        file: {str} file path to audio file

        ------------
        Returns:
        {Result} result of the evaluation 
        '''
        return self.evaluate_multi_files([file_path])[0]
    
    def evaluate_folder(self, folderpath, print_r=False, progress_bar=False):
        return self.evaluate_multi_files(file_paths_in_folder(folderpath), print_r=print_r,progress_bar=progress_bar)

    def evaluate_multi_files(self, filepaths, print_r=False, progress_bar=False):
        transform = transforms.Compose([
            lambda x: pad(x),  
            lambda x: torch.Tensor(x)
        ])
        dataset = EvalDataset(filepaths, transform=transform)
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False)

        # Perform evaluation
        if print_r:
            print("✅: Real | ❌: Deepfake")
            print("--------------------------------------------------\n")
        # total count of files
        results = []
        self.model.eval()
        if progress_bar:
            print()
        with torch.no_grad():
            for batch_idx, (batch_x, batch_y, batch_meta) in enumerate(tqdm(data_loader, desc="Evaluating Authenticity of Files") if progress_bar else data_loader):
                batch_x = batch_x.to(self.device)
                output = self.model(batch_x, Freq_aug=False)
                score = output[:, 1].item()  

                if print_r:
                    # res = "✅" if score > min_real_score else "❌"
                    print(f"{batch_meta['filename'][0]}")
                    print(f"Score: {score}")
                    # results.append((batch_meta['filename'][0], score))
                results.append(self.result_handler.generate_result(batch_meta['filename'][0], score))

        return results



class EvalDataset(Dataset):
    def __init__(self, filepaths, transform=None):
        self.filepaths = filepaths
        self.transform = transform
    
    def __len__(self):
        return len(self.filepaths)
    
    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        waveform, sample_rate = load_audio_librosa(filepath)
        
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
        
        if self.transform:
            waveform = self.transform(waveform[0])  # Process only the first channel
        
        # Set a dummy label (0: spoof, 1: bonafide) and metadata
        return waveform, torch.tensor([1]), {'filename': os.path.basename(filepath)}


            
def init_whisper_specrnet(device="", weights_path="", config_path="", threshold=.45, reval_threshold=0, no_sep_threshold=0):
    if device == "":
        device = get_best_device()
    get_best_device()
    if config_path == "":
        config_path = get_path_relative_base("pretrained_models/whisper_specrnet/config.yaml")
    if weights_path == "":
        weights_path = get_path_relative_base("pretrained_models/whisper_specrnet/weights.pth")
    config = yaml.safe_load(open(config_path, "r"))
    model_name, model_parameters = config["model"]["name"], config["model"]["parameters"]

    print(f"loading model...\n")
    model = WhisperSpecRNet(
        input_channels=config.get("input_channels", 1),
        freeze_encoder=config.get("freeze_encoder", False),
        device=device,
    )
    
    model.load_state_dict(torch.load(weights_path, map_location=device))


    seed = config["data"].get("seed", 42)
    set_seed(seed)

    model_rawgat = DeepfakeClassificationModel(result_handler=DfResultHandler(-3, "Fake", "Real", 3, .95))    
    return model, model_rawgat, config, device
def eval_file(f, preloaded_model, config, device):

    # Evaluate a single file
    pred, label = evaluate_nn(
        model=preloaded_model,
        model_config=config["model"],
        device=device,
        single_file=f,  # Specify the single file path
    )
    return pred, 'spoof' if label == 0 else 'real', f"{('spoof' if label == 0 else 'real'):<10} (raw value: {pred:.4f})"
