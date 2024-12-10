import os 
from datasets import Dataset
import torchaudio
from transformers import AutoFeatureExtractor

TRAIN_SAVE_FOLDER = "processed_train_dataset_wav2vec_SVDD"
DEV_SAVE_FOLDER = "processed_dev_dataset_wav2vec_SVDD"

def load_from_disk():
    """
    Loads pregenerated datasets from disk.

    Usage:
        train_dataset, dev_dataset = load_from_disk()

    Returns:
        The training dataset and development dataset.
    """
    if not os.path.exists(TRAIN_SAVE_FOLDER) or not os.path.exists(DEV_SAVE_FOLDER):
        raise FileNotFoundError("Preprocessed datasets not found. Run MapWav2VecSVDDDataset.py to generate them.")
    
    return Dataset.load_from_disk(TRAIN_SAVE_FOLDER), Dataset.load_from_disk(DEV_SAVE_FOLDER)

class SVDDDataset:
    def __init__(self, train_audio_dir, train_txt, dev_audio_dir, dev_txt):
        self.train_folder = train_audio_dir
        self.train_txt = train_txt
        self.dev_folder = dev_audio_dir
        self.dev_txt = dev_txt

        self.dev_dataset = self.create_dataset(dev_txt, dev_audio_dir)
        self.train_dataset = self.create_dataset(train_txt, train_audio_dir)
    
    def create_dataset(self, label_txt_path, audio_files_path):
        dataset = []
        with open(label_txt_path, "r") as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split(" ")
                audio_path = os.path.join(audio_files_path, parts[2] + ".flac")
                label = 1 if parts[-1].strip().lower() == "deepfake" else 0
                # not all paths in txt exist as they are from other datasets I did not download
                if os.path.exists(audio_path):  # Check if file exists
                    dataset.append((audio_path, label))
                # else:
                #     print(f"Warning: File not found {audio_path}")
        return dataset
def preprocess_function(batch):
    audio_arrays = []
    lengths = []
    
    for audio_path in batch["audio"]:
        waveform, sample_rate = torchaudio.load(audio_path)

        # Resample to match feature extractor's expected sample rate
        if sample_rate != feature_extractor.sampling_rate:
            resampler = torchaudio.transforms.Resample(sample_rate, feature_extractor.sampling_rate)
            waveform = resampler(waveform)
        
        audio_arrays.append(waveform.squeeze(0).numpy())
        lengths.append(waveform.shape[1])  # Store the raw length
    
    # Process waveforms with the feature extractor (dynamic padding within the batch)
    inputs = feature_extractor(
        audio_arrays,
        sampling_rate=feature_extractor.sampling_rate,
        padding="longest",  # Pads only to the longest sequence in the batch
        return_tensors="pt",
    )
    inputs["lengths"] = lengths  # Keep track of original lengths for processing
    return inputs

def main():

    # Initialize feature extractor for Wav2Vec2
    feature_extractor = AutoFeatureExtractor.from_pretrained("facebook/wav2vec2-base")


    # Paths to dataset
    training_folder = "/Users/christiankilduff/Deepfake_Detection_Resources/Datasets/SVDD/train"
    dev_folder = "/Users/christiankilduff/Deepfake_Detection_Resources/Datasets/SVDD/dev"

    SVDD_data = SVDDDataset(
        train_txt=training_folder + "/train.txt",
        train_audio_dir=training_folder + "/train_set",
        dev_audio_dir=dev_folder + "/dev_set",
        dev_txt=dev_folder + "/dev.txt"
    )

    # Create a HuggingFace Dataset
    train_dataset = Dataset.from_dict({
        "audio": [item[0] for item in SVDD_data.train_dataset],
        "label": [item[1] for item in SVDD_data.train_dataset]
    })
    dev_dataset = Dataset.from_dict({
        "audio": [item[0] for item in SVDD_data.dev_dataset],
        "label": [item[1] for item in SVDD_data.dev_dataset]
    })

    # Preprocess and save the training dataset
    train_dataset = train_dataset.map(preprocess_function, remove_columns=["audio"], batched=True)
    train_dataset.save_to_disk(TRAIN_SAVE_FOLDER)

    # Preprocess and save the development dataset
    dev_dataset = dev_dataset.map(preprocess_function, remove_columns=["audio"], batched=True)
    dev_dataset.save_to_disk(DEV_SAVE_FOLDER)

# Ensure main() runs only if this file is executed directly
if __name__ == "__main__":
    main()