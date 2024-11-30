import os

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