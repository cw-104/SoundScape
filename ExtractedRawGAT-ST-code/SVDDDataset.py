import os
import csv

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
        self.csv_dataset = []
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
                    self.csv_dataset.append({
                        "file_path": audio_path,
                        "speaker_id": singer_id,
                        "label": label
                    })

                # else:
                #     print(f"Warning: File not found {audio_path}")
        return dataset
    def save_to_csv(self, _csv_folder_path, overwrite=False):
        self.csv_folder_path = os.path.expanduser(_csv_folder_path)
        if not os.path.exists(self.csv_folder_path):
            if not os.mkdir(self.csv_folder_path):
                raise RuntimeError("cannot not create final level folder")
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
            write_csv(self.csv_dev_path, self.dev_dataset)
            print(f"Wrote {self.csv_dev_fname} to {self.csv_dev_path}")

        # write train csv
        if os.path.exists(self.csv_train_path):
            print(f"Train csv exists: {'Overwriting' if overwrite else 'not overwriting, set overwite=True or delete csv to rewrite'}")
        else:
            write_csv(self.csv_train_path, self.train_dataset)
            print(f"Wrote {self.csv_train_fname} to {self.csv_train_path}")