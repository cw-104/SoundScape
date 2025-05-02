import os, sys
from sound_scape.backend.Models import xlsr, rawgat, vocoder, whisper_specrnet, CLAD
import csv
from tqdm import tqdm
from colorama import Fore,init
init(autoreset=True)
csv_file = f"final_eval_all_models.csv"

def get_val_files(dataset_path="../../soundscape-dataset", isolated=False):
    """
    returns real_files, fake_files
    """
    eval_path = os.path.join(dataset_path, "eval")
    iso_val = "-isolated" if isolated else ""
    real_path = os.path.join(eval_path, "bonafide" + iso_val)
    fake_path = os.path.join(eval_path, "fake" + iso_val)

    # return only mp3 files
    return [os.path.join(real_path, x) for x in os.listdir(real_path) if x.endswith(".mp3")], [os.path.join(fake_path, x) for x in os.listdir(fake_path) if x.endswith(".mp3")]


def get_pth_in_folder(folder_path):
    """
    Get all pth files in a folder
    """
    return [os.path.join(folder_path, x) for x in os.listdir(folder_path) if x.endswith(".pth") or x.endswith(".pth.tar")]


def get_model_of(type, path):
    if type == "xlsr":
        return xlsr(model_path=path, device="mps")
    elif type == "rawgat":
        return rawgat(model_path=path, device="mps")
    elif type == "vocoder":
        return vocoder(model_path=path, device="mps")
    elif type == "whisper":
        return whisper_specrnet(weights_path=path, device="mps")
    elif type == "clad":
        return CLAD(model_path=path, device="mps")


if __name__ == "__main__":
    # raw args
    argv = sys.argv
    max_workers = 5
    if len(argv) > 1:
        max_workers = int(argv[1])
        print(f"{Fore.BLUE}Max workers set to {max_workers}")
    # create if not exists
    if not os.path.exists(csv_file):
        with open(csv_file, "w") as f:
            writer = csv.writer(f)
            writer.writerow(["Model", "File", "Certainty", "Label","Correct Label", "Isolated"])
    
    models = {
    "whisper": [
        "/Users/christiankilduff/Deepfake_Detection_Resources/SoundScape/src/pretrained_models/whisper_specrnet/weights.pth",
        "/Users/christiankilduff/Deepfake_Detection_Resources/To Test/whisper/ckpt_0.pth",
        "/Users/christiankilduff/Deepfake_Detection_Resources/To Test/whisper/isog_ckpt.pth",
        "/Users/christiankilduff/Deepfake_Detection_Resources/To Test/whisper/short_iso_ckpt.pth"

    ],
    "rawgat": [
        "/Users/christiankilduff/Deepfake_Detection_Resources/SoundScape/src/pretrained_models/RawGAT/RawGAT.pth",
        "/Users/christiankilduff/Deepfake_Detection_Resources/SoundScape/Trained Models/rawgat/iso/epoch_0.pth",
        "/Users/christiankilduff/Deepfake_Detection_Resources/SoundScape/Trained Models/rawgat/iso/epoch_1.pth",
        "/Users/christiankilduff/Deepfake_Detection_Resources/SoundScape/Trained Models/rawgat/iso/epoch_6.pth",
        "/Users/christiankilduff/Deepfake_Detection_Resources/SoundScape/Trained Models/rawgat/noniso/epoch_4.pth",
        "/Users/christiankilduff/Deepfake_Detection_Resources/SoundScape/Trained Models/rawgat/noniso/epoch_342.pth",
        "/Users/christiankilduff/Deepfake_Detection_Resources/SoundScape/Trained Models/rawgat/noniso/epoch_351.pth",
    ],
    "clad": [
        "/Users/christiankilduff/Deepfake_Detection_Resources/SoundScape/src/pretrained_models/CLAD/CLAD_150_10_2310.pth.tar",
        "/Users/christiankilduff/Deepfake_Detection_Resources/SoundScape/Trained Models/clad/pretrained_models/SSCLAD/CLAD_150_10_2310.pth.tar",
        "/Users/christiankilduff/Deepfake_Detection_Resources/SoundScape/Trained Models/clad/pretrained_models/SSCLAD/SS_CLAD_2025-04-08_10-21-37.pth.tar",
        "/Users/christiankilduff/Deepfake_Detection_Resources/SoundScape/Trained Models/clad/pretrained_models/SSCLAD/SS_CLAD_2025-04-09_13-22-53.pth.tar",
        "/Users/christiankilduff/Deepfake_Detection_Resources/SoundScape/Trained Models/clad/pretrained_models/SSCLAD/SS_CLAD_2025-04-09_19-46-37.pth.tar",
        "/Users/christiankilduff/Deepfake_Detection_Resources/SoundScape/Trained Models/clad/pretrained_models/SSCLAD/SS_CLAD_2025-04-15_15-12-12.pth.tar",
        "/Users/christiankilduff/Deepfake_Detection_Resources/SoundScape/Trained Models/clad/pretrained_models/CLAD_150_10_2310.pth.tar"
    ],
    "xlsr": [
        "/Users/christiankilduff/Deepfake_Detection_Resources/SoundScape/src/pretrained_models/XLS-R/epoch_1_58.pth",
        "/Users/christiankilduff/Deepfake_Detection_Resources/SoundScape/src/pretrained_models/XLS-R/MMpaper_model.pth",
        "/Users/christiankilduff/Deepfake_Detection_Resources/SoundScape/Trained Models/xlsr/iso/53_epoch_84.pth",
        "/Users/christiankilduff/Deepfake_Detection_Resources/SoundScape/Trained Models/xlsr/iso/68.88_epoch_13.pth",
        "/Users/christiankilduff/Deepfake_Detection_Resources/SoundScape/Trained Models/xlsr/iso/epoch_11.pth",
        "/Users/christiankilduff/Deepfake_Detection_Resources/SoundScape/Trained Models/xlsr/noniso/epoch_11.pth",
        "/Users/christiankilduff/Deepfake_Detection_Resources/SoundScape/Trained Models/xlsr/noniso/epoch_86.pth",
        "/Users/christiankilduff/Deepfake_Detection_Resources/SoundScape/Trained Models/xlsr/noniso/og_61_epoch_147.pth",
    ],
    "vocoder": [
        "/Users/christiankilduff/Deepfake_Detection_Resources/SoundScape/src/pretrained_models/vocoder/librifake_pretrained_lambda0.5_epoch_25.pth",
        "/Users/christiankilduff/Deepfake_Detection_Resources/SoundScape/Trained Models/vocoder/iso/epoch_0.pth",
        "/Users/christiankilduff/Deepfake_Detection_Resources/SoundScape/Trained Models/vocoder/iso/epoch_1.pth",
        "/Users/christiankilduff/Deepfake_Detection_Resources/SoundScape/Trained Models/vocoder/noniso/63.10_(bin69.33)epoch_64.pth",
        "/Users/christiankilduff/Deepfake_Detection_Resources/SoundScape/Trained Models/vocoder/noniso/63.10_(bin72.33)epoch_60.pth",
        "/Users/christiankilduff/Deepfake_Detection_Resources/SoundScape/Trained Models/vocoder/noniso/64.29(bin99.33)epoch_73.pth",
        "/Users/christiankilduff/Deepfake_Detection_Resources/SoundScape/Trained Models/vocoder/noniso/64.29_(bin66)epoch_20.pth"

    ],
    }

    global iso_real_files, iso_fake_files, og_real_files, og_fake_files
    iso_real_files, iso_fake_files = get_val_files(isolated=True)
    og_real_files, og_fake_files = get_val_files(isolated=False)


    global model_bars
    # resolves to list of function to call creator_or_update_bar on specfic model
    model_bars = {}

    import time
    import concurrent.futures


    total = sum(len(model) for model in models.values())

    def get_processed_files(model_path, csv_file):
        if not os.path.exists(csv_file):
            return []
        """
        files to skip that have already been processed

        header
        "Model", "File", "Certainty", "Label", "Correct Label", "Isolated"
        """
        to_skip = []
        with open(csv_file, "r") as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                col_model = row[0]
                col_file = row[1]
                if col_model.strip() != model_path.strip():
                    continue
                if col_file in iso_real_files or col_file in iso_fake_files or col_file in og_real_files or col_file in og_fake_files:
                    to_skip.append(col_file)
        return to_skip
    def process_model(model_type, model_path):
        global model_bars, models
        if model_type not in model_bars:
            model_bars[model_type] = tqdm(total=len(models[model_type]), desc=model_type, leave=False)
        skip_files = get_processed_files(model_path, csv_file)
        total = len(iso_real_files) + len(iso_fake_files) + len(og_real_files) + len(og_fake_files)
        if len(skip_files) >= total:
            return model_type, []
        # [Model", "File", "Certainty", "Label", "Correct Label", "Isolated"]
        result_rows = []
        model = get_model_of(model_type, model_path)
        bar = tqdm(total=total, desc=f"Processing {model_type}: {os.path.basename(model_path)}", leave=False)
        # isolated
        for file_path in iso_real_files:
            if file_path in skip_files:
                bar.update(1)
                continue
            cert, label = model.evaluate(file_path)
            result_rows.append([model_path, file_path, cert, label, "Real", True])
            bar.update(1)
        for file_path in iso_fake_files:
            if file_path in skip_files:
                bar.update(1)
                continue
            cert, label = model.evaluate(file_path)
            result_rows.append([model_path, file_path, cert, label, "Fake", True])
            bar.update(1)
        # non isolated
        for file_path in og_real_files:
            if file_path in skip_files:
                bar.update(1)
                continue
            cert, label = model.evaluate(file_path)
            result_rows.append([model_path, file_path, cert, label, "Real", False])
            bar.update(1)
        for file_path in og_fake_files:
            if file_path in skip_files:
                bar.update(1)
                continue
            cert, label = model.evaluate(file_path)
            result_rows.append([model_path, file_path, cert, label, "Fake", False])
            bar.update(1)
        return model_type, result_rows
            



    overall_progress = tqdm(total=total, desc="Overall Progress", position=max_workers+3, leave=True)

    

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_model, m, models[m][i]): (m, i) for m in models.keys() for i in range(len(models[m]))}        
        # Process results as they complete
        for future in concurrent.futures.as_completed(futures):
            model_type, rows = future.result()
            model_bars[model_type].update(1)
            overall_progress.update(1)
            with open(csv_file, "a") as f:
                writer = csv.writer(f)
                writer.writerows(rows)
            time.sleep(.01)


