from sound_scape.backend.Models import whisper_specrnet, rawgat, xlsr, vocoder, CLAD
import os
from tqdm import tqdm
from colorama import Fore, init
import csv
init(autoreset=True)

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__),"val_files/"))
WHISPER_ISO_CSV = os.path.join(base_path, "whisper_iso.csv")
WHISPER_NORM_CSV = os.path.join(base_path, "whisper_norm.csv")
RAWGAT_ISO_CSV = os.path.join(base_path, "rawgat_iso.csv")
RAWGAT_NORM_CSV = os.path.join(base_path, "rawgat_norm.csv")
XLSR_ISO_CSV = os.path.join(base_path, "xlsr_iso.csv")
XLSR_NORM_CSV = os.path.join(base_path, "xlsr_norm.csv")
VOCODER_ISO_CSV = os.path.join(base_path, "vocoder_iso.csv")
VOCODER_NORM_CSV = os.path.join(base_path, "vocoder_norm.csv")
CLAD_ISO_CSV = os.path.join(base_path, "clad_iso.csv")
CLAD_NORM_CSV = os.path.join(base_path, "clad_norm.csv")

os.mkdir(base_path) if not os.path.exists(base_path) else None

def skip_lines(csv_file, model_path, real_files, fake_files, correct_label_col=4, file_path_col=2, model_path_col=1):
    """
    Skip the lines already processed by a model
    """
    with open(csv_file, 'r', newline='') as f:
        reader = csv.reader(f)
        # read lines as a csv
        num_real_skipped = 0
        num_fake_skipped = 0
        i = 0
        reader = csv.reader(f)
        for row in reader:
            if model_path.strip() in row[model_path_col].strip():
                # print(f"[{Fore.BLUE}{row[file_path_col]}] | [{Fore.LIGHTRED_EX}{real_files[i]}] ({Fore.YELLOW}{row[correct_label_col]})")
                
                # get the real and fake files
                if "Real" in row[correct_label_col]:
                    try:
                        real_files.remove(row[file_path_col])
                    except:
                        pass
                    num_real_skipped += 1
                elif "Fake" in row[correct_label_col]:
                    try:
                        fake_files.remove(row[file_path_col])
                    except:
                        pass

                    num_fake_skipped += 1

        if num_real_skipped > 0 or num_fake_skipped > 0:
            print(f"{Fore.YELLOW}Skipping {num_real_skipped} real files and {num_fake_skipped} fake files for model: " + model_path)
        else:
            print(f"{Fore.GREEN}Model not yet evaluated " + model_path)

        # real_files = real_files[num_real_skipped:]
        # fake_files = fake_files[num_fake_skipped:]

    return real_files, fake_files


def _basic_val(iso_csv, norm_csv, model, device, real_files=None, fake_files=None, auto_gen_files=True, isolated=False):
    """
    Validate the model with the given files.
    """
    if isolated:
        csv_file = iso_csv
    else:
        csv_file = norm_csv

    if auto_gen_files:
        real_files, fake_files = get_val_files(model, dataset_path="../../soundscape-dataset", isolated=isolated)
    

    # header
    # model, file, label, correct_label, certainty, isolated 

    # skip lines if model path already in csv
    model_path = model[1]
    real_fake = skip_lines(csv_file, model_path, real_files, fake_files)
    if real_fake is None:
        return None
    real_files, fake_files = real_fake
    model = model[0](model_path=model[1], device=model[2])
    bar = tqdm(total=len(real_files) + len(fake_files), desc="Validating", unit="file")

    # append to file
    with open(csv_file, 'a') as f:
        # write header if not exists
        if os.stat(csv_file).st_size == 0:
            f.write("model_name,model_path,file,label,correct_label,certainty,isolated\n")

        for file in real_files:
            pred, label = model.raw_eval(file)
            f.write(f"{model.name}, {model.model_path}, {file}, {label}, Real, {pred}, {isolated}\n")
            bar.update(1)
            bar.refresh()

        for file in fake_files:
            pred, label = model.raw_eval(file)
            f.write(f"{model.name}, {model.model_path}, {file}, {label}, Fake, {pred}, {isolated}\n")
            bar.update(1)
            bar.refresh()
        bar.close()

def val_whisper(model: whisper_specrnet, device, real_files=None, fake_files=None, auto_gen_files=True, isolated=False):
    """
    Validate the whisper model with the given files.
    """
    def create_model(model_path, device):
        whisper_specrnet(weights_path=model_path, device=device)
    
    model = [create_model, model, device]
    _basic_val(WHISPER_ISO_CSV, WHISPER_NORM_CSV, model, device, real_files, fake_files, auto_gen_files, isolated)

def val_xlsr(model: xlsr, device, real_files=None, fake_files=None, auto_gen_files=True, isolated=False):
    """
    Validate the xlsr model with the given files.
    """
    def create_model(model_path, device):
        return xlsr(model_path=model_path, device=device)
    
    model = [create_model, model, device]
    # model = xlsr(model_path=model, device=device)
    # _basic_val(XLSR_ISO_CSV, XLSR_NORM_CSV, model, device, real_files, fake_files, auto_gen_files, isolated)
    _basic_val(XLSR_ISO_CSV, XLSR_NORM_CSV, model, device, real_files, fake_files, auto_gen_files, isolated)

def val_CLAD(model: CLAD, device, real_files=None, fake_files=None, auto_gen_files=True, isolated=False):
    """
    Validate the CLAD model with the given files.
    """
    def create_model(model_path, device):
        return CLAD(model_path=model_path, device=device)
    
    model = [create_model, model, device]
    _basic_val(CLAD_ISO_CSV, CLAD_NORM_CSV, model, device, real_files, fake_files, auto_gen_files, isolated)

def val_vocoder(model : vocoder, device, real_files=None, fake_files=None, auto_gen_files=True, isolated=False):
    """
    Validate the model with the given files.
    """

    if isolated:
        csv_file = VOCODER_ISO_CSV
    else:
        csv_file = VOCODER_NORM_CSV

    if auto_gen_files:
        real_files, fake_files = get_val_files(model, dataset_path="../../soundscape-dataset", isolated=isolated)
    

    # header
    # model, file, label, correct_label, certainty, binary[0], binary[1], multi[0], multi[1], isolated
    # skip lines if model path already in csv
    real_fake = skip_lines(csv_file, model, real_files, fake_files)
    if real_fake is None:
        return None
    real_files, fake_files = real_fake

    model = vocoder(device=device, model_path=model)
    bar = tqdm(total=len(real_files) + len(fake_files), desc="Validating", unit="file")
    # append to file
    with open(csv_file, 'a') as f:
        # write header if not exists
        if os.stat(csv_file).st_size == 0:
            f.write("model_name,model_path,file,label,correct_label,binary[0],binary[1],multi[0],multi[1],isolated\n")
        
        for file in real_files:
            multi, binary, label, pred = model.raw_eval(file)
            f.write(f"{model.name}, {model.model_path}, {file}, {label}, Real, {binary[0]}, {binary[1]}, {multi[0]}, {multi[1]}, {isolated}\n")
            bar.update(1)
            bar.refresh()

        for file in fake_files:
            multi, binary, label, pred = model.raw_eval(file)
            f.write(f"{model.name}, {model.model_path}, {file}, {label}, Fake, {binary[0]}, {binary[1]}, {multi[0]}, {multi[1]}, {isolated}\n")
            bar.update(1)
            bar.refresh()
        bar.close()

def val_rawgat(model: rawgat, device,real_files=None, fake_files=None, auto_gen_files=True, isolated=False):
    """
    Validate the model with the given files.
    """

    if isolated:
        csv_file = RAWGAT_ISO_CSV
    else:
        csv_file = RAWGAT_NORM_CSV
    
    if auto_gen_files:
        real_files, fake_files = get_val_files(model, dataset_path="../../soundscape-dataset", isolated=isolated)

    # skip lines if model path already in csv
    real_fake = skip_lines(csv_file, model, real_files, fake_files)
    if real_fake is None:
        return None
    real_files, fake_files = real_fake
    model = rawgat(model_path=model, device=device)
    
    # header
    # model, file, label, correct_label, raw, isolated
    bar = tqdm(total=len(real_files) + len(fake_files), desc="Validating", unit="file")
    
    # append to file
    with open(csv_file, 'a') as f:
        # write header if not exists
        if os.stat(csv_file).st_size == 0:
            f.write("model_name,model_path,file,label,correct_label,raw,isolated\n")
        
        for file in real_files:
            raw, label = model.raw_eval(file)
            f.write(f"{model.name}, {model.model_path}, {file}, {label}, Real, {raw}, {isolated}\n")
            bar.update(1)
            bar.refresh()
        for file in fake_files:
            raw, label = model.raw_eval(file)
            f.write(f"{model.name}, {model.model_path}, {file}, {label}, Fake, {raw}, {isolated}\n")
            bar.update(1)
            bar.refresh()
        bar.close()

def get_val_files(self, dataset_path="../../soundscape-dataset", isolated=False):
    """
    returns real_files, fake_files
    """
    eval_path = os.path.join(dataset_path, "eval")
    iso_val = "-isolated" if isolated else ""
    real_path = os.path.join(eval_path, "bonafide" + iso_val)
    fake_path = os.path.join(eval_path, "fake" + iso_val)

    # return only mp3 files
    return [os.path.join(real_path, x) for x in os.listdir(real_path) if x.endswith(".mp3")], [os.path.join(fake_path, x) for x in os.listdir(fake_path) if x.endswith(".mp3")]

if __name__ == "__main__":

    def get_pth_in_folder(folder_path):
        """
        Get all pth files in a folder
        """
        return [os.path.join(folder_path, x) for x in os.listdir(folder_path) if x.endswith(".pth")]

    model_paths = [
        "/Users/christiankilduff/Deepfake_Detection_Resources/To Test/xlsr/og_58.75_epoch_121.pth",
        "/Users/christiankilduff/Deepfake_Detection_Resources/To Test/xlsr/og_61_epoch_147.pth"
    ]
    # folder_path = ""
    # model_paths = get_pth_in_folder(folder_path)

    print("\n\n------------\n")
    for model_path in model_paths:
        print()

        print(f"{Fore.CYAN}Eval isolated on " + model_path)
        val_xlsr(model=model_path, device="mps", isolated=True)

        print()

        print(f"{Fore.CYAN}Eval non isolated on " + model_path)
        val_xlsr(model=model_path, device="mps", isolated=False)

