from sound_scape.backend.Models import whisper_specrnet, rawgat, xlsr, vocoder, CLAD
import os
from tqdm import tqdm

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
    
    bar = tqdm(total=len(real_files) + len(fake_files), desc="Validating", unit="file")

    # header
    # model, file, label, correct_label, certainty, isolated 

    # append to file
    with open(csv_file, 'a') as f:
        # write header if not exists
        if os.stat(csv_file).st_size == 0:
            f.write("model_name, model_path, file, label, correct_label, certainty, isolated\n")
        
        for file in real_files:
            pred, label = model.raw_eval(file)
            f.write(f"{model.name}, {model.model_path}, {file}, {label}, Real, {pred}, {isolated}\n")
            bar.update(1)

        for file in fake_files:
            pred, label = model.raw_eval(file)
            f.write(f"{model.name}, {model.model_path}, {file}, {label}, Fake, {pred}, {isolated}\n")
            bar.update(1)
        bar.close()

def val_whisper(model: whisper_specrnet, device, real_files=None, fake_files=None, auto_gen_files=True, isolated=False):
    """
    Validate the whisper model with the given files.
    """
    model = whisper_specrnet(weights_path=model, device=device)
    _basic_val(WHISPER_ISO_CSV, WHISPER_NORM_CSV, model, device, real_files, fake_files, auto_gen_files, isolated)

def val_xlsr(model: xlsr, device, real_files=None, fake_files=None, auto_gen_files=True, isolated=False):
    """
    Validate the xlsr model with the given files.
    """
    model = xlsr(model_path=model, device=device)
    _basic_val(XLSR_ISO_CSV, XLSR_NORM_CSV, model, device, real_files, fake_files, auto_gen_files, isolated)

def val_CLAD(model: CLAD, device, real_files=None, fake_files=None, auto_gen_files=True, isolated=False):
    """
    Validate the CLAD model with the given files.
    """
    model = CLAD(model_path=model, device=device)
    _basic_val(CLAD_ISO_CSV, CLAD_NORM_CSV, model, device, real_files, fake_files, auto_gen_files, isolated)

def val_vocoder(model : vocoder, device, real_files=None, fake_files=None, auto_gen_files=True, isolated=False):
    """
    Validate the model with the given files.
    """
    model = vocoder(device=device, model_path=model)

    if isolated:
        csv_file = VOCODER_ISO_CSV
    else:
        csv_file = VOCODER_NORM_CSV

    if auto_gen_files:
        real_files, fake_files = get_val_files(model, dataset_path="../../soundscape-dataset", isolated=isolated)
    
    bar = tqdm(total=len(real_files) + len(fake_files), desc="Validating", unit="file")

    # header
    # model, file, label, correct_label, certainty, binary[0], binary[1], multi[0], multi[1], isolated

    # append to file
    with open(csv_file, 'a') as f:
        # write header if not exists
        if os.stat(csv_file).st_size == 0:
            f.write("model_name, model_path, file, label, correct_label, binary[0], binary[1], multi[0], multi[1], isolated\n")
        
        for file in real_files:
            multi, binary, label, pred = model.raw_eval(file)
            f.write(f"{model.name}, {model.model_path}, {file}, {label}, Real, {binary[0]}, {binary[1]}, {multi[0]}, {multi[1]}, {isolated}\n")
            bar.update(1)

        for file in fake_files:
            multi, binary, label, pred = model.raw_eval(file)
            f.write(f"{model.name}, {model.model_path}, {file}, {label}, Fake, {binary[0]}, {binary[1]}, {multi[0]}, {multi[1]}, {isolated}\n")
            bar.update(1)
        bar.close()

def val_rawgat(model: rawgat, device,real_files=None, fake_files=None, auto_gen_files=True, isolated=False):
    """
    Validate the model with the given files.
    """
    model = rawgat(model_path=model, device=device)

    if isolated:
        csv_file = RAWGAT_ISO_CSV
    else:
        csv_file = RAWGAT_NORM_CSV
    
    if auto_gen_files:
        real_files, fake_files = get_val_files(model, dataset_path="../../soundscape-dataset", isolated=isolated)
    # header
    # model, file, label, correct_label, raw, isolated
    bar = tqdm(total=len(real_files) + len(fake_files), desc="Validating", unit="file")

    # append to file
    with open(csv_file, 'a') as f:
        # write header if not exists
        if os.stat(csv_file).st_size == 0:
            f.write("model_name, model_path, file, label, correct_label, raw, isolated\n")
        
        for file in real_files:
            raw, label = model.raw_eval(file)
            f.write(f"{model.name}, {model.model_path}, {file}, {label}, Real, {raw}, {isolated}\n")
            bar.update(1)
        for file in fake_files:
            raw, label = model.raw_eval(file)
            f.write(f"{model.name}, {model.model_path}, {file}, {label}, Fake, {raw}, {isolated}\n")
            bar.update(1)
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
    model_paths = [
        "/Users/christiankilduff/Deepfake_Detection_Resources/To Test/xlsr/og_58.75_epoch_121.pth",
        "/Users/christiankilduff/Deepfake_Detection_Resources/To Test/xlsr/og_61_epoch_147.pth"
    ]
    for model_path in model_paths:
        print()
        print("Eval isolated on " + model_path)
        val_xlsr(model=model_path, device="mps", isolated=True)
        print()
        print("Eval non isolated on " + model_path)
        val_xlsr(model=model_path, device="mps", isolated=False)

