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
    # check if csv file exists
    if not os.path.exists(csv_file):
        return real_files, fake_files # do nothing

    with open(csv_file, 'r', newline='') as f:
        reader = csv.reader(f)
        # read lines as a csv
        num_real_skipped = 0
        num_fake_skipped = 0
        reader = csv.reader(f)
        for row in reader:
            if model_path.strip() in row[model_path_col].strip():
                # get the real and fake files
                if "Real" in row[correct_label_col]:
                    num_real_skipped += 1
                elif "Fake" in row[correct_label_col]:
                    num_fake_skipped += 1

        if num_real_skipped > 0 or num_fake_skipped > 0:
            if num_real_skipped >= len(real_files):
                real_files = []
                # print(f"{Fore.GREEN}Skipping all real files")
            else:
                real_files = real_files[num_real_skipped:]
                # print(f"{Fore.YELLOW}Skipping {num_real_skipped} real")
            if num_fake_skipped >= len(fake_files):
                fake_files = []
                # print(f"{Fore.YELLOW}Skipping all fake files")
            else:
                fake_files = fake_files[num_fake_skipped:]
                # print(f"{Fore.GREEN}Skipping {num_fake_skipped} fake")
        # else:
            # print(f"{Fore.GREEN}Model not yet evaluated " + model_path)

        # real_files = real_files[num_real_skipped:]
        # fake_files = fake_files[num_fake_skipped:]
    return real_files, fake_files


def _basic_val(iso_csv, norm_csv, model, device, real_files=None, fake_files=None, auto_gen_files=True, isolated=False, bar_desc="Validating"):
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
    bar = tqdm(total=len(real_files) + len(fake_files), desc=bar_desc, unit="file", leave=False)

    to_append = []

    for file in real_files:
        pred, label = model.raw_eval(file)
        # to_append.append(f"{model.name}, {model.model_path}, {file}, {label}, Real, {pred}, {isolated}\n")
        to_append.append([f"{model.name}", f"{model.model_path}", f"{file}", f"{label}", "Real", f"{pred}", f"{isolated}"])
        bar.update(1)
        bar.refresh()

    for file in fake_files:
        pred, label = model.raw_eval(file)
        # to_append.append(f"{model.name}, {model.model_path}, {file}, {label}, Fake, {pred}, {isolated}\n")
        to_append.append([f"{model.name}", f"{model.model_path}", f"{file}", f"{label}", "Fake", f"{pred}", f"{isolated}"])
        bar.update(1)
        bar.refresh()

    # append to file
    with open(csv_file, 'a') as f:
        # write header if not exists
        if os.stat(csv_file).st_size == 0:
            f.write("model_name,model_path,file,label,correct_label,certainty,isolated\n")
        # for line in to_append:
        #     # f.write(line)
        csv_writer = csv.writer(f)
        csv_writer.writerows(to_append)
    bar.close()

def val_whisper(model, device, real_files=None, fake_files=None, auto_gen_files=True, isolated=False, bar_desc="Validating"):
    """
    Validate the whisper model with the given files.
    """
    def create_model(model_path, device):
        return whisper_specrnet(weights_path=model_path, device=device)
    
    model = [create_model, model, device]
    _basic_val(WHISPER_ISO_CSV, WHISPER_NORM_CSV, model, device, real_files, fake_files, auto_gen_files, isolated, bar_desc)

def val_xlsr(model, device, real_files=None, fake_files=None, auto_gen_files=True, isolated=False, bar_desc="Validating"):
    """
    Validate the xlsr model with the given files.
    """
    def create_model(model_path, device):
        return xlsr(model_path=model_path, device=device)
    
    model = [create_model, model, device]
    # model = xlsr(model_path=model, device=device)
    # _basic_val(XLSR_ISO_CSV, XLSR_NORM_CSV, model, device, real_files, fake_files, auto_gen_files, isolated)
    _basic_val(XLSR_ISO_CSV, XLSR_NORM_CSV, model, device, real_files, fake_files, auto_gen_files, isolated, bar_desc)

def val_CLAD(model, device, real_files=None, fake_files=None, auto_gen_files=True, isolated=False, bar_desc="Validating"):
    """
    Validate the CLAD model with the given files.
    """
    def create_model(model_path, device):
        return CLAD(model_path=model_path, device=device)
    
    model = [create_model, model, device]
    _basic_val(CLAD_ISO_CSV, CLAD_NORM_CSV, model, device, real_files, fake_files, auto_gen_files, isolated, bar_desc)

def val_vocoder(model, device, real_files=None, fake_files=None, auto_gen_files=True, isolated=False, bar_desc="Validating"):
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
    bar = tqdm(total=len(real_files) + len(fake_files), desc=bar_desc, unit="file", leave=False)
    # append to file
    to_append = []

    for file in real_files:
        multi, binary, label, pred = model.raw_eval(file)
        # to_append.append(f"{model.name}, {model.model_path},{file},{label},Real,{binary[0]},{binary[1]},{multi[0]},{multi[1]},{isolated}\n")
        to_append.append([f"{model.name}", f"{model.model_path}", f"{file}", f"{label}", "Real", f"{binary[0]}", f"{binary[1]}", f"{multi[0]}", f"{multi[1]}", f"{isolated}"])
        bar.update(1)
        bar.refresh()

    for file in fake_files:
        multi, binary, label, pred = model.raw_eval(file)
        # to_append.append(f"{model.name},{model.model_path},{file},{label}, Fake,{binary[0]},{binary[1]},{multi[0]},{multi[1]},{isolated}\n")
        to_append.append([f"{model.name}", f"{model.model_path}", f"{file}", f"{label}", "Fake", f"{binary[0]}", f"{binary[1]}", f"{multi[0]}", f"{multi[1]}", f"{isolated}"])
        bar.update(1)
        bar.refresh()
    
    with open(csv_file, 'a') as f:
        # write header if not exists
        if os.stat(csv_file).st_size == 0:
            f.write("model_name,model_path,file,label,correct_label,binary[0],binary[1],multi[0],multi[1],isolated\n")
        csv_writer = csv.writer(f)
        csv_writer.writerows(to_append)
    bar.close()

def val_rawgat(model, device,real_files=None, fake_files=None, auto_gen_files=True, isolated=False, bar_desc="Validating"):
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
    # bar = tqdm(total=len(real_files) + len(fake_files), desc=bar_desc, unit="file", leave=False)
    
    to_append = []
    results_real = model.raw_eval_multi(real_files)
    for res in results_real:
        raw, label, file = res.raw_value, res.classification, res.file_name
        # to_append.append(f"{model.name}, {model.model_path}, {file}, {label}, Real, {raw}, {isolated}\n")
        to_append.append([f"{model.name}", f"{model.model_path}", f"{file}", f"{label}", "Real", f"{raw}", f"{isolated}"])
        # bar.update(1)
        # bar.refresh()
    results_fake = model.raw_eval_multi(fake_files)
    for res in results_fake:
        raw, labe, file = res.raw_value, res.classification, res.file_name
        to_append.append(f"{model.name}, {model.model_path}, {file}, {label}, Fake, {raw}, {isolated}\n")
        to_append.append([f"{model.name}", f"{model.model_path}", f"{file}", f"{label}", "Fake", f"{raw}", f"{isolated}"])
        # bar.update(1)
        # bar.refresh()

    # append to file
    with open(csv_file, 'a') as f:
        # write header if not exists
        if os.stat(csv_file).st_size == 0:
            f.write("model_name,model_path,file,label,correct_label,raw,isolated\n")
        # for line in to_append:
        #     f.write(line)
        csv_writer = csv.writer(f)
        csv_writer.writerows(to_append)
    # bar.close()

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
    # ARGS
    # device optional def "cuda"

    # any length list of paired path, name, path, name, ...
    from sys import argv
    args =  argv
    models = []
    device = "cuda"
    max_workers = 4
    if len(args) > 1:
        args = args[1:]
        print(args)
        if not "--device" in args:
            print(f"{Fore.BLUE}Defaulting to device {device} (--device <cpu/mps/cuda> to specficy)")
        else:
            device = args.pop(args.index("--device") + 1)
            print(f"{Fore.BLUE}Using device {device}")
            args.remove("--device")

        if not "--max-workers" in args:
            print(f"{Fore.BLUE}Defaulting to max-workers {max_workers} (--max-workers <int> to specficy)")
        else:
            max_workers = int(args.pop(args.index("--max-workers") + 1))
            print(f"{Fore.BLUE}Using max-workers {max_workers}")
            args.remove("--max-workers")


        if len(args) % 2 != 0 or len(args) < 2:
            print(f"{Fore.RED}Error: Must provide paired model path and name (even number of pairs)")
            print(f"{Fore.YELLOW}Example command: --device cpu whisper \"path/to/whisper model path\"")
            exit()
        for i in range(0, len(args), 2):
            name = args[i]
            path = args[i + 1]
            if name not in ["whisper", "xlsr", "rawgat", "vocoder", "clad"]:
                print(f"{Fore.RED}Error: Model name ({name}) invalid; must be one of [whisper, xlsr, rawgat, vocoder, clad]")
                print(f"{Fore.YELLOW}Example command: --device cpu whisper \"path/to/whisper model path\"")
                exit()
            if not os.path.exists(path):
                print(f"{Fore.RED}Error: path does not exist (pair: {name} {path})")
                exit()
            models.append((path.strip(), name.strip()))
            print(f"{Fore.GREEN}Will eval {name} model(s) in folder {path}")
    print()

    def get_pth_in_folder(folder_path):
        """
        Get all pth files in a folder
        """
        return [os.path.join(folder_path, x) for x in os.listdir(folder_path) if x.endswith(".pth")]


    import concurrent.futures
    import time

    model_packs = []
    for folder, name in models:
        models = get_pth_in_folder(folder)
        for model in models:
            # True = isolated
            model_packs.append([name, model, device, True])
            # False = not isolated
            model_packs.append([name, model, device, False])


    global total
    total = len(model_packs)
    def val(i, packed):
        global total
        name, model_path, device, isolated = packed
        # print(f"evaling model {i} ({name}) path: {model_path} on device {device} isolated: {isolated}")
        isolated_txt = "isolated" if isolated else "not isolated"
        desc = f"Val {name} model {os.path.basename(model_path)} {isolated_txt} {i}/{total}"
        if "whisper" in name:
            return val_whisper(model=model, device=device, isolated=isolated, bar_desc=desc)
        elif "xlsr" in name:
            return val_xlsr(model=model, device=device, isolated=isolated, bar_desc=desc)
        elif "rawgat" in name:
            return val_rawgat(model=model, device=device, isolated=isolated, bar_desc=desc)
        elif "vocoder" in name:
            return val_vocoder(model=model, device=device, isolated=isolated, bar_desc=desc)
        elif "clad" in name:
            return val_CLAD(model=model, device=device, isolated=isolated)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # results = [executor.submit(model, i) for i, model in enumerate(model_paths)]
        results = list(tqdm(executor.map(val, [i, (name, model, device, isolated) in ((i, pack) for i, pack in enumerate(model_packs))]), total=len(model_packs), desc="Tasks Complete",position=max_workers+1))
