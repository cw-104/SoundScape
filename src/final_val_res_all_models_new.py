from colorama import Fore, init
import os, csv, sys
init(autoreset=True)

csv_all_models = "final_eval_all_models.csv"

def get_results(model_kinds=None):
    # model_kinds none or only contains None values
    if not model_kinds or all([kind is None for kind in model_kinds]):
        model_kinds = ['whisper', 'rawgat', 'xlsr', 'clad', 'vocoder']
    results = {}
    with open(csv_all_models, "r") as f:
        reader = csv.DictReader(f)
        # Model,File,Certainty,Label,Correct Label,Isolated
        for row in reader:
            model = row['Model']
            # if kind in not model_kinds, skip
            if not any(kind in model.lower() for kind in model_kinds):
                continue
            if model not in results:
                results[model] = {
                    'og': {
                        'model_path': model,
                        'raw': [],
                        'total-real': 0,
                        'correct-real': 0,
                        'total-fake': 0,
                        'correct-fake': 0,
                        'accuracy-real': 0,
                        'accuracy-fake': 0,
                        'accuracy': 0,
                    },
                    'iso': {
                        'model_path': model,
                        'raw': [],
                        'total-real': 0,
                        'correct-real': 0,
                        'total-fake': 0,
                        'correct-fake': 0,
                        'accuracy-real': 0,
                        'accuracy-fake': 0,
                        'accuracy': 0,
                    },
                }
            type = 'og'
            if row['Isolated'] == "True":
                type = 'iso'

            filep = row['File']
            certainty = float(row['Certainty'])
            label = row['Label']
            correct_label = row['Correct Label']
            
            if correct_label == "Real":
                results[model][type]['total-real'] += 1
                if correct_label == label:
                    results[model][type]['correct-real'] += 1
            elif correct_label == "Fake":
                results[model][type]['total-fake'] += 1
                if correct_label == label:
                    results[model][type]['correct-fake'] += 1
            
            results[model][type]['raw'].append([filep, certainty, label, correct_label])
    for model in results.keys():
        for type in ['og', 'iso']:
            results[model][type]['accuracy-real'] = results[model][type]['correct-real'] / results[model][type]['total-real'] if results[model][type]['total-real'] > 0 else 0
            results[model][type]['accuracy-fake'] = results[model][type]['correct-fake'] / results[model][type]['total-fake'] if results[model][type]['total-fake'] > 0 else 0
            results[model][type]['accuracy'] = (results[model][type]['accuracy-real'] + results[model][type]['accuracy-fake']) / 2
    return results

def all_results(model_kinds=None):
    # model_kinds none or only contains None values
    if not model_kinds or all([kind is None for kind in model_kinds]):
        model_kinds = ['whisper', 'rawgat', 'xlsr', 'clad', 'vocoder']
    results = get_results()
    for kind in model_kinds:
        print(f"{Fore.BLUE}=======")
        print(f"{Fore.BLUE}------------{kind}------------")
        print(f"{Fore.BLUE}=======")
        for key in results:
            if kind not in key.lower():
                continue
            model_res = results[key]
            for type in ['og', 'iso']:
                model_type_res = model_res[type]
                print(f"{Fore.CYAN}Model: {key} {Fore.YELLOW}| Type: {type} |")
                print(f"Accuracy Real({model_type_res['total-real']} files): {Fore.GREEN}{model_type_res['accuracy-real']*100:.2f}{Fore.RESET} | Accuracy Fake({model_type_res['total-fake']} files): {Fore.RED}{model_type_res['accuracy-fake']*100:.2f}{Fore.RESET}")
                print(f"{Fore.MAGENTA}Accuracy: {model_type_res['accuracy']*100:.2f}")
                print("-------")
                print()

def get_file_res_of(filep, model_res):
    def clean_basename(filep):
        filep = os.path.basename(filep)
        # remove anything thats not a letter or number
        filep = ''.join(e for e in filep if e.isalnum()).replace("sep","").replace("mp3","")
        return filep
    filep = clean_basename(filep)
    for filep2, cert, label, corr_label in model_res['raw']:
        filep2 = clean_basename(filep2)
        if filep2 == filep:
            return cert, label
    return None


if __name__ == "__main__":
    only_model = None
    if len(sys.argv) > 1:
        only_model = sys.argv[1]

    results = get_results()
    all_results(model_kinds=[only_model])

    whisper_og = "/Users/christiankilduff/Deepfake_Detection_Resources/To Test/whisper/short_iso_ckpt.pth"
    whisper_og = results[whisper_og]['og']
    whisper_iso = "/Users/christiankilduff/Deepfake_Detection_Resources/To Test/whisper/isog_ckpt.pth"
    whisper_iso = results[whisper_iso]['iso']

    # rawgat_og = "/Users/christiankilduff/Deepfake_Detection_Resources/SoundScape/Trained Models/rawgat/iso/epoch_6.pth"
    rawgat_og = "/Users/christiankilduff/Deepfake_Detection_Resources/SoundScape/Trained Models/rawgat/noniso/epoch_342.pth"
    rawgat_og = results[rawgat_og]['og']
    # rawgat_iso = "/Users/christiankilduff/Deepfake_Detection_Resources/SoundScape/Trained Models/rawgat/noniso/epoch_351.pth"
    rawgat_iso = "/Users/christiankilduff/Deepfake_Detection_Resources/SoundScape/Trained Models/rawgat/iso/epoch_0.pth"
    rawgat_iso = results[rawgat_iso]['iso']

    # cert < .01 = real
    xlsr_og = "/Users/christiankilduff/Deepfake_Detection_Resources/SoundScape/Trained Models/xlsr/noniso/og_61_epoch_147.pth"
    xlsr_og = results[xlsr_og]['og']
    xlsr_iso = "/Users/christiankilduff/Deepfake_Detection_Resources/SoundScape/Trained Models/xlsr/noniso/og_61_epoch_147.pth"
    xlsr_iso = results[xlsr_iso]['iso']

    # if cert > 1.8 = real
    clad_og = "/Users/christiankilduff/Deepfake_Detection_Resources/SoundScape/Trained Models/clad/pretrained_models/CLAD_150_10_2310.pth.tar"
    clad_og = results[clad_og]['og']
    clad_iso = "/Users/christiankilduff/Deepfake_Detection_Resources/SoundScape/Trained Models/clad/pretrained_models/CLAD_150_10_2310.pth.tar"
    clad_iso = results[clad_iso]['iso']

    vocoder_og = "/Users/christiankilduff/Deepfake_Detection_Resources/SoundScape/Trained Models/vocoder/noniso/64.29(bin99.33)epoch_73.pth"
    vocoder_og = results[vocoder_og]['og']
    vocoder_iso = "/Users/christiankilduff/Deepfake_Detection_Resources/SoundScape/Trained Models/vocoder/iso/epoch_10.pth"
    vocoder_iso = results[vocoder_iso]['iso']

    
    combined_results = {
        "correct-real": 0,
        "correct-fake": 0,
        "total-real": 0,
        "total-fake": 0,
        "accuracy-real": 0,
        "accuracy-fake": 0,
        "accuracy": 0,
    }

       
    min_real = 7
    def adjust_label_real(model, cert, label):
        """
        We make the bar for a model to classify as fake higher, this means that for a model to classify fake
        we are more confident it is correct.

        We want to reduce the number of false positives, so we make models lean towards real and only classify fake
        with high confidence.
        """
        # whisper: lower certainty values do not tell us much, if its a very high or low cert, then we assume real
        if "whisper" in model.lower():
            # return label == "Real"
            # return label == "Real"
            if cert < 0.5:
                return True
            elif label == "Real":
                return True

        # THIS CSV WAS CREATED BEFORE WE HALVED THE CERT VALUES OUTPUTTED BY CLAD, raw clad scores are in 100s, so we need to adjust to a 0-1 scale
        # 1.8 IS CHANGED TO .45 (1.8/4) IN THE FINAL CODE TO ADJUST FOR BETTER READBALE CERT
        # CLAD: cert tends to be based on analysis > .45= real < .45 = fake
        # (CLAD cert is where we calc real fake, and it operates differently, high is real, low is fake)
        elif "clad" in model.lower():
            if cert > 1.8:
                return True
        # xlsr: if very low certainty, then we assume real (we also adjust cert to be on a better scale for xlsr: * 40)
        elif "xlsr" in model.lower():
            if cert < .01 or label == "Real":
                return True
        # rawgat: if very low certainty, then we assume real
        elif "rawgat" in model.lower():
            if cert < .2 or label == "Real":
                return True
        # leave vocoder, it does not have strong outliers, solid as is
        elif "vocoder" in model.lower():
            if label == "Real":
                return True
        return False # otherwise we think its Fake

    for filep, whisper_og_cert, whisper_og_label, correct_label in whisper_og['raw']:
        try:
            models = [whisper_og, whisper_iso, rawgat_og, rawgat_iso, xlsr_og, xlsr_iso, vocoder_og, vocoder_iso, clad_og, clad_iso]
            results = []
            for model in models:
                cert, label = get_file_res_of(filep, model)
                results.append([model['model_path'], cert, label])
        except:
            continue
        
        votes_real = 0
        for result in results:
            model, cert, label = result
            if adjust_label_real(model, cert, label):
                votes_real += 1
        
        guessed_label = "Fake"
        if votes_real > min_real:
            guessed_label = "Real"

        # check correct
        if correct_label == "Real":
            combined_results['total-real'] += 1
            if guessed_label == "Real":
                combined_results['correct-real'] += 1
        elif correct_label == "Fake":
            combined_results['total-fake'] += 1
            if guessed_label == "Fake":
                combined_results['correct-fake'] += 1
    if combined_results['total-real'] > 0:
        combined_results['accuracy-real'] = combined_results['correct-real'] / combined_results['total-real']
    if combined_results['total-fake'] > 0:
        combined_results['accuracy-fake'] = combined_results['correct-fake'] / combined_results['total-fake']
    combined_results['accuracy'] = (combined_results['accuracy-real'] + combined_results['accuracy-fake']) / 2
    
    
    print(f"{Fore.BLUE}COMBINED RESULTS")
    print(f"Accuracy Real({combined_results['total-real']} files): {Fore.GREEN}{combined_results['accuracy-real']*100:.2f}{Fore.RESET} | Accuracy Fake({combined_results['total-fake']} files): {Fore.RED}{combined_results['accuracy-fake']*100:.2f}{Fore.RESET}")
    print(f"{Fore.MAGENTA}Accuracy: {combined_results['accuracy']*100:.2f} ({combined_results['accuracy-real']*100:.2f} + {combined_results['accuracy-fake']*100:.2f})/2")