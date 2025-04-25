from colorama import Fore, init
import os, csv, sys
init(autoreset=True)

csv_all_models = "final_eval_all_models.csv"

def get_results():
    results = {}
    with open(csv_all_models, "r") as f:
        reader = csv.DictReader(f)
        # Model,File,Certainty,Label,Correct Label,Isolated
        for row in reader:
            model = row['Model']
            if model not in results:
                results[model] = {
                    'og': {
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
                        'raw': []
                        ,
                        'total-real': 0,
                        'correct-real': 0,
                        'total-fake': 0,
                        'correct-fake': 0,
                        'accuracy-real': 0,
                        'accuracy-fake': 0,
                        'accuracy': 0,
                    },
                }
            filep = row['File']
            certainty = row['Certainty']
            certainty = float(certainty)
            label = row['Label']
            correct_label = row['Correct Label']
            type = 'og'
            if row['Isolated'] == "True":
                type = 'iso'
            if correct_label == "Real":
                results[model][type]['total-real'] += 1
                if correct_label == label:
                    results[model][type]['correct-real'] += 1
            elif correct_label == "Fake":
                results[model][type]['total-fake'] += 1
                if correct_label == label:
                    results[model][type]['correct-fake'] += 1
            
            results[model][type]['raw'].append([filep, float(certainty), label, correct_label])
    for model in results.keys():
        for type in ['og', 'iso']:
            results[model][type]['accuracy-real'] = results[model][type]['correct-real'] / results[model][type]['total-real'] if results[model][type]['total-real'] > 0 else 0
            results[model][type]['accuracy-fake'] = results[model][type]['correct-fake'] / results[model][type]['total-fake'] if results[model][type]['total-fake'] > 0 else 0
            results[model][type]['accuracy'] = (results[model][type]['accuracy-real'] + results[model][type]['accuracy-fake']) / 2
    return results

def all_results():
    results = get_results()
    for kind in ['whisper', 'rawgat', 'xlsr', 'clad', 'vocoder']:
        # if kind not in ['clad']:
            # continue
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
    results = get_results()
    all_results()

    og_whisper = "/Users/christiankilduff/Deepfake_Detection_Resources/To Test/whisper/short_iso_ckpt.pth"
    og_whisper = results[og_whisper]['og']
    iso_whisper = "/Users/christiankilduff/Deepfake_Detection_Resources/To Test/whisper/isog_ckpt.pth"
    iso_whisper = results[iso_whisper]['iso']

    rawgat_og = "/Users/christiankilduff/Deepfake_Detection_Resources/SoundScape/Trained Models/rawgat/iso/epoch_6.pth"
    rawgat_og = results[rawgat_og]['og']
    rawgat_iso = "/Users/christiankilduff/Deepfake_Detection_Resources/SoundScape/Trained Models/rawgat/noniso/epoch_351.pth"
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
    votes_real = 0
    for filep, whisper_og_cert, whisper_og_label, correct_label in og_whisper['raw']:
        try:
            whisper_iso_cert, whisper_iso_label = get_file_res_of(filep, iso_whisper)
            rawgat_iso_cert, rawgat_iso_label = get_file_res_of(filep, rawgat_iso)
            rawgat_og_cert, rawgat_og_label = get_file_res_of(filep, rawgat_og)
            xlsr_og_cert, xlsr_og_label = get_file_res_of(filep, xlsr_og)
            xlsr_iso_cert, xlsr_iso_label = get_file_res_of(filep, xlsr_iso)
            vocoder_iso_cert, vocoder_iso_label = get_file_res_of(filep, vocoder_iso)
            vocoder_og_cert, vocoder_og_label = get_file_res_of(filep, vocoder_og)
            clad_og_cert, clad_og_label = get_file_res_of(filep, clad_og)
            clad_iso_cert, clad_iso_label = get_file_res_of(filep, clad_iso)
        except:
            continue
        
        votes_real = 0
        # whisper
        for cert, label in [(whisper_iso_cert, whisper_iso_label), (whisper_og_cert, whisper_og_label)]:
            if cert < 0.99 and cert > .01 and label == "Fake":
                votes_real += 1
            elif label == "Real":
                votes_real += 1

        # CLAD
        for cert, label in [(clad_iso_cert, clad_iso_label), (clad_og_cert, clad_og_label)]:
            if cert > 1.8 and label == "Real":
                votes_real += 1
        
        # xlsr
        for cert, label in [(xlsr_iso_cert, xlsr_iso_label), (xlsr_og_cert, xlsr_og_label)]:
            if cert < .01:
                votes_real += 1
        
        # vocoder and rawgat
        for cert, label in [(vocoder_iso_cert, vocoder_iso_label), (vocoder_og_cert, vocoder_og_label), (rawgat_iso_cert, rawgat_iso_label), (rawgat_og_cert, rawgat_og_label)]:
            if label == "Real":
                votes_real += 1
        
        guessed_label = "Fake"
        if votes_real > 7:
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
    print(f"{Fore.MAGENTA}Accuracy: {combined_results['accuracy']*100:.2f}")