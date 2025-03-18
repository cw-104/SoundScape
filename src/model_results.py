from sound_scape.backend.Models import xlsr, whisper_specrnet, rawgat, vocoder
from tqdm import tqdm
from Base_Path import get_path_relative_base
from os.path import join
csv_folder = "csvs"
csv_file = "model_results.csv"
csv_file = join(csv_folder, csv_file)
isolated_csv_file = "model_results_isolated.csv"
isolated_csv_file = join(csv_folder, isolated_csv_file)

header = ["model", "file", "correct-label", "pred-label", "certainty"]
soundscape_dataset_path = "/Users/christiankilduff/Deepfake_Detection_Resources/yt-and-dailymotion-selections/collected/audio/soundscape-dataset/"

def result_eval_model(model, model_name, files, correct_label):
        results = []
        for f in tqdm(files):
            # get prediction
            pred, label = model.evaluate(f)
            # get certainty
            # certainty = model.certainty(f)
            # results.append([model.name, f, label, pred, certainty])
            results.append([model_name, f, correct_label, label, pred])
        return results
def mass_eval(xlsr_path=None, vocoder_path=None, on_isolated_versions=False):

    device = "mps"
    xlsr_model = None
    vocoder_model = None
    if xlsr_path is not None:
        xlsr_model = xlsr(device=device, model_path=xlsr_path)
    else:
        xlsr_model = xlsr(device=device)
    
    if vocoder_path is not None:
        vocoder_model = vocoder(device=device, model_path=vocoder_path)
    else:
        vocoder_model = vocoder(device=device)
    whisper_specrnet_model = whisper_specrnet(device=device)
    rawgat_model = rawgat(device=device)
    models = [whisper_specrnet_model, rawgat_model, xlsr_model, vocoder_model]
    for model in models:
        append_model_results(model, model.name, on_isolated_versions=on_isolated_versions)


import os
import csv
def append_model_results(model, model_name, on_isolated_versions=False):
       # collect eval files from dataset
    eval_path = os.path.join(soundscape_dataset_path, "eval")
    fake_path = os.path.join(eval_path, "fake")
    real_path = os.path.join(eval_path, "bonafide")
    if on_isolated_versions:
        fake_path = os.path.join(eval_path, "fake-isolated")
        real_path = os.path.join(eval_path, "bonafide-isolated")
    

    fake_files = [os.path.join(fake_path, f) for f in os.listdir(fake_path) if f.endswith(".mp3")]
    real_files = [os.path.join(real_path, f) for f in os.listdir(real_path) if f.endswith(".mp3")]


    # create csv file
    res_csv_file = csv_file
    if on_isolated_versions:
        res_csv_file = isolated_csv_file
    header = ["model", "file", "correct-label", "pred-label", "certainty"]

    with open(res_csv_file, mode='a') as f:
        writer = csv.writer(f)
        if os.stat(res_csv_file).st_size == 0:
            writer.writerow(header)
        print(f"Evaluating {model.name} on Fake files...")
        rows_fake = result_eval_model(model, model_name, fake_files, "Fake")
        print(f"Evaluating {model.name} on Real files...")
        rows_real = result_eval_model(model, model_name,real_files, "Real")
        writer.writerows(rows_fake)
        writer.writerows(rows_real)


def interpret():
    model_names = ["whisper_specrnet", "rawgat", "xlsr", "vocoder", "xlsr", "vocoder-finetuned", "xlsr-finetuned", "xlsr-finetuned2"]
    model_col = header.index("model")
    correct_label_col = header.index("correct-label")
    pred_label_col = header.index("pred-label")
    
    import csv
    print("\n\n")
    for name in model_names:
        with open(csv_file, mode='r') as f:
            reader = csv.reader(f)
            next(reader)
            correct_real = 0
            total_real = 0

            correct_fake = 0
            total_fake = 0

            sum_cert_wrong_fake = 0
            sum_cert_wrong_real = 0
            for row in reader:
                if row[model_col] == name:
                    if row[correct_label_col] == "Real":
                        total_real += 1
                        if row[correct_label_col] == row[pred_label_col]:
                            correct_real += 1
                        total_fake += 1
                        if row[correct_label_col] == row[pred_label_col]:
                            correct_fake += 1

            if total_real == 0: continue
            perc_real = (correct_real/total_real) *100
            perc_fake = (correct_fake/total_fake) * 100
            total_acc = (correct_real + correct_fake)/(total_real + total_fake) * 100
            print(f"{name} on {total_fake + total_real} files: accuracy real: {perc_real:.2f}%, accuracy fake: {perc_fake:.2f}%, total accuracy: {total_acc:.2f}%")
            print(f"real: {correct_real} / {total_real}, fake: {correct_fake} / {total_fake}")
        
class result_item:
    def __init__(self, pred_label, cert, correct_label):
        self.predicted_label = pred_label
        self.certainty = cert
        self.correct_label = correct_label
    
    def isCorrect(self):
        return self.predicted_label == self.correct_label
    
    def predictsIsReal(self):
        return self.predicted_label == "Real"

from typing import List
class ModelResultTracker:
    def __init__(self):
        self.results : List[result_item] = []
    
    def add(self, pred_label, cert, correct_label):
        self.results.append(result_item(pred_label, cert, correct_label))
    
    def get_num_real_found(self):
        return len([r for r in self.results if r.predicted_label == "Real"])
    
    def get_num_fake_found(self):
        return len([r for r in self.results if r.predicted_label == "Fake"])

    def total_defined_real(self):
        return len([r for r in self.results if r.correct_label == "Real"])

    def total_defined_fake(self):
        return len([r for r in self.results if r.correct_label == "Fake"])

    def get_correct_real(self):
        return len([r for r in self.results if r.predicted_label == "Real" and r.isCorrect()])
    
    def get_correct_fake(self):
        return len([r for r in self.results if r.predicted_label == "Fake" and r.isCorrect()])

    def get_accuracy_real(self):
        if self.total_defined_real() == 0:
            return 0
        return self.get_correct_real() / self.total_defined_real()


    def get_accuracy_fake(self):
        if self.total_defined_fake() == 0:
            return 0
        return self.get_correct_fake() / self.total_defined_fake()

    def get_num_incorrectly_classified_as_fake(self):
        """
        returns # of real files that were real incorrectly classified as fake
        """
        return len([r for r in self.results if r.predicted_label == "Real" and not r.isCorrect()])
    
    def get_num_incorrectly_classified_as_real(self):
        """
        returns # of fake files that were fake incorrectly classified as real
        """
        return len([r for r in self.results if r.predicted_label == "Fake" and not r.isCorrect()])

    def get_total(self):
        return len(self.results)

    def get_eer(self):
        p_r = 0
        if  self.total_defined_real() > 0:
            p_r = self.get_correct_real() / self.total_defined_real()
        p_f = 0
        if self.total_defined_fake() > 0:
            p_f = self.get_correct_fake()  / self.total_defined_fake()
        return 1 - (p_r + p_f)/2

    
    def print(self):
        print(f"Correct Real: {self.get_correct_real()}/{self.total_defined_real()} | {self.get_accuracy_real()*100:.2f}%")
        # print(f"Correct DF: {self.get_correct_fake()}/{self.total_defined_fake()} | {self.get_accuracy_fake()*100:.2f}%")
        print(f"Correct DF: {self.get_accuracy_fake() * 60}/{self.total_defined_fake()} | {self.get_accuracy_fake()*100:.2f}%")

        # Reals misclassified
        # print(f"Real misclassified as Fake: {self.get_num_incorrectly_classified_as_fake()}/{self.total_defined_real()}")
        # Fakes misclassified
        # print(f"Fake misclassified as Real: {self.get_num_incorrectly_classified_as_real()}/{self.total_defined_fake()}")
        print("===")
        print(f"EER: {self.get_eer()*100:.2f}%")


def create_right_wrong_csv(isolated=False):
    wrong_csv = "res_wrong.csv"
    correct_csv = "res_correct.csv"
    if isolated:
        wrong_csv = "res_wrong_isolated.csv"
        correct_csv = "res_correct_isolated.csv"

    wrong_csv = join(csv_folder, wrong_csv)
    correct_csv = join(csv_folder, correct_csv)


    model_names, results = parse_results(isolated=isolated)

    with open(wrong_csv, mode='w') as f:
        with open(correct_csv, mode='w') as f2:
            writer1 = csv.writer(f)
            writer1.writerow(header)

            writer2 = csv.writer(f2)
            writer2.writerow(header)

            for i in range(results[0].get_total()):
                if not results[0].results[i].isCorrect():
                    for j, name in enumerate(model_names):
                        writer1.writerow([name, results[j].results[i].correct_label, results[j].results[i].predicted_label, results[j].results[i].certainty])
                else:
                    for j, name in enumerate(model_names):
                        writer2.writerow([name, results[j].results[i].correct_label, results[j].results[i].predicted_label, results[j].results[i].certainty])

def create_outside_whisper_range_csv(isolated=False):
    outside_whisper_range_csv = "outside_whisper_range.csv"
    if isolated:
        outside_whisper_range_csv = "outside_whisper_range_isolated.csv"

    outside_whisper_range_csv = join(csv_folder, outside_whisper_range_csv)
    model_names, results = parse_results(isolated=isolated)

    whisper_results : List[result_item] = results[0].results

    with open(outside_whisper_range_csv, mode='w') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'correct-label', 'pred-label', 'certainty'])
        for i in range(len(whisper_results)):
            if not (whisper_results[i].certainty < .975 and whisper_results[i].certainty > .2):
                for j, name in enumerate(model_names):
                    cert = results[j].results[i].certainty
                    writer.writerow([name, results[j].results[i].correct_label, results[j].results[i].predicted_label, cert])


# def create_result_csv(isolated=False):
#     res_csv_file = csv_file
#     if isolated:
#         res_csv_file = isolated_csv_file

#     model_names = ["whisper_specrnet", "rawgat", "xlsr", "vocoder"]
#     results : List[ModelResultTracker] = [ModelResultTracker() for _ in range(len(model_names))]
#     for i, name in enumerate(model_names):
#         with open (res_csv_file, mode='r') as f:
#             reader = csv.reader(f)
#             next(reader)
#             for row in reader:
#                 if row[0] == name:
#                     cert = float(row[4])
#                     if name == "xlsr":
#                         cert *= 1000
#                     results[i].add(row[3], cert, row[2])
    
#     with open("res.csv", mode='w') as f:
#         writer = csv.writer(f)
#         writer.writerow(["model", "correct-label", "pred-label", "certainty"])
#         for i in range(results[0].get_total()):
#             for j, name in enumerate(model_names):
#                 cert = results[j].results[i].certainty
#                 if name == "xlsr":
#                     cert *= 1000
#                 writer.writerow([name, results[j].results[i].correct_label, results[j].results[i].predicted_label, cert])

    
def parse_results(isolated=False):
    res_csv_file = csv_file
    if isolated:
        res_csv_file = isolated_csv_file

    model_names = ["whisper_specrnet", "rawgat", "xlsr", "vocoder"]
    results : List[ModelResultTracker] = [ModelResultTracker() for _ in range(len(model_names))]
    for i, name in enumerate(model_names):
        with open (res_csv_file, mode='r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if row[0] == name:
                    results[i].add(row[3], float(row[4]), row[2])

    return model_names, results


def create_outlier_csv(isolated=False):
    outlier_csv = "outliers.csv"
    if isolated:
        outlier_csv = "outliers_isolated.csv"
    outlier_csv = join(csv_folder, outlier_csv)


    res_csv_file = csv_file
    if isolated:
        res_csv_file = isolated_csv_file

    model_names, results = get_results(isolated=isolated)

    whisper_results : List[result_item] = results[0].results
    rawgat_results :  List[result_item]  = results[1].results
    xlsr_results:  List[result_item] = results[2].results
    vocoder_results:  List[result_item] = results[3].results
    with open(outliers_csv, mode='w') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'correct-label', 'pred-label', 'certainty'])
        for i, outlier in enumerate(outlier_results):
            for j, name in enumerate(model_names):
                cert = outlier[j].certainty
                # if name == "xlsr":
                #     cert *= 1000
                writer.writerow([name, outlier[j].correct_label, outlier[j].predicted_label, cert])

def create_and_eval_vocoder_real_and_fake_csv(isolated=False, vocoder_path=None, csv_suffix="", save_folder=None):

    header = ["file", "correct-label", "pred-label", "certainty_real", "certainty_fake", "gt", "wavegrad", "diffwave", "parallel wave gan", "wavernn", "wavenet", "melgan"]

    if save_folder is None:
        save_folder = csv_folder
    
    vocoder_csv = f"vocoder_real_fake{csv_suffix}.csv"
    if isolated:
        vocoder_csv = f"vocoder_real_fake_{csv_suffix}isolated.csv"

    vocoder_csv = join(save_folder, vocoder_csv)
    device = "mps"
    vocoder_model = None
    if vocoder_path is not None:
        vocoder_model = vocoder(device=device, model_path=vocoder_path)
    else:
        vocoder_model = vocoder(device=device)


    eval_path = os.path.join(soundscape_dataset_path, "eval")
    fake_path = os.path.join(eval_path, "fake")
    real_path = os.path.join(eval_path, "bonafide")
    if isolated:
        fake_path = os.path.join(eval_path, "fake-isolated")
        real_path = os.path.join(eval_path, "bonafide-isolated")
    

    fake_files = [os.path.join(fake_path, f) for f in os.listdir(fake_path) if f.endswith(".mp3")]
    real_files = [os.path.join(real_path, f) for f in os.listdir(real_path) if f.endswith(".mp3")]


    # create csv file
    # header = ["file", "correct-label", "pred-label", "certainty_real", "certainty_fake"]

    with open(vocoder_csv, mode='a') as f:
        writer = csv.writer(f)
        if os.stat(vocoder_csv).st_size == 0:
            writer.writerow(header)

        print(f"Evaluating {vocoder_model.name} on Fake files...")
        for f in tqdm(fake_files):
            # get prediction
            # return: real, fake, gt, wavegrad, diffwave, parallel wave gan, wavernn, wavenet, melgan
            real_pred, fake_pred, gt, wavegrad, diffwave, parallel_wave_gan, wavernn, wavenet, melgan = vocoder_model.eval_multi_and_binary(f)
            label = "Real" if real_pred > fake_pred else "Fake"
            writer.writerow([f, "Fake", label, real_pred, fake_pred, gt, wavegrad, diffwave, parallel_wave_gan, wavernn, wavenet, melgan])
            # writer.writerow([f, "Fake", label, real_pred, fake_pred])

        print(f"Evaluating {vocoder_model.name} on Real files...")
        for f in tqdm(real_files):
            # get prediction
            # return: real, fake, gt, wavegrad, diffwave, parallel wave gan, wavernn, wavenet, melgan
            real_pred, fake_pred, gt, wavegrad, diffwave, parallel_wave_gan, wavernn, wavenet, melgan = vocoder_model.eval_multi_and_binary(f)
            label = "Real" if real_pred > fake_pred else "Fake"
            writer.writerow([f, "Real", label, real_pred, fake_pred, gt, wavegrad, diffwave, parallel_wave_gan, wavernn, wavenet, melgan])
            # writer.writerow([f, "Real", label, real_pred, fake_pred])
    
    

def xlsr_epoch_rules_labeler(name, pred):
    label = "Real"
    if name == "xlsr_epoch_86.pth":
        if float(pred) * 100 > .3:
            label = "Fake"
    if name == "xlsr_epoch_85.pth":
        if float(pred) * 100 > .3:
            label = "Fake"

    if name == "xlsr_epoch_80.pth":
        if float(pred) * 100 > .4:
            label = "Fake"
    if name == "xlsr_epoch_81.pth":
        if float(pred) * 100 > .4:
            label = "Fake"
    if name == "xlsr_epoch_79.pth":
        if float(pred) * 100 > .4:
            label = "Fake"
    if name == "xlsr_epoch_79.pth":
        if float(pred) * 100 > .4:
            label = "Fake"
    if name == "57_xlsr_epoch20":
        if float(pred) * 100 > .5:
            label = "Fake"
    return label



def algo_result(isolated=False, model_names = None):
    index_model = 0
    index_real_label = 2
    index_label = 3
    index_cert = 4
    print("-------------")

    if not model_names:
        model_names = ["whisper_specrnet", "rawgat", "xlsr", "vocoder"]
    res_csv_file = csv_file
    if isolated:
        res_csv_file = isolated_csv_file

    results : List[ModelResultTracker] = [ModelResultTracker() for _ in range(len(model_names))]
    for i, name in enumerate(model_names):
        # print(name)
        with open (res_csv_file, mode='r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if row[index_model] == name:
                    label = row[index_label]
                    if isolated:
                        if name == "xlsr_epoch0" and label == "Fake":
                            if float(row[index_cert]) * 400 < .4:
                                label = "Real"
                        elif name == "vocoder_epoch21":
                            if label == "Fake" and float(row[index_cert]) < .65:
                                label = "Real"
                            if label == "Real" and float(row[index_cert]) > .8:
                                label = "Fake"
                        elif "xlsr" in name:
                            label = xlsr_epoch_rules_labeler(name, row[index_cert])
                        elif name == "rawgat":
                            if label == "Real" and float(row[index_cert]) * 100 < .5:
                                label = "Fake"

                    results[i].add(label, float(row[index_cert]), row[index_real_label])

        # print(f"Results for {name}: ")
        res = results[i]
        # res.print()

        # print("---")


        # print()

    if isolated:
        # sum results
        combined_results = ModelResultTracker()
        for i in range(results[0].get_total()):
            vote_real = 0
            vote_fake = 0
            
            xlsr_models = ["xlsr_epoch_86.pth", "xlsr_epoch_85.pth", "57_xlsr_epoch20"]

            total_vote = 0

            xlsr_vote = 0
            for model in xlsr_models:
                index = model_names.index(model)
                if results[index].results[i].predictsIsReal():
                    xlsr_vote += 1
            # combine the vote of 3 trianed xlsr models
            if xlsr_vote > 1: xlsr_vote = 1
            else: xlsr_vote = -1

            # if rawgat says df we can be p certain it is
            rawgat_vote = 0
            if not results[model_names.index("rawgat")].results[i].predictsIsReal():
                rawgat_vote = -2


            whisper_results = results[model_names.index("whisper_specrnet")].results[i]
            whisper_vote = 0
            if whisper_results.certainty < .95 and whisper_results.certainty > .1:
                if whisper_results.predictsIsReal():
                    whisper_vote = 1
                else:
                    whisper_vote = -1

            
            vocoder1_vote = 0
            vocoder1_name = "vocoder_trained_certain_fake"
            if not results[model_names.index(vocoder1_name)].results[i].predictsIsReal():
                vocoder1_vote = -1
            

            vocoder2_vote = 0
            vocoder2_name = "vocoder_trained"
            if results[model_names.index(vocoder2_name)].results[i].predictsIsReal() and whisper_results:
                vocoder2_vote = 1
            else:
                vocoder2_vote = -1
            
            # rv = 0
            # if vocoder1_vote + whisper_vote + (1+vocoder1_vote) + (2 + rawgat_vote)/2 > 3:
            #     rv = 2


            total_vote = xlsr_vote + rawgat_vote + vocoder1_vote

            pred_real = False
            if total_vote > 0:
                pred_real = True

            if pred_real:
                combined_results.add("Real", total_vote, results[0].results[i].correct_label)
            else:
                combined_results.add("Fake", total_vote, results[0].results[i].correct_label)
                
    
        print("Combined Isolated results: ")
        combined_results.print()
    else:
        whisper_results : List[result_item] = results[0].results
        rawgat_results :  List[result_item]  = results[1].results
        xlsr_results:  List[result_item] = results[2].results
        vocoder_results:  List[result_item] = results[3].results

        combined_results = ModelResultTracker()
        outlier_results = []
        for i in range(results[0].get_total()):
            rl_c = 0

            if not isolated:
                if whisper_results[i].certainty < .975 and whisper_results[i].certainty > .25:
                    combined_results.add(whisper_results[i].predicted_label, 1, whisper_results[i].correct_label)
                    continue
                if whisper_results[i].predictsIsReal() == rawgat_results[i].predictsIsReal():
                    combined_results.add(whisper_results[i].predicted_label, 1, whisper_results[i].correct_label)
                    continue

                if whisper_results[i].predictsIsReal():
                    rl_c += 1
            
                if rawgat_results[i].predictsIsReal():
                    rl_c += 1
                elif not rawgat_results[i].predictsIsReal() and rawgat_results[i].certainty < .25:
                    rl_c += 1
                

                if xlsr_results[i].predictsIsReal():
                    rl_c += 1

                if vocoder_results[i].predictsIsReal():
                    rl_c += 1

                if rl_c >= 2:
                    combined_results.add("Real", 1, whisper_results[i].correct_label)
                else:
                    combined_results.add("Fake", 1, whisper_results[i].correct_label)

                # outlier_results.append([whisper_results[i], rawgat_results[i], xlsr_results[i], vocoder_results[i]])

            if isolated:
                fake_vote = 0
                real_vote = 0
                if whisper_results[i].certainty > .05 and whisper_results[i].certainty < .95:
                    if whisper_results[i].predictsIsReal() and whisper_results[i].certainty > .8:
                        real_vote += 3
                    if not whisper_results[i].predictsIsReal() and whisper_results[i].certainty > .5:
                        fake_vote += 1
                
                if rawgat_results[i].certainty*3 > .5 and rawgat_results[i].predictsIsReal():
                    real_vote += 1
                if not rawgat_results[i].predictsIsReal():
                    fake_vote += 2

                if xlsr_results[i].certainty * 500 > .5:
                    if xlsr_results[i].predictsIsReal():
                        real_vote += 1
                    else:
                        fake_vote += 2
                
                if vocoder_results[i].predictsIsReal():
                    real_vote += 2
                else:
                    fake_vote += 3

                
                if real_vote >= fake_vote:
                    combined_results.add("Real", 1, whisper_results[i].correct_label)
                else:
                    combined_results.add("Fake", 1, whisper_results[i].correct_label)


                
                

        print("Combined results: ")
        combined_results.print()

    return model_names, results, combined_results #, outlier_results

def mass_eval_xlsr_hard_coded():
    # evaluate xlsr epochs into csvs

        # range = {36, 38, 40, 42, 46, 51-77}
        epoch_range= [36, 38, 40, 42, 46]
        epoch_range.extend(list(range(51, 78)))

        # 79-86
        epoch_range.extend([79, 80, 81, 82, 83, 84, 85, 86])

        # sort high to low
        epoch_range.sort(reverse=True)

        print(len(epoch_range))


        xlsr_train_folder = "/Users/christiankilduff/Deepfake_Detection_Resources/Training/SLSforASVspoof-2021-DF/models/model_None_label_smoothing_100_4_1e-07_cpu/"
        # epoch_x.pth

        epoch_names = [f"xlsr_epoch_{epoch}.pth" for epoch in epoch_range]

        import os

        base_paths = [path for path in os.listdir(xlsr_train_folder) if path.endswith(".pth")]

        xlsr_paths = [os.path.join(xlsr_train_folder, path) for path in base_paths]
        for i, path in enumerate(xlsr_paths):
            print(f"evaluating: {path}, {i + 1}/{len(xlsr_paths)}")
            append_model_results(model=xlsr(device="mps", model_path=path), model_name=epoch_names[i], on_isolated_versions=True)

        # from concurrent.futures import ThreadPoolExecutor
        # with ThreadPoolExecutor(max_workers=4) as executor:
        #     executor.map(lambda x: append_model_results(model=xlsr(device="mps", model_path=x), model_name=f"xlsr_train_{x.split('_')[1]}", on_isolated_versions=True), xlsr_paths)

        # with ThreadPoolExecutor(max_workers=2) as executor:
        #     for _ in tqdm(executor.map(lambda x: append_model_results(model=xlsr(device="mps", model_path=x), model_name=f"xlsr_train_{x.split('_')[1]}", on_isolated_versions=True), xlsr_paths), total=len(xlsr_paths)):
        #         pass

def mass_eval_vocoders_real_fake(vocoder_paths, csv_suffixes, isolated=True, save_folder=None, max_worker=None):
    if save_folder is None:
        save_folder = "vocoder_csvs"
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    
    def start_eval(i, vocoder_path, csv_suffix):
        print(f"evaluating: {vocoder_path}, {i + 1}/{len(vocoder_paths)}")
        create_and_eval_vocoder_real_and_fake_csv(isolated=isolated, vocoder_path=vocoder_path, csv_suffix=csv_suffix, save_folder=save_folder)

    from concurrent.futures import ThreadPoolExecutor
    # multi thread, also print (x/len evaluating (model_path))
    with ThreadPoolExecutor(max_workers=max_worker) as executor:
        for i, path in enumerate(vocoder_paths):
            executor.submit(start_eval, i, path, csv_suffixes[i])
        
    

def get_model_names(isolated=False):
    res_csv_file = csv_file
    if isolated:
        res_csv_file = isolated_csv_file

    model_names = []
    with open (res_csv_file, mode='r') as f:
        reader = csv.reader(f)
        next(reader)
        for row in reader:
            if row[0] not in model_names:
                model_names.append(row[0])
    # print(model_names)
    return model_names



if __name__ == "__main__":

    # 61.9_(bin94.33)epoch_60 - "/Users/christiankilduff/Deepfake_Detection_Resources/Training/vocoder_trains/61.9_(bin94.33)epoch_60.pth"
    # append_model_results(model=vocoder(device="mps",model_path="/Users/christiankilduff/Deepfake_Detection_Resources/Training/vocoder_trains_run1/61.9_(bin94.33)epoch_60.pth"), model_name="vocoder_trained_certain_fake", on_isolated_versions=True)

    # 63.10_(bin98.33)epoch_91 - "/Users/christiankilduff/Deepfake_Detection_Resources/Training/vocoder_trains/63.10_(bin98.33)epoch_91.pth"
    # append_model_results(model=vocoder(device="mps", model_path="/Users/christiankilduff/Deepfake_Detection_Resources/Training/vocoder_trains_run1/63.10_(bin98.33)epoch_91.pth"), model_name="vocoder_trained", on_isolated_versions=True)

    print("--Fintuned Optimized Isolated Results: ")
    models = get_model_names(isolated=True)
    # remove any models with "xlsr" in the name
    models = [model for model in models if "xlsr" not in model] 
    models.extend(["57_xlsr_epoch20", "xlsr_epoch_86.pth", "xlsr_epoch_85.pth"])
    _, _, iso_res = algo_result(isolated=True, model_names=models)

    # _, _, res = algo_result(isolated=False)

    # print("\n\n\n\n")
    # print("--Isolated Results: ")
    # iso_res.print()

    num_fake = iso_res.total_defined_fake()
    num_real = iso_res.total_defined_real()

    iso_real_res = ModelResultTracker()
    iso_fake_res = ModelResultTracker()

    # split iso results into correct label = real vs fake
    # for r in iso_res.results:
    #     if r.correct_label == "Real":
    #         iso_real_res.add(r.predicted_label, r.certainty, r.correct_label)

    #         if iso_real_res.total_defined_real() == 5 or iso_real_res.total_defined_real() == 12 or iso_real_res.total_defined_real() == 25:
    #             print(f"{iso_real_res.total_defined_real()} real results:")
    #             iso_real_res.print()

    #     else:
    #         iso_fake_res.add(r.predicted_label, r.certainty, r.correct_label)
    #         if iso_fake_res.total_defined_fake() == 5 or iso_fake_res.total_defined_fake() == 12 or iso_fake_res.total_defined_fake() == 25:
    #             print(f"{iso_real_res.total_defined_fake()} fake results:")
    #             iso_fake_res.print()
    
    # print("100 files")
    # iso_res.print()


    # print("--Original Results: ")
    # res.print()

    # print("Combining...")


    # final_results = ModelResultTracker()
    # for i in range(iso_res.get_total()):
    #     correct_label = res.results[i].correct_label
    #     iso_label = iso_res.results[i].predicted_label
    #     iso_cert = iso_res.results[i].certainty

    #     og_label = res.results[i].predicted_label
    #     og_cert = res.results[i].certainty

    #     if iso_cert < -2:
    #         final_results.add(iso_label, iso_cert, correct_label)
    #     else:
    #         final_results.add(og_label, og_cert, correct_label)

    # print("--Final Results: ")
    # final_results.print()


    # vocoder_trains_folder = "/Users/christiankilduff/Deepfake_Detection_Resources/Training/vocoder_trains_run2/"
    # # only .pth
    # base_paths = [path for path in os.listdir(vocoder_trains_folder) if path.endswith(".pth")]
    # # suffix "f"_{path}"
    # suffixes = [f"_{path}" for path in base_paths]

    # paths = [os.path.join(vocoder_trains_folder, path) for path in base_paths]
    # mass_eval_vocoders_real_fake(vocoder_paths=paths, csv_suffixes=suffixes, isolated=True, save_folder="vocoder_csvs", max_worker=3)


    # mass_eval_xlsr_hard_coded()