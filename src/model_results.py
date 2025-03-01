from sound_scape.backend.Models import xlsr, whisper_specrnet, rawgat, vocoder
from tqdm import tqdm
from Base_Path import get_path_relative_base
csv_file = "model_results.csv"
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
def mass_eval(xlsr_path=None, vocoder_path=None):

    device = "mps"
    xlsr = xlsr(device=device)
    whisper_specrnet = whisper_specrnet(device=device)
    rawgat = rawgat(device=device)
    vocoder = vocoder(device=device)
    models = [whisper_specrnet, rawgat, xlsr, vocoder]
    for model in models:
        append_model_results(model, model.name)


import os
import csv
def append_model_results(model, model_name):
       # collect eval files from dataset
    eval_path = os.path.join(soundscape_dataset_path, "eval")
    fake_path = os.path.join(eval_path, "fake")
    real_path = os.path.join(eval_path, "bonafide")

    fake_files = [os.path.join(fake_path, f) for f in os.listdir(fake_path) if f.endswith(".mp3")]
    real_files = [os.path.join(real_path, f) for f in os.listdir(real_path) if f.endswith(".mp3")]


    # create csv file
    csv_file = "model_results.csv"
    header = ["model", "file", "correct-label", "pred-label", "certainty"]

    with open(csv_file, mode='a') as f:
        writer = csv.writer(f)
        if os.stat(csv_file).st_size == 0:
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
        return self.get_correct_real() / self.total_defined_real()


    def get_accuracy_fake(self):
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
        return 1 - (self.get_correct_real() + self.get_correct_fake()) / self.get_total()

    
    def print(self):
        print(f"Correct: {self.get_accuracy_real()*100:.2f}% of real, {self.get_accuracy_fake()*100:.2f}% of fake")
        print(f"False df: {self.get_num_incorrectly_classified_as_fake()} False rl: {self.get_num_incorrectly_classified_as_real()}")
        print(f"EER: {self.get_eer()*100:.2f}%")

def algo_result():
    index_model = 0
    index_real_label = 2
    index_label = 3
    index_cert = 4

    print()
    model_names = ["whisper_specrnet", "rawgat", "xlsr-finetuned", "vocoder"]
    results : List[ModelResultTracker] = [ModelResultTracker() for _ in range(len(model_names))]
    for i, name in enumerate(model_names):
        with open (csv_file, mode='r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if row[index_model] == name:
                    results[i].add(row[index_label], float(row[index_cert]), row[index_real_label])
        print(f"Results for {name}: ")
        res = results[i]
        res.print()

        print("---")


        print()

    wrong_csv = "res_wrong.csv"
    correct_csv = "res_correct.csv"
    with open(wrong_csv, mode='w') as f:
        with open(correct_csv, mode='w') as f2:
            writer1 = csv.writer(f)
            writer1.writerow(header)

            writer2 = csv.writer(f2)
            writer2.writerow(header)

            for i, name in enumerate(model_names):
                with open(csv_file, mode='r') as f:
                    reader = csv.reader(f)
                    next(reader)
                    for row in reader:
                        if row[index_model] == name and row[index_label] != row[index_real_label]:
                            writer1.writerow(row)
                        elif row[index_model] == name and row[index_label] == row[index_real_label]:
                            writer2.writerow(row)

    whisper_results : List[result_item] = results[0].results
    rawgat_results :  List[result_item]  = results[1].results
    xlsr_results:  List[result_item] = results[2].results
    vocoder_results:  List[result_item] = results[3].results

    print()

    outside_whisper_range_csv = "outside_whisper_range.csv"
    with open(outside_whisper_range_csv, mode='w') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'correct-label', 'pred-label', 'certainty'])
        for i in range(len(whisper_results)):
            if not (whisper_results[i].certainty < .975 and whisper_results[i].certainty > .2):
                for j, name in enumerate(model_names):
                    cert = results[j].results[i].certainty
                    if name == "xlsr-finetuned":
                        cert *= 1000
                    writer.writerow([name, results[j].results[i].correct_label, results[j].results[i].predicted_label, cert])

    combined_results = ModelResultTracker()
    outlier_results = []
    for i in range(results[0].get_total()):
        rl_c = 0

        if whisper_results[i].certainty < .975 and whisper_results[i].certainty > .2:
            combined_results.add(whisper_results[i].predicted_label, 1, whisper_results[i].correct_label)
            continue
        
        if whisper_results[i].predictsIsReal() == rawgat_results[i].predictsIsReal():
            combined_results.add(whisper_results[i].predicted_label, 1, whisper_results[i].correct_label)
            continue
        # if not vocoder_results[i].predictsIsReal():
        #     combined_results.add("Fake", 1, whisper_results[i].correct_label)



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

        outlier_results.append([whisper_results[i], rawgat_results[i], xlsr_results[i], vocoder_results[i]])

    print("Combined results: ")
    combined_results.print()

    outliers_csv = "outliers.csv"

    with open(outliers_csv, mode='w') as f:
        writer = csv.writer(f)
        writer.writerow(['model', 'correct-label', 'pred-label', 'certainty'])
        for i, outlier in enumerate(outlier_results):
            for j, name in enumerate(model_names):
                cert = outlier[j].certainty
                if name == "xlsr-finetuned":
                    cert *= 1000
                writer.writerow([name, outlier[j].correct_label, outlier[j].predicted_label, cert])

                







    
        
    

    

           
        

if __name__ == "__main__":
    mass_eval()
    # append_model_results(vocoder(device="mps",model_path="/Users/christiankilduff/Downloads/epoch_7.pth"), "vocoder-finetuned")
    # try:
    #     interpret()
    # except:
    #     pass
    # if not better try:
    
    # append_model_results(xlsr(device="mps",model_path=get_path_relative_base("pretrained_models/XLS-R/epoch_1_58.pth")), "xlsr-finetuned")
    # interpret()
    interpret()
    algo_result()

    