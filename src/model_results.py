from sound_scape.backend.Models import xlsr, whisper_specrnet, rawgat, vocoder
from tqdm import tqdm
csv_file = "model_results.csv"
header = ["model", "file", "correct-label", "pred-label", "certainty"]
def mass_eval():

    device = "mps"
    xlsr = xlsr(device=device)
    whisper_specrnet = whisper_specrnet(device=device)
    rawgat = rawgat(device=device)
    vocoder = vocoder(device=device)

    # collect eval files from dataset
    eval_path = "/Users/christiankilduff/Deepfake_Detection_Resources/yt-and-dailymotion-selections/collected/soundscape-dataset/eval"
    import os
    fake_path = os.path.join(eval_path, "fake")
    real_path = os.path.join(eval_path, "bonafide")

    fake_files = [os.path.join(fake_path, f) for f in os.listdir(fake_path) if f.endswith(".mp3")]
    real_files = [os.path.join(real_path, f) for f in os.listdir(real_path) if f.endswith(".mp3")]

    import csv

    # create csv file
    csv_file = "model_results.csv"
    header = ["model", "file", "correct-label", "pred-label", "certainty"]
    models = [whisper_specrnet, rawgat, xlsr, vocoder]

    def eval_model(model, files, correct_label):
        results = []
        print(f"Evaluating {model.name} on correct_label files...")
        for f in tqdm(files):
            # get prediction
            pred, label = model.evaluate(f)
            # get certainty
            # certainty = model.certainty(f)
            # results.append([model.name, f, label, pred, certainty])
            results.append([model.name, f, correct_label, label, pred])
        return results

    with open(csv_file, mode='w') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for model in models:
            rows_fake = eval_model(model, fake_files, "Fake")
            rows_real = eval_model(model, real_files, "Real")
            writer.writerows(rows_fake)
            writer.writerows(rows_real)

def interpret():
    model_names = ["whisper_specrnet", "rawgat", "xlsr", "vocoder"]
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
            for row in reader:
                if row[model_col] == name:
                    if row[correct_label_col] == "Real":
                        total_real += 1
                        if row[correct_label_col] == row[pred_label_col]:
                            correct_real += 1
                    else:
                        total_fake += 1
                        if row[correct_label_col] == row[pred_label_col]:
                            correct_fake += 1
            
            perc_real = (correct_real/total_real) *100
            perc_fake = (correct_fake/total_fake) * 100
            total_acc = (correct_real + correct_fake)/(total_real + total_fake) * 100
            print(f"{name} on {total_fake + total_real} files: accuracy real: {perc_real:.2f}%, accuracy fake: {perc_fake:.2f}%, total accuracy: {total_acc:.2f}%")
            print(f"real: {correct_real} / {total_real}, fake: {correct_fake} / {total_fake}")
        


if __name__ == "__main__":
    # mass_eval()
    interpret()
    