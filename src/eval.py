import os
from sound_scape.backend import Models
from tqdm import tqdm

sound_scape_dataset_path = "/Users/christiankilduff/Deepfake_Detection_Resources/Training/soundscape-dataset/"

def get_eval_files(isolated=False):
    eval_folder = os.path.join(sound_scape_dataset_path, "eval")
    real = "bonafide"
    fake = "fake"
    if isolated:
        real = "bonafide-isolated"
        fake = "fake-isolated"

    # mp3s in joined eval and real/fake
    real_files = [os.path.join(eval_folder, real, f) for f in os.listdir(os.path.join(eval_folder, real)) if f.endswith(".mp3")]
    fake_files = [os.path.join(eval_folder, fake, f) for f in os.listdir(os.path.join(eval_folder, fake)) if f.endswith(".mp3")]

    return real_files, fake_files


def eval_model(model, isolated=False):
    real_files, fake_files = get_eval_files(isolated=isolated)
    
    print(f"starting eval '({'' if isolated else 'not '}isolated)'...")
    # evaluate real files
    avg_cert = 0
    num_correct = 0
    len_real = len(real_files)
    eer = 0
    for file in tqdm(real_files):
        certainty, label = model.evaluate(file)
        if label == "Real":
            num_correct += 1
        # avg_cert += (certainty - .5) / len_real

    print(f"Real Average Certainty: {avg_cert} | Correct: {num_correct}/{len_real} | Accuracy: {num_correct / len_real}")
    eer += num_correct / len_real / 2
    # evaluate fake files
    avg_real = num_correct / len_real
    avg_cert = 0
    num_correct = 0
    len_fake = len(fake_files)
    for file in tqdm(fake_files):
        certainty, label = model.evaluate(file)
        if label == "Fake":
            num_correct += 1
        # avg_cert += (certainty - .5) / len_fake
    eer += num_correct / len_fake / 2
    eer = 1 - eer
    print(f"Fake Average Certainty: {avg_cert} | Correct: {num_correct}/{len_fake} | Accuracy: {num_correct / len_fake} | EER: {eer}")
    return isolated, avg_real, num_correct / len_fake, eer

def eval_whisper(isolated=True):
    # model_path = "/Users/christiankilduff/Downloads/Training/model_trains/whisper/isog_ckpt-1.pth"
    """
    ISOLATED:
    Real Average Certainty: -0.17535322702085404 | Correct: 17/57 | Accuracy: 0.2982456140350877
    Fake Average Certainty: -0.150854110250475 | Correct: 26/40 | Accuracy: 0.65

    NOT ISOLATED:
    Real Average Certainty: -0.0473494198527078 | Correct: 29/64 | Accuracy: 0.453125
    Fake Average Certainty: -0.010948996809181854 | Correct: 23/45 | Accuracy: 0.5111111111111111
    """




    model_path = "/Users/christiankilduff/Downloads/Training/model_trains/whisper/isog_ckpt.pth"
    """
    ISOLATED:
    Real Average Certainty: -0.09484519646209438 | Correct: 22/57 | Accuracy: 0.38596491228070173
    Fake Average Certainty: -0.07466946988097332 | Correct: 23/40 | Accuracy: 0.575

    NOT ISOLATED:
    Real Average Certainty: 0.08311415845491865 | Correct: 38/64 | Accuracy: 0.59375
    Fake Average Certainty: 0.10605645489170763 | Correct: 19/45 | Accuracy: 0.4222222222222222
    """
    model = Models.whisper_specrnet(device="mps", weights_path=model_path)

    eval_model(model, isolated=isolated)

def eval_rawgat(isolated=True, model_path=None):
    if model_path is None:
        model_path = "/Users/christiankilduff/Downloads/Training/model_trains/Rawgat/isolated-runs-1/models/model_WCE_3000_12_0.0001-isolated/epoch_5.pth"
    print(f"evaluating model: {model_path}")
    model = Models.rawgat.of_path(device="mps", weights_path=model_path)

    isolated, avg_real, avg_fake, eer = eval_model(model, isolated=isolated)
    print(f"{model_path}\t{isolated}\t{avg_real}\t{avg_fake}\t{eer}")
def eval_xlsr(isolated=True, model_path=None):
    if model_path is None:
        model_path = "/Users/christiankilduff/Downloads/Training/model_trains/XLSR/isolated-runs-1/models/model_WCE_3000_12_0.0001-isolated/epoch_5.pth"
    print(f"evaluating model: {model_path}")
    model = Models.xlsr(device="mps", model_path=model_path)

    isolated, avg_real, avg_fake, eer = eval_model(model, isolated=isolated)
    print(f"{model_path}\t{isolated}\t{avg_real}\t{avg_fake}\t{eer}")

# main
if __name__ == "__main__":
    # parse args --model
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="path")
    parser.add_argument("--epoch", type=str, default=None, help="path")
    args = parser.parse_args()
    if args.model is not None:
        # eval_rawgat(model_path=args.model)
        eval_xlsr(isolated=False, model_path=args.model)
        exit()
    if args.epoch is not None:
        # eval_rawgat(model_path=f"/Users/christiankilduff/Downloads/Training/model_trains/Rawgat/isolated-runs-1/models/model_WCE_3000_12_0.0001-isolated/epoch_{args.epoch}.pth")
        eval_rawgat(model_path=f"/Users/christiankilduff/Downloads/Training/model_trains/Rawgat/og-runs1/models/model_WCE_1000_12_0.0001/epoch_{args.epoch}.pth", isolated=False)
        exit()
    # eval_rawgat(isolated=True)
    # eval_rawgat(isolated=False)
    eval_xlsr(isolated=False, model_path="/Users/christiankilduff/Downloads/Training/model_trains/SLS/epoch_10.pth")


