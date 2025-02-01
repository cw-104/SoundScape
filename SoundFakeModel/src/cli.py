import torch
import yaml
import argparse
import os 
from backend.whisper_eval import evaluate_nn
from backend.whisper_specrnet import WhisperSpecRNet, set_seed
from backend.Isolate import separate_file


from backend.Evaluate import DeepfakeClassificationModel
from backend.Results import DfResultHandler
from backend.Isolate import separate_file
from Base_Path import get_path_relative_base
def eval_file(f, preloaded_model, config, device):

    # Evaluate a single file
    pred, label = evaluate_nn(
        model=preloaded_model,
        model_config=config["model"],
        device=device,
        single_file=f,  # Specify the single file path
    )
    return pred, 'spoof' if label == 0 else 'real', f"{('spoof' if label == 0 else 'real'):<10} (raw value: {pred:.4f})"

def main():
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Evaluate a model on a single file")
    parser.add_argument(
        "--file", type=str, required=False, default=None, help="Path to the single audio file to evaluate"
    )
    parser.add_argument(
        "--folder", type=str, required=False, default=None, help="Path to the audio folder to evaluate",
    )
    parser.add_argument(
        "--device", type=str, required=False, default="cpu", help="Device to use (cuda, mps, cpu)",
    )
    parser.add_argument(
        "--threshold", type=float, required=False, default=.49, help="Threshold > than this value is considered real, else fake",
    )
    parser.add_argument(
        "--sep", action="store_true", help="Separate the audio file before evaluation"
    )
    parser.add_argument(
        "--both", action="store_true", help="Eval before and after separation"
    )
    parser.add_argument(
        "--reval_threshold", type=float, required=False, default=0, help="threshold above or below which to revaluate on better separation model"
    )
    parser.add_argument(
        "--no_sep_threshold", type=float, required=False, default=0, help="threshold above or below which to revaluate without separation"
    )
    parser.add_argument(
        "--rm_sil", action="store_true", help="remove stretchs of silence from audio file before evaluation"
    )
    parser.add_argument(
        "--low_conf", type=float, required=False, default=0.01, help="threshold below which to lose confidence for deepfake and toss result"
    )
    parser.add_argument(
        "--flip", action="store_true", help="flip to predict DF"
    )
    args = parser.parse_args()
    if not args.file and not args.folder:
        print("Please specify a file or folder to evaluate")
        return
    # Set device
    device = args.device
    threshold = args.threshold
    reval_threshold = args.reval_threshold
    no_sep_threshold = args.no_sep_threshold
    # config = yaml.safe_load(open("pretrained_models/whisper_specrnet/config.yaml", "r"))
    config = yaml.safe_load(open(get_path_relative_base("pretrained_models/whisper_specrnet/config.yaml"), "r"))
    model_name, model_parameters = config["model"]["name"], config["model"]["parameters"]
    print(f"loading model...\n")
    model = WhisperSpecRNet(
        input_channels=config.get("input_channels", 1),
        freeze_encoder=config.get("freeze_encoder", False),
        device=device,
    )
    # model.load_state_dict(torch.load("pretrained_models/whisper_specrnet/weights.pth", map_location=device))
    model.load_state_dict(torch.load(get_path_relative_base("pretrained_models/whisper_specrnet/weights.pth"), map_location=device))
        


    seed = config["data"].get("seed", 42)
    set_seed(seed)


    # print("----prediction metric----")
    # print("if raw value is <.5 label 0 else 1 ")
    # print("1 = real, 0 = fake")


    model_rawgat = DeepfakeClassificationModel(result_handler=DfResultHandler(-3, "Fake", "Real", 3, .95))
    print("\n\n\n\n\n")
    # Single file path
    if args.file:
        file = args.file
        if args.sep:
            print("separating file...")
            file = separate_file(args.file, output_dir=os.path.abspath("separated"), model="htdemucs", mp3=True, trim_silence=args.rm_sil)
            print("separated file path: ", file)
            if file is None:
                print("Seperation failed")
                return
        pred, _, r = eval_file(file, model, config, device)
        if pred < args.low_conf:
            if not args.sep:
                #eval with other model
                # print(f"low confidence in prediction, reevaluating {file} with different model")
                res = model_rawgat.evaluate_file(os.path.join(file))
                pred = res.percent_certainty # scale to proper decimal place
            else: # if we are sepating evaluate based on the delta change before and after seperation
                # delta change before and after seperation
                #delta = model_rawgat.evaluate_file(file).raw_value - model_rawgat.evaluate_file(args.file).raw_value
                #pred = max(delta/5, .95)
                pred_norm, _, _ = eval_file(file, model, config, device)
                pred_sep, _, _ = eval_file(args.file, model, config, device)
                delta = pred_sep - pred_norm
                print("delta change in prediction after separation: ", delta)
                if delta < 0:
                    pred = threshold - pred
                else: 
                    pred = threshold + pred

        if pred < threshold: 
            label = 'Fake'
            percent = threshold - pred
        else:
            label = 'Real'
            percent = pred - threshold
        percent *= 100
        print(f"Evaluation: {label} ({max(percent, .95):.1f}% confidence) (raw: {pred:.4f})")


    if args.folder:
        print("evaluating ", args.folder)
        print()
        total = 0
        n_rl = 0
        for f in sorted(os.listdir(args.folder)):
            # if is an audio file wav/flac/mp3 and file name 
            if f.endswith(".wav") or f.endswith(".flac") or f.endswith(".mp3"):
                print("-------------------")
                file = f
                if args.sep:
                    print("separating file...")
                    file = separate_file(os.path.join(args.folder, f), output_dir=os.path.abspath("separated"), model="htdemucs", mp3=True, trim_silence=args.rm_sil)
                    print("separated file path: ", file)
                    print()
                    if file is None:
                        print("Seperation failed")
                        continue
                pred, label, str_out = eval_file(os.path.join(args.folder, file), model, config, device)
                # re-eval if pred is close to threshold ie is abnormally high or low surity (ie 0.01 or .99) might suggest artificating
                if pred < reval_threshold or pred > (1 - reval_threshold):
                    # print("revaluting with better separation model")
                    file = separate_file(os.path.join(args.folder, f), output_dir=os.path.abspath("separated"), model="htdemucs_ft", mp3=True, trim_silence=args.rm_sil)
                    pred, label, str_out = eval_file(file, model, config, device)
                    print()
                if pred < args.low_conf:
                    #eval with other model
                    # print(f"low confidence in prediction, reevaluating {file} with different model")
                    res = model_rawgat.evaluate_file(os.path.join(args.folder, file))
                    pred = res.percent_certainty 
                    if res.is_lower_class:
                        pred = threshold - res.percent_certainty 
                    else: 
                        pred = threshold + res.percent_certainty
                    pred = max(min(pred, .8), 0.1)


                total+=1
                n_rl+= 1 if pred > threshold else 0
                # if pred < threshold: 
                #     print(f"<F>alse negative {str_out} {file}")
                if pred < threshold: 
                    label = 'Fake'
                    percent = threshold - pred
                else:
                    label = 'Real'
                    percent = pred - threshold
                percent *= 100
                print(f"Evaluation: {label} ({max(percent, .95):.1f}% confidence) (raw: {pred:.4f}) (file: {file})")
                if args.flip:
                    print(f"Percent DF {total-n_rl}/{total}: {((total-n_rl)/total)*100:2.2f}%")
                else:
                    print(f"Percent Real {n_rl}/{total}: {(n_rl/total)*100:2.2f}%")
                print("-------------------")


if __name__ == "__main__":
    main()
