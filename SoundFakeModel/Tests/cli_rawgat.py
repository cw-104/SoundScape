from Evaluate import DeepfakeClassificationModel
from Results import DfResultHandler
from Isolate import separate_file
import argparse
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
        "--threshold", type=float, required=False, default=-3, help="Threshold > than this value is considered real, else fake",
    )
    parser.add_argument(
        "--sep", action="store_true", help="Separate the audio file before evaluation"
    )
    args = parser.parse_args()
    model = DeepfakeClassificationModel(result_handler=DfResultHandler(args.threshold, "Fake", "Real", 10, .95))

    if args.file:
        file = args.file
        if args.sep:
            file = separate_file(file, "separated")
        print(model.evaluate_file(file))
    elif args.folder:
        results = model.evaluate_folder(args.folder,progress_bar=True)
        n_rl = 0
        for r in results:
            if not r.is_lower_class: n_rl += 1
            print(r)
        print(f"percent real: {((n_rl/len(results)) * 100):.2f}%")
        
if __name__ == "__main__":
    main()