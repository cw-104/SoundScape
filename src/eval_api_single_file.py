import os
from argparse import ArgumentParser
from colorama import init, Fore
from sound_scape.api.Bindings import ModelBindings
from sound_scape.backend.Isolate import separate_file
"""
CORRECT
py eval_api_single_file.py --file "/Users/christiankilduff/Downloads/new_audip/real/SZA - Snooze (Lyrics) [CBx6e9cZlBQ].mp3" --correct_label "Real"

CORRECT
py eval_api_single_file.py --file "/Users/christiankilduff/Downloads/new_audip/real/SZA - Nobody Gets Me (Lyrics) [tOTr9CCutiE].mp3" --correct_label "Real"


REMOVE LIVE PERFORMANCES FROM REAL EVAL
"""
init(autoreset=True)


def clean_basename(filep):
    filep = os.path.basename(filep)
    # remove anything thats not a letter or number
    filep = ''.join(e for e in filep if e.isalnum()).replace("sep","").replace("mp3","")
    return filep


if __name__ == "__main__":
    args = ArgumentParser()
    # Args : eval_dataset_path, real_time_isolate_files (if present: true), folder_to_separate_to
    args.add_argument("--file", type=str, required=True, help="Path to file to evaluate")
    args.add_argument("--folder_to_separate_to", type=str, default="eval-separated-single")
    args.add_argument("--correct_label", type=str, required=True, help="Correct label for the file (Real or Fake)")
    # TODO add help print
    args = args.parse_args()
    file_path = args.file
    sep_folder = args.folder_to_separate_to
    correct_label = args.correct_label
    
    print(f"{Fore.YELLOW}File to evaluate: {file_path}")
    print(f"{Fore.CYAN}clean name: {clean_basename(file_path)}")
    print(f"{Fore.YELLOW}Folder to separate to: {sep_folder}")
    print(f"{Fore.YELLOW}Correct label: {correct_label}")

    if not os.path.exists(sep_folder):
        os.mkdir(sep_folder)
    
    api_binding = ModelBindings()

    iso_file = separate_file(file_path, sep_folder, mp3=True)

    model_res, combined_results = api_binding.get_model_results(file_path, iso_file)
    combined_label = combined_results["label"]
    
    print("------full-------")
    print()
    print(f"{Fore.GREEN}{model_res}")
    print()
    print("----combined-----")
    print()
    print(f"{Fore.BLUE}{combined_results}")
    print()

    print(f"{Fore.CYAN}Correct label: {correct_label} | Predicted label: {combined_label}")
    if correct_label == combined_label:
        print(f"{Fore.GREEN}Correct")
    else:
        print(f"{Fore.RED}Incorrect")

    print()
    print("Done")
    print()
    print()