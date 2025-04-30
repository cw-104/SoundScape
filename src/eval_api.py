from sound_scape.api.Bindings import ModelBindings
from threading import Thread
from time import sleep
from queue import Queue
from argparse import ArgumentParser
from tqdm import tqdm
import os

class binding_thread:
    def __init__(self):
        self._file_queue = Queue()
        self._thread = Thread(target=self._run)
        self._thread.start()
        self.results = []

    def _run(self):
        print("starting api eval thread")
        self._bindings = ModelBindings()
        while True:
            if not self._file_queue.empty():
                eval_params = self._file_queue.get()
                combined_res = self._evaluate(filep=eval_params['file'], correct_label=eval_params['correct_label'], iso_file=eval_params['separated_file'], folder_to_sep_to=eval_params['folder_to_sep_to'])
                self.results.append(combined_res)
            sleep(.01)

    def queue_eval(self, filep, correct_label, iso_file=None, folder_to_sep_to="eval-separated"):
        
        self._file_queue.put({
            'file': filep,
            'separated_file': iso_file,
            'folder_to_sep_to': folder_to_sep_to,
            'correct_label': correct_label,
        })

    def _evaluate(self, filep, correct_label, iso_file=None, folder_to_sep_to="eval-separated"):
        # Call the evaluate method on the bindings object
        if not iso_file:
            # Separate the file if not precomputed
            iso_file = self._bindings.separate_file(filep, folder_to_sep_to, mp3=True)
        # get model eval results
        _, combined_results = self._bindings.get_model_results(filep, iso_file)
        print("\nevaluating\n")
        return {
            "prediction": combined_results["prediction"],
            "label": combined_results["label"],
            "correct_label": correct_label
        }

def get_matching_files(og_files, iso_files):
    """
    gets an array of files matched to their isolated and non-isolated forms
    """
    def clean_basename(filep):
        filep = os.path.basename(filep)
        # remove anything thats not a letter or number
        filep = ''.join(e for e in filep if e.isalnum()).replace("sep","").replace("mp3","")
        return filep

    matched = [] # list of tuples (og, iso)
    for iso_file in iso_files:
        iso_file_clean = clean_basename(iso_file)
        for og_file in og_files:
            og_file_clean = clean_basename(og_file)
            if og_file_clean in iso_file_clean:
                matched.append((og_file, iso_file))
                break
    
    return matched


if __name__ == "__main__":
    args = ArgumentParser()
    # Args : eval_dataset_path, use_dataset_iso (true/false), folder_to_separate_to
    args.add_argument("--eval_dataset_path", type=str, default="../../soundscape-dataset/eval/")
    args.add_argument("--use_dataset_isolated-files", type=bool, default=True)
    args.add_argument("--folder_to_separate_to", type=str, default="eval-separated")
    args = args.parse_args()
    eval_dataset_path = args.eval_dataset_path
    use_dataset_iso = args.use_dataset_isolated_files
    sep_folder = args.folder_to_separate_to
    print("Eval dataset path: ", eval_dataset_path)
    print("Use precomputed isolated files: ", use_dataset_iso)
    print("Folder to separate to: ", sep_folder)
    if not os.path.exists(eval_dataset_path):
        os.mkdir(sep_folder)

    # "bonafide"
    # "fake"

    # "bonafide-isolated"
    # "fake-isolated"


    real_files = os.listdir(os.path.join(eval_dataset_path, "bonafide"))
    fake_files = os.listdir(os.path.join(eval_dataset_path, "fake"))


    
    
    binding_thread = binding_thread()

    if use_dataset_iso:
        real_iso_files = os.listdir(os.path.join(eval_dataset_path, "bonafide-isolated"))
        real_files = get_matching_files(real_files, real_iso_files)
        fake_iso_files = os.listdir(os.path.join(eval_dataset_path, "fake-isolated"))
        fake_files = get_matching_files(fake_files, fake_iso_files)

        for og_real, iso_real in tqdm(real_files, desc="Processing real files", unit="file", position=0, leave=False):
            og_filep = os.path.join(eval_dataset_path, "bonafide", og_real)
            iso_filep = os.path.join(eval_dataset_path, "bonafide-isolated", iso_real)

            # labels are "Real" and "Fake" (bonafide is only in dataset naming)
            binding_thread.queue_eval(filep=og_filep, iso_file=iso_filep, correct_label="Real", folder_to_sep_to=sep_folder)
        for og_fake, iso_fake in tqdm(fake_files, desc="Processing fake files", unit="file", position=0, leave=False):
            og_filep = os.path.join(eval_dataset_path, "fake", og_fake)
            iso_filep = os.path.join(eval_dataset_path, "fake-isolated", iso_fake)

            # labels are "Real" and "Fake" (bonafide is only in dataset naming)
            binding_thread.queue_eval(filep=og_filep, iso_file=iso_fake, correct_label="Fake", folder_to_sep_to=sep_folder)

    else:
        for real_file in tqdm(real_files, desc="Processing real files", unit="file", position=0, leave=fake_files):
            filep = os.path.join(eval_dataset_path, "bonafide", real_file)
            # labels are "Real" and "Fake" (bonafide is only in dataset naming)
            binding_thread.queue_eval(filep=filep, correct_label="Real", folder_to_sep_to=sep_folder)
        for fake_file in tqdm(fake_files, desc="Processing fake files", unit="file", position=0, leave=False):
            filep = os.path.join(eval_dataset_path, "fake", fake_file)
            # labels are "Real" and "Fake" (bonafide is only in dataset naming)
            binding_thread.queue_eval(filep=filep, correct_label="fake", folder_to_sep_to=sep_folder)
            
while not binding_thread._file_queue.empty():
    sleep(1)
    # wait for the thread to finish

print("Writing results to file")
# save binding_thread results to a file
with open("eval_api_results.txt", "w") as f:
    for result in binding_thread.results:
        f.write(f"{result['prediction']},{result['label']},{result['correct_label']}\n")