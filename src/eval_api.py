from sound_scape.api.Bindings import ModelBindings
from threading import Thread
from time import sleep
from queue import Queue
from argparse import ArgumentParser
from tqdm import tqdm
import os
import json
from sound_scape.backend.Isolate import separate_file


class api_binding_thread:
    def __init__(self):
        self._file_queue = Queue()
        self._thread = Thread(target=self._run)
        self._thread.start()
        self.results = []
        self._isolation_queue = Queue()
        self.max_isolation_threads = 4
        self._isolation_thread = None
        self.n_complete = 0

    # isolation thread manager
    def _iso_thread_run(self):
        # worker task thread
        def sep_worker(filep, folder_to_sep_to, correct_label):
            # check if file already in folder
            not_in_folder = True
            for p in os.listdir(folder_to_sep_to):
                if clean_basename(p) in clean_basename(filep):
                    print("File already isolated")
                    iso_file = os.path.join(folder_to_sep_to, p)
                    not_in_folder = False

            
            if not_in_folder:
                # separate the file
                iso_file = separate_file(filep, folder_to_sep_to, mp3=True)
            
            # add to the queue
            self._file_queue.put({
                'file': filep,
                'separated_file': iso_file,
                'folder_to_sep_to': folder_to_sep_to,
                'correct_label': correct_label,
            })

        worker_threads = []
        while not self._isolation_queue.empty():
            while len(worker_threads) < self.max_isolation_threads:
                params = self._isolation_queue.get()
                filep = params['file']
                folder_to_sep_to = params['folder_to_sep_to']
                correct_label = params['correct_label']
                worker_thread = Thread(target=sep_worker, args=(filep, folder_to_sep_to, correct_label))
                worker_thread.start()
                worker_threads.append(worker_thread)
            # remove finished threads from list
            worker_threads = [t for t in worker_threads if t.is_alive()]
            sleep(1)

    def _run(self):
        print("starting api eval thread")
        self._bindings = ModelBindings()

        # remove unnessary memory
        del self._bindings.identifier
        del self._bindings.file_ids
        del self._bindings.processing_thread

        with open("api_debug_all_results.txt", "w") as f:
            f.write("")
        while True:
            if not self._file_queue.empty():
                eval_params = self._file_queue.get()
                _, combined_res = self._evaluate(filep=eval_params['file'], correct_label=eval_params['correct_label'], iso_file=eval_params['separated_file'], folder_to_sep_to=eval_params['folder_to_sep_to'])
                self.results.append(combined_res)
                # self.model_results.append(model_res)
                print("Evaled", eval_params['separated_file'])
                self.n_complete += 1
            sleep(.01)

    def queue_eval(self, filep, correct_label, iso_file=None, folder_to_sep_to="eval-separated"):
        if correct_label == "Real":
            folder_to_sep_to = os.path.join(folder_to_sep_to, "Real-iso")
        elif correct_label == "Fake":
            folder_to_sep_to = os.path.join(folder_to_sep_to, "Fake-iso")
        if iso_file:
            self._file_queue.put({
                'file': filep,
                'separated_file': iso_file,
                'folder_to_sep_to': folder_to_sep_to,
                'correct_label': correct_label,
            })
        else:
            # isolate in real time
            self._isolation_queue.put({
                'file': filep,
                # 'separated_file': iso_file,
                'folder_to_sep_to': folder_to_sep_to,
                'correct_label': correct_label,
            })
            # If isolation management thread not active, create the thread
            if self._isolation_thread == None or not self._isolation_thread.is_alive():
                self._isolation_thread = Thread(target=self._iso_thread_run)
                self._isolation_thread.start()


    def _evaluate(self, filep, correct_label, iso_file=None, folder_to_sep_to="eval-separated"):
        # Call the evaluate method on the bindings object
        # if not iso_file:
            # Separate the file if not precomputed
            # iso_file = separate_file(filep, folder_to_sep_to, mp3=True)
        # get model eval results
        model_res, combined_results = self._bindings.get_model_results(filep, iso_file)
        model_res['filep'] = filep
        with open("api_debug_all_results.txt", "a") as f:
            f.write(json.dumps(model_res))
            f.write("\n")
        return model_res, {
            "prediction": combined_results["prediction"],
            "label": combined_results["label"],
            "correct_label": correct_label
        }

def clean_basename(filep):
    filep = os.path.basename(filep)
    # remove anything thats not a letter or number
    filep = ''.join(e for e in filep if e.isalnum()).replace("sep","").replace("mp3","")
    return filep

def get_matching_files(og_files, iso_files):
    """
    gets an array of files matched to their isolated and non-isolated forms
    """

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
    # Args : eval_dataset_path, real_time_isolate_files (if present: true), folder_to_separate_to
    args.add_argument("--eval_dataset_path", type=str, default="../../soundscape-dataset/eval/")
    args.add_argument("--real_time_isolate_files", action="store_true", default=False)
    args.add_argument("--folder_to_separate_to", type=str, default="eval-separated")
    # TODO add help print
    args = args.parse_args()
    eval_dataset_path = args.eval_dataset_path
    use_dataset_iso = not args.real_time_isolate_files
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


    
    
    api_binding_thread = api_binding_thread()

    if use_dataset_iso:
        real_iso_files = os.listdir(os.path.join(eval_dataset_path, "bonafide-isolated"))
        real_files = get_matching_files(real_files, real_iso_files)
        fake_iso_files = os.listdir(os.path.join(eval_dataset_path, "fake-isolated"))
        fake_files = get_matching_files(fake_files, fake_iso_files)

        for og_real, iso_real in real_files:
            og_filep = os.path.join(eval_dataset_path, "bonafide", og_real)
            iso_filep = os.path.join(eval_dataset_path, "bonafide-isolated", iso_real)

            # labels are "Real" and "Fake" (bonafide is only in dataset naming)
            api_binding_thread.queue_eval(filep=og_filep, iso_file=iso_filep, correct_label="Real", folder_to_sep_to=sep_folder)
        
        for og_fake, iso_fake in fake_files:
            og_filep = os.path.join(eval_dataset_path, "fake", og_fake)
            iso_filep = os.path.join(eval_dataset_path, "fake-isolated", iso_fake)

            # labels are "Real" and "Fake" (bonafide is only in dataset naming)
            api_binding_thread.queue_eval(filep=og_filep, iso_file=iso_filep, correct_label="Fake", folder_to_sep_to=sep_folder)

    else:
        for real_file in real_files:
            filep = os.path.join(eval_dataset_path, "bonafide", real_file)
            # labels are "Real" and "Fake" (bonafide is only in dataset naming)
            api_binding_thread.queue_eval(filep=filep, correct_label="Real", folder_to_sep_to=sep_folder)
        
        for fake_file in fake_files:
            filep = os.path.join(eval_dataset_path, "fake", fake_file)
            # labels are "Real" and "Fake" (bonafide is only in dataset naming)
            api_binding_thread.queue_eval(filep=filep, correct_label="Fake", folder_to_sep_to=sep_folder)
            
total = len(real_files) + len(fake_files)


def save_results():
    # save binding_thread results to a file
    with open("eval_api_results.txt", "w") as f:
        f.write(json.dumps(api_binding_thread.results))

progress_bar = tqdm(total=total, desc="Classifying files", unit="file", position=1, leave=True) 
# while not api_binding_thread._file_queue.empty():
while api_binding_thread.n_complete < total:
    sleep(1)
    # wait for the thread to finish
    progress_bar.n = api_binding_thread.n_complete
    progress_bar.refresh()

print("Sleeping for 2.5 min as last file might not be finished")
sleep(60 * 2.5)

save_results()