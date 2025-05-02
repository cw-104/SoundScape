from sound_scape.backend.Isolate import separate_file
from argparse import ArgumentParser
import os
from colorama import Fore, init
from queue import Queue
from threading import Thread
from tqdm import tqdm
from time import sleep
init(autoreset=True)


def clean_basename(filep):
    filep = os.path.basename(filep)
    # remove anything thats not a letter or number
    filep = ''.join(e for e in filep if e.isalnum()).replace("sep","").replace("mp3","")
    return filep


if __name__ == "__main__":
    parser = ArgumentParser(description="Separate a dataset into different files.")
    # args: max-threads, soundscape-dataset-path, subset (eval, dev, train)
    parser.add_argument(
        "--max-threads",
        type=int,
        default=4,
        help="Maximum number of threads to use for separation.",
    )
    parser.add_argument(
        "--soundscape-dataset-path",
        type=str,
        default="../../soundscape-dataset/",
        help="Path to the soundscape dataset. (Default: '../../soundscape-dataset/')",
    )
    parser.add_argument(
        "--out-folder",
        type=str,
        default=None,
        help="Path to the folder where the separated files will be saved, ignore for most cases | default will just isolate directly into dataset",
    )
    parser.add_argument(
        "--subset",
        type=str,
        default="eval",
        choices=["eval", "dev", "train"],
        help="Subset of the dataset to process. (Default: 'eval')",
    )
    args = parser.parse_args()
    
    max_threads = args.max_threads
    soundscape_dataset_path = args.soundscape_dataset_path
    subset = args.subset
    # out folder for separation is dataset folder by def
    out_folder = args.out_folder if args.out_folder else os.path.join(soundscape_dataset_path, subset)
    if soundscape_dataset_path == "../../soundscape-dataset/":
        print(f"{Fore.YELLOW}USING DEFAULT DATASET PATH: '../../soundscape-dataset/'")
    else:
        print(f"{Fore.BLUE}Soundscape dataset path: ", soundscape_dataset_path)
    if subset == "eval":
        print(f"{Fore.YELLOW}USING DEFAULT SUBSET: 'eval'")
    else:
        print(f"{Fore.BLUE}Subset: ", subset)
    print(f"{Fore.BLUE}Max threads: ", max_threads)

    isolated_save_folder_real = os.path.join(out_folder, "bonafide-isolated")
    isolated_save_folder_fake = os.path.join(out_folder, "fake-isolated")


    if not os.path.exists(isolated_save_folder_real):
        os.makedirs(isolated_save_folder_real)
    if not os.path.exists(isolated_save_folder_fake):
        os.makedirs(isolated_save_folder_fake)

    real_files = [x for x in os.listdir(os.path.join(soundscape_dataset_path, subset, "bonafide")) if x.endswith(".mp3")]
    fake_files = [x for x in os.listdir(os.path.join(soundscape_dataset_path, subset, "fake")) if x.endswith(".mp3")]

    worker_threads = []
    def isolate_worker(filep, save_folder):
        global progress_bar
        separate_file(
            filep,
            save_folder,
            model="htdemucs",
            mp3=True,
        )
        progress_bar.update(1)
    
    file_queue = Queue()

    # put real and fake files in queue
    for real_file in real_files:
        file_queue.put((real_file, isolated_save_folder_real))
    for fake_file in fake_files:
        file_queue.put((fake_file, isolated_save_folder_fake))
    global progress_bar
    progress_bar = tqdm(total=file_queue.qsize(), desc="Isolating files", unit="file", leave=True, position=max_threads)
    while not file_queue.empty():
        # clear inactive threads
        worker_threads = [ x for x in worker_threads if x.is_alive() ]
        # if we have less than max_threads, start a new thread
        while len(worker_threads) < max_threads and not file_queue.empty():
            filep, save_folder = file_queue.get()
            worker_thread = Thread(target=isolate_worker, args=(filep, save_folder))
            worker_thread.start()
            worker_threads.append(worker_thread)
            sleep(.1)
        sleep(.1)
        progress_bar.refresh()
