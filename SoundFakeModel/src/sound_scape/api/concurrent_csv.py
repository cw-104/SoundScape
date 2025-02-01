import csv, os, time, threading
from multiprocessing import Process, Lock
from queue import Queue

class ConcurrentCsv:

    _WRITE_ACTION = 0
    _UPDATE_ACTION = 1

    def __init__(self, csv_file, col_names):
        self.lock = Lock()
        self._data_queue = Queue()
        self._thread = None
        self.csv_file_path = csv_file
        self.col_names = col_names

        if not os.path.exists(csv_file):
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(col_names)


    def insert_data(self, values : [str]):
        self._data_queue.put({
            'action': _WRITE_ACTION,
            'values': values
        })
        try_start_thread()
        

    def update_data(self, row_index : int, col_value_pairs: [[int, str]] ):
        self._data_queue.put({
            'action': _UPDATE_ACTION,
            'row_index': row_index,
            'col_value_pairs': col_value_pairs
        })
        try_start_thread()

    def try_start_thread(self):
        if not self._thread or not self._thread.is_alive():
            self._thread = threading.Thread(target=self._update_csv_from_queue)
            self._thread.start()
    
    def find_row_index(self, col_index : int, search_value):
        # get data from csv
        if self.lock.acquire(timeout=10):
            try:
                with open(self.csv_file_path, 'r', newline='') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if row[col_index] == search_value:
                            return row
            finally:
                lock.release()

    """
    returns first row that matches search value
    """
    def find_data_by_search(self, col_index : int, search_value):
        if self.lock.acquire(timeout=10):
            try:
                with open(self.csv_file_path, 'r', newline='') as f:
                    reader = csv.reader(f)
                    for row in reader:
                        if row[col_index] == search_value:
                            return row
            finally:
                lock.release()

    def get_data_row(self, row_index : int):
        if self.lock.acquire(timeout=10):
            try:
                with open(self.csv_file_path, 'r', newline='') as f:
                    reader = csv.reader(f)
                    for i, row in enumerate(reader):
                        if i == row_index:
                            return row
            finally:
                lock.release()

    def _update_csv_from_queue(self):
        if self.lock.acquire(timeout=10):  # Wait for up to 5 seconds
            try:
                with open(self.csv_file_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    while not self._data_queue.empty():
                        
                        action_map = self._data_queue.get()
                        if action_map['action'] == _WRITE_ACTION:
                            writer.writerow(
                                action_map['values']
                            )
                        elif action_map['action'] == _UPDATE_ACTION:
                            for col, value in action_map['col_value_pairs']:
                                row[col] = value
                            writer.writerow(row)
                        else:
                            print(f'Unknown action {ACTION} for pairs {pairs}')
                        
            finally:
                lock.release()  # Ensure the lock is released
        else:
            print(f'Could not acquire lock for Status={status}, Results={results}, skipping write.')

