from sound_scape.backend.Isolate import separate_file
import threading
from queue import Queue
from json import dumps as to_json

from sound_scape.backend.Models import whisper_specrnet, rawgat

class ModelBindings:
    def __init__(self):
        self.whisper_model = whisper_specrnet()
        self.rawgat_model = rawgat()
        self.file_processing_queue = Queue()
        self.processing_thread = None

        self.file_ids = FileIds()

    def run_processing_thread(self):
        while not self.file_processing_queue.empty():
            self.process_file(self.file_processing_queue.get())

    def upload_file(self, id, file_name):
        self.file_ids.add_file(id, file_name)
        self.file_processing_queue.put(id)
        print('checking thread')
        if not self.processing_thread or not self.processing_thread.is_alive():
            print("running processing thread")
            self.processing_thread = threading.Thread(target=self.run_processing_thread)
            self.processing_thread.start()

    def process_file(self, id):
        print(f"DEBUG: Starting processing for ID {id}")

    # Check if file exists in file_ids
        if not self.file_ids.exists(id):
            print(f"ERROR: File ID {id} does not exist!")
            self.file_ids.update_state(id, "Does not exist")
            return

    # Set status to "processing"
        print(f"DEBUG: Setting status to 'processing' for ID {id}")
        self.file_ids.update_state(id, "processing")

        # Get file path
        path = self.file_ids.get_path(id)
        print(f"DEBUG: File path for ID {id}: {path}")

        # Skip separation for now
        sep_file = path  # Keep as is for testing

        try:
            # Evaluate with Whisper model
            print(f"DEBUG: Evaluating Whisper model for ID {id}")
            wpred, wlabel = self.whisper_model.evaluate(path)
            wpred_sep, wlabel_sep = self.whisper_model.evaluate(sep_file)

            # Evaluate with RawGAT model
            print(f"DEBUG: Evaluating RawGAT model for ID {id}")
            rpred, rlabel = self.rawgat_model.evaluate(path)
            rpred_sep, rlabel_sep = self.rawgat_model.evaluate(sep_file)

            # Format results
            result = to_json({
                "status": "finished",
                "whisper": {
                    "unseparated_results": {"prediction": wpred, "label": wlabel},
                    "separated_results": {"prediction": wpred_sep, "label": wlabel_sep},
                },
                "rawgat": {
                    "unseparated_results": {"prediction": rpred, "label": rlabel},
                    "separated_results": {"prediction": rpred_sep, "label": rlabel_sep},
                }
            })

            # Update file_ids with results
            print(f"DEBUG: Saving results for ID {id}")
            self.file_ids.set_results(id, result)

        except Exception as e:
            print(f"ERROR: Processing failed for ID {id}: {e}")
            self.file_ids.update_state(id, f"error: {str(e)}")



class FileIds:
    def __init__(self):
        """
        structure:
        {
            <id>: {
                filename: <filename>,
                status: {
                    state: <uploaded/processing/finished>,
                }
                [after processing] results: {
                    }
                }
        }
        """
        self.file_ids = {}

    def update_state(self, id, state):
        self.file_ids[id]['status']['state'] = state

    def get_path(self, id):
        return self.file_ids[id]['filename']

    def add_file(self, id, filename):
        self.file_ids[id] = {
            'filename': filename,
            'status': {
                'state': 'uploaded'
                }
            }
    def set_results(self, id, results):
        if id not in self.file_ids:
            print(f"ERROR: Tried to set results for missing file ID {id}")
            return

        #ensure "status" exists before updating
        if "status" not in self.file_ids[id]:
            self.file_ids[id]["status"] = {}

        #eroperly update state to "finished"
        self.file_ids[id]["status"]["state"] = "finished"
        self.file_ids[id]["results"] = results  

        print(f"DEBUG: Results set for {id}: {self.file_ids[id]}")
    def get_status(self, id):
        if not self.exists(id):
            return {'status': 'error: not found'}

        #should make sure status and state exist before it tries to return
        status_data = self.file_ids.get(id, {}).get("status",{})

        #check and return finished when exists
        state = status_data.get("state", "processing")

        print(f"DEBUG: get_status() called for {id}, returning state: {state}") #debus

        return {"state": state}

    def has_results(self, id):
        return id in self.file_ids and "results" in self.file_ids[id]
    def get_results(self, id):
        if not self.exists(id):
            return {
                'status': 'error: not finished'
            }
        if not self.has_results(id):
            return{"status": "error: not finished"}
        return self.file_ids.get(id, {}).get("results", {"status": "error: no results"})

    def exists(self, id):
        return id in self.file_ids.keys()
