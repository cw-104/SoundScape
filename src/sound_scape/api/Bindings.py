from sound_scape.backend.Isolate import separate_file
import threading, os
from queue import Queue
from json import dumps as to_json
from Base_Path import get_path_relative_base
UPLOADS_FOLDER = get_path_relative_base("uploads")
import requests
import time

AIXPLAIN_API_KEY = "d97f1786ec0701d8809259408d07e4f739f0794bd80a7c30c024004597aff085"
AGENT_ID = "67acb51f56173fdefab4fc62"
POST_URL = f"https://platform-api.aixplain.com/sdk/agents/{AGENT_ID}/run"

headers = {
    "x-api-key": AIXPLAIN_API_KEY,
    "Content-Type": 'application/json'
}

from sound_scape.backend.Models import whisper_specrnet, rawgat, xlsr, vocoder
from sound_scape.backend.clad_integration import run_clad

class ModelBindings:
    def __init__(self):
        self.whisper_model = whisper_specrnet()
        self.rawgat_model = rawgat()
        self.vocoder_model = vocoder()
        #self.xlsr_model = xlsr()
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

    def get_model_results(self, original_path, sep_path):
        # Eval Whisper
        wpred, wlabel = self.whisper_model.evaluate(original_path)
        wpred_sep, wlabel_sep = self.whisper_model.evaluate(sep_path)
        
        # Eval RawGAT
        rpred, rlabel = self.rawgat_model.evaluate(original_path)
        rpred_sep, rlabel_sep= self.rawgat_model.evaluate(sep_path)

        # Eval Vocoder
        vpred, vlabel = self.vocoder_model.evaluate(original_path)

        # Eval XLSR
        # xpred, xlabel = self.xlsr_model.evaluate(original_path) commenting out for now

        #  Run CLAD model on original audio
        clad_result = run_clad(original_path)

        results = {
            "Whisper": [wpred, wlabel, wpred_sep, wlabel_sep],
            "RawGAT": [rpred, rlabel, rpred_sep, rlabel_sep],
            "Vocoder": [vpred, vlabel],
            # "XLSR": [xpred, xlabel], commenting out for now
            "CLAD": clad_result  
        }
        return results
    def process_file(self, file_id):
        file_path = self.file_ids.get_file(file_id)
        if not file_path:
            print(f"[ERROR] File ID {file_id} not found.")
            return

        try:
            sep_path = separate_file(file_path, get_path_relative_base("temp"))
            results = self.get_model_results(file_path, sep_path)
            self.file_ids.set_results(file_id, results)
            self.file_ids.update_status(file_id, "finished")
            print(f"[INFO] Processing complete for {file_id}")
        except Exception as e:
            print(f"[ERROR] Processing failed for {file_id}: {e}")
            self.file_ids.update_status(file_id, "failed")

            


class FileIds:
    def __init__(self):
        self.ids = {}

    def add_file(self, id, file_path):
        self.ids[id] = {"file": file_path, "status": "processing"}

    def get_file(self, id):
        return self.ids.get(id, {}).get("file", None)

    def get_results(self, id):
        original_path = self.get_file(id)
        sep_path = separate_file(original_path, get_path_relative_base("temp"))
        model_results = ModelBindings().get_model_results(original_path, sep_path)
        return model_results
    
    def set_results(self, id, results):
        if id in self.ids:
            self.ids[id]["results"] = results

    
    def exists(self,id):
        return id in self.ids
    
    def update_status(self, id, status):
        if id in self.ids:
            self.ids[id]["status"] = status

    def get_status(self, id):
        if id in self.ids:
            return {"status": self.ids[id]["status"]}
        else:
            return {"status": "error", "error": "Invalid ID"}
