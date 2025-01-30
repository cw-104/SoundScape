from backend.Evaluate import init_whisper_specrnet, eval_file
from backend.Isolate import separate_file
from backend.Evaluate import DeepfakeClassificationModel
from backend.Results import DfResultHandler
import threading
from queue import Queue
from json import dumps as to_json

class ModelBindings:
    def __init__(self):
        self.whisper_model, self.rawgat_model, self.config, self.device = init_whisper_specrnet()
        self.rawgat_model = DeepfakeClassificationModel(result_handler=DfResultHandler(-3, "Fake", "Real", 10, .45))

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
        if not self.file_ids.exists(id):
            self.file_ids.update_state(id, 'Does not exist')
            return
        # set to processing state
        self.file_ids.update_state(id, 'processing')
        # get path
        path = self.file_ids.get_path(id)

        # Separate the file
        # sep_file = separate_file(path, os.path.join(UPLOADS_FOLDER, "separated-uploads"), mp3=True)
        sep_file = path # skip separation for testing

        # Eval Whisper
        pred, label, str_out = eval_file(path, self.whisper_model, self.config, self.device)
        pred_sep, label_sep, str_out_sep = eval_file(sep_file, self.whisper_model, self.config, self.device)
        
        # Eval RawGAT
        rawgat_res = self.rawgat_model.evaluate_file(path)
        rawgat_res_sep= self.rawgat_model.evaluate_file(sep_file)

        # to json results
        result = to_json({
            'status': 'finished',
            'whisper': {
            'unseparated_results': {
                'prediction': pred,
                'label': label,
                'output': str_out
            },
            'separated_results': {
                'prediction': pred_sep,
                'label': label_sep,
                'output': str_out_sep
            }
            },
            'rawgat': {
                'results': {
                'prediction': rawgat_res.raw_value,
                'label': rawgat_res.classification,
            },
            'separated_results': {
                'prediction': rawgat_res_sep.raw_value,
                'label': rawgat_res_sep.classification,
            }
            }
        })
        self.file_ids.set_results(id, result)


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
        self.update_state(id, 'finished')
        self.file_ids[id]['results'] = results

    def get_status(self, id):
        if not self.exists(id):
            return {
                'status': 'error: not found'
            }
        return self.file_ids[id]['status']

    def has_results(self, id):
        return 'results' in self.file_ids[id]
    def get_results(self, id):
        if not self.has_results(id):
            return {
                status: 'error: not finished'
            }
        return self.file_ids[id]['results']

    def exists(self, id):
        return id in self.file_ids.keys()
