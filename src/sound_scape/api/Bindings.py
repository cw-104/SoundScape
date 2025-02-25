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

class ModelBindings:
    def __init__(self):
        self.whisper_model = whisper_specrnet()
        self.rawgat_model = rawgat()
        self.vocoder_model = vocoder(device='mps')
        self.xlsr_model = xlsr(device='mps')
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
        vpred_sep, vlabel_sep = self.vocoder_model.evaluate(sep_path)

        # Eval XLSR
        xpred, xlabel = self.xlsr_model.evaluate(original_path)
        xpred_sep, xlabel_sep = self.xlsr_model.evaluate(sep_path)

        # to json results
        return {
            'status': 'finished',
            'whisper': {
            'unseparated_results': {
                'prediction': wpred,
                'label': wlabel,
            },
            'separated_results': {
                'prediction': wpred_sep,
                'label': wlabel_sep,
            }
            },
            'rawgat': {
            'unseparated_results': {
                'prediction': rpred,
                'label': rlabel,
            },
            'separated_results': {
                'prediction': rpred_sep,
                'label': rlabel_sep,
            },
            },
            'xlsr': {
            'unseparated_results': {
                'prediction': xpred,
                'label': xlabel,
            },
            'separated_results': {
                'prediction': xpred_sep,
                'label': xlabel_sep,
            },
            },
            'vocoder': {
            'unseparated_results': {
                'prediction': vpred,
                'label': vlabel,
            },
            'separated_results': {
                'prediction': vpred_sep,
                'label': vlabel_sep
            }}}


    def append_explain_results(self, json):
        data = {
        "query": "We are using 5 models to predict the results of an audio file based on if it is a deepfake or authentic voice. We need you to explain to the user of our site the results of the model and what they mean. We are giving you the results and then want you to give a short explanation to the user as to why the answer is what it is based on the results. Here is the data: " + to_json(json)+ "Give a simple paragraph for the user explaining how the data results give the final outcome answer of real or fake.",
	    # "sessionId": "<SESSIONID_TEXT_DATA>",  # Optional: Specify sessionId from the previous message
        }

        # POST request to execute the agent
        response = requests.post(POST_URL, headers=headers, json=data)
        response_data = response.json()
        request_id = response_data.get("requestId")

        get_url = f"https://platform-api.aixplain.com/sdk/agents/{request_id}/result"
        
      
        # Polling loop: GET request until the result is completed
        while True:
            get_response = requests.get(get_url, headers=headers)
            result = get_response.json()
    
            if result.get("completed"):
                print(result)
                break
            else:
                print("results not arrived")
                time.sleep(5) # Wait for 5 seconds before checking the result again
        json['explaination'] = result['data']['output']
        return json

    def append_identification_results(self, json):
        # pass vocal to match
        artist = "(placeholder) Taylor Swift"

        json['identified-artist'] = artist
        return json

    def process_file(self, id):
        if not self.file_ids.exists(id):
            self.file_ids.update_state(id, 'Does not exist')
            return
        # set to processing state
        self.file_ids.update_state(id, 'processing')
        # get path
        path = self.file_ids.get_path(id)

        # Separate the file
        sep_file = separate_file(path, os.path.join(UPLOADS_FOLDER, "separated-uploads"), mp3=True)

        # get model eval results
        result_json = self.get_model_results(path, sep_file)


        # add ai explaination
        result_json = self.append_explain_results(result_json)

        # match the vocals to an artist
        result_json = self.append_identification_results(result_json)


        self.file_ids.set_results(id, to_json(result_json))


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
