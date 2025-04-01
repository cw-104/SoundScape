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


from sound_scape.backend.Models import whisper_specrnet, rawgat, xlsr, vocoder, CLAD

class ModelBindings:
    def __init__(self):
        self.whisper_model = whisper_specrnet()
        self.rawgat_model = rawgat()
        self.vocoder_model = vocoder()
        self.xlsr_model = xlsr()
        self.CLAD = CLAD()
        self.file_processing_queue = Queue()
        self.models = [self.whisper_model, self.rawgat_model, self.vocoder_model, self.xlsr_model, self.CLAD]
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
        """
        RESULT EX: 
        'whisper': {
            'unseparated_results': {
                'prediction': wpred,
                'label': wlabel,
            },
            'separated_results': {
                'prediction': wpred_sep,
                'label': wlabel_sep,
            }
        }
        """
        
        results_map = {}

        for model in self.models:
            pred, label = model.evaluate(original_path)
            pred_sep, label_sep = model.evaluate(sep_path)
            results_map[model.name] = {
                'unseparated_results': {
                    'prediction': pred,
                    'label': label,
                },
                'separated_results': {
                    'prediction': pred_sep,
                    'label': label_sep,
                }
            }
        

        return results_map


    def combined_results(self, results):
        # label = str "Real"/"Fake"
        # whichever num of real or fake is higher

        def most_label(res_class):
            count_real = 0
            count_fake = 0
            for model in results:
                if results[model][res_class]['label'] == 'Real':
                    count_real += 1
                else:
                    count_fake += 1
            return 'Real' if count_real > count_fake else 'Fake'
        return {
            'separated': {
                'prediction': sum([results[model]['separated_results']['prediction'] for model in results]) / len(results),
                'label': most_label('separated_results')
            },
            'unseparated': {
                'prediction': sum([results[model]['unseparated_results']['prediction'] for model in results]) / len(results),
                'label': most_label('unseparated_results')
            },
            'final': {
                'prediction': sum([results[model]['separated_results']['prediction'] for model in results]) / len(results),
                'label': most_label('separated_results')
            }
        }


    def explain_results(self, results_json):
        data = {
        "query": "We are using 5 models to predict the results of an audio file based on if it is a deepfake or authentic voice. We need you to explain to the user of our site the results of the model and what they mean. We are giving you the results and then want you to give a short explanation to the user as to why the answer is what it is based on the results. Here is the data: " + to_json(results_json)+ "Give a simple paragraph for the user explaining how the data results give the final outcome answer of real or fake. We need you to explain why our model came to this conclusion, do not give an indecisive answer.",
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
                # print(result)
                break
            else:
                print("results not arrived")
                time.sleep(2.5) # Wait for 5 seconds before checking the result again
        return result['data']['output']

    def identify_artist(self, file_path):
        # pass vocal to match
        artist = "(placeholder) Taylor Swift"
        return artist

    def process_file(self, id):
        if not self.file_ids.exists(id):
            self.file_ids.update_state(id, 'Does not exist')
            return
        # set to processing state
        self.file_ids.update_state(id, 'processing')
        # get path
        path = self.file_ids.get_path(id)

        # Separate the file
        self.file_ids.update_substate(id, 'separating')
        sep_file = separate_file(path, os.path.join(UPLOADS_FOLDER, "separated-uploads"), mp3=True)

        # get model eval results
        self.file_ids.update_substate(id, 'evaluating')
        result_json = {}
        result_json['model_results'] = self.get_model_results(path, sep_file)

        # combined results
        result_json['combined'] = self.combined_results(result_json['model_results'])


        # add ai explaination
        self.file_ids.update_substate(id, 'explaining')
        result_json['explaination'] = self.explain_results(result_json)

        # match the vocals to an artist
        self.file_ids.update_substate(id, 'identifying')
        result_json['artist_id'] = self.identify_artist(sep_file)


        self.file_ids.set_results(id, to_json(result_json))

class FileIds:
    def __init__(self):
        """
        structure:
        {
            <id>: {
                filename: <filename>,
                state: <uploaded/processing/finished>,
                [after processing] results: {
                    }
                }
        }
        """
        self.file_ids = {}

    def update_state(self, id, state):
        # overwrite status and clearing substate
        self.file_ids[id]['status'] = {
            'state': state
        }

    def update_substate(self, id, substate):
        self.file_ids[id]['status']['substate'] = substate

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
                'state': 'id not found'
            }
        return self.file_ids[id]['status']

    def has_results(self, id):
        return 'results' in self.file_ids[id]
    def get_results(self, id):
        if not self.has_results(id):
            return {
                "state": 'Results not ready'
            }
        return self.file_ids[id]['results']

    def exists(self, id):
        return id in self.file_ids.keys()