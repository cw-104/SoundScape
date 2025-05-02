from sound_scape.backend.Isolate import separate_file
import threading, os
from queue import Queue
from json import dumps as to_json
from Base_Path import get_path_relative_base
UPLOADS_FOLDER = get_path_relative_base("uploads")
import requests
import time
from sound_scape.identify.identification import identifier
import concurrent.futures

AIXPLAIN_API_KEY = "305948bf8f6fb9c3a45097960725ccfb46bc27608372934ebccaaef3894807f0"
# AGENT_ID = "67acb51f56173fdefab4fc62"


AGENT_ID = "67911beb6217a7d8cf00b496"
POST_URL = f"https://platform-api.aixplain.com/sdk/agents/{AGENT_ID}/run"

headers = {
	"x-api-key": AIXPLAIN_API_KEY,
	"Content-Type": 'application/json'
}


from sound_scape.backend.Models import whisper_specrnet, rawgat, xlsr, vocoder, CLAD

class ModelBindings:
    def __init__(self):
        self.vocoder_iso = vocoder(model_path=get_path_relative_base(os.path.join("trained_models", "iso_vocoder.pth")))
        self.vocoder_og = vocoder(model_path=get_path_relative_base(os.path.join("trained_models", "og_vocoder.pth")))
        self.CLAD = CLAD(model_path=get_path_relative_base(os.path.join("trained_models", "clad.pth")))
        self.xlsr = xlsr(model_path=get_path_relative_base(os.path.join("trained_models", "xlsr.pth")))
        self.rawgat_og = rawgat(model_path=get_path_relative_base(os.path.join("trained_models", "og_rawgat.pth")))
        self.rawgat_iso = rawgat(model_path=get_path_relative_base(os.path.join("trained_models", "iso_rawgat.pth")))
        self.whisper_iso = whisper_specrnet(model_path=get_path_relative_base(os.path.join("trained_models", "iso_whisper.pth")))
        self.whisper_og = whisper_specrnet(model_path=get_path_relative_base(os.path.join("trained_models", "og_whisper.pth")))


        self.file_processing_queue = Queue()
        self.iso_models = [self.vocoder_iso, self.rawgat_iso, self.xlsr, self.whisper_iso, self.CLAD]
        self.og_models = [self.vocoder_og, self.rawgat_og, self.whisper_og, self.xlsr, self.CLAD]
        self.processing_thread = None

        self.file_ids = FileIds()

        self.identifier = identifier()

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

        
        def process_real_label(model_name, pred, label, iso):
            """
            We make the bar for a model to classify as fake higher, this means that for a model to classify fake
            we are more confident it is correct.

            We want to reduce the number of false positives, so we make models lean towards real and only classify fake
            with high confidence.

            Returns true if the model result is real false if not
            """
            # we want to prioritize getting reals correct and lower false negatives, so we to shift towards real based on cert, meaning we cut-off certain values that if it were to guess fake, we change to real

            if "whisper" in model_name.lower():  # if low whisper pred count as fake
                if iso:
                    pred *= 10 # iso lowers pred a lot
                if pred > 0.99 or pred < 0.01: # if conf extermely high or low result should be assumed real
                    return True, pred
                return label == "Real", pred
            elif (
                "clad" in model_name.lower()
            ):  # clad score tends to be > .5 real < .5 fake (CLAD cert is where we directly calc real fake not necessarily the outputted label, and it operates differently, high is real, low is fake)
                if pred > 0.5:
                    return True, pred
            elif "xlsr" in model_name.lower():  # if prediction is very low, count as Real
                if not iso:
                    pred *= 10 # iso lowers pred a lot more than og
                if pred < 0.9:
                    return True, pred
                return label == "Real", pred - .9 * 10

            elif "rawgat" in model_name.lower():
                # (rawgat pred is range - to + not restricted between -1 and 1 so we divide pred)
                if not iso:
                    return pred > 0, abs(pred) / 10 # pretty good rates at just > 0
                if iso:
                    # label for non-iso seems to be flipped and high certainty best and we make high bar for fake
                    return pred < 5, abs(pred - 5) / 10
                    
            elif "vocoder" in model_name.lower():
                if not iso:
                    return label == "Real", pred
                if iso:
                    if pred < .6:
                        return True, pred
                    return label == "Real", (pred - .6) * 2
            return False, pred


        
        results_map = {}
        votes_real = 0

        sum_real_pred = 0
        sum_fake_pred = 0

        def _evaluate_model_helper(model, path, iso):
            pred, label = model.evaluate(path)
            # Lean label real, reduce false positives
            is_real, adj_pred = process_real_label(model.name, pred, label, iso)

            return model.name, abs(max(0, min(.95, adj_pred))), "Real" if is_real else "Fake", iso
        
        # Use ThreadPoolExecutor to evaluate models concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = []
            for model in self.iso_models:
                futures.append(executor.submit(_evaluate_model_helper, model, sep_path, iso=True))
            for model in self.og_models:
                futures.append(executor.submit(_evaluate_model_helper, model, original_path, iso=False))

            for future in concurrent.futures.as_completed(futures):
                model_name, pred, label, iso = future.result()
                sub_key = "unseparated_results" if not iso else "separated_results"

                if label == "Real":
                    votes_real += 1 # vote for real classification
                    sum_real_pred += pred  # Update real predictions sum for avg
                else:
                    sum_fake_pred += pred  # Update fake predictions sum for avg

                # need to add result keys (and model keys) if not there
                if model_name not in results_map:
                    results_map[model_name] = {
                        'unseparated_results': {
                        },
                        'separated_results': {
                        }
                    }
                results_map[model_name][sub_key] = {
                    'prediction': pred,
                    'label': label,
                }
        
        label = "Fake"
        avg_pred = 0
        votes_fake = 10 - votes_real
        if votes_real > 7: # this is a high value for votes_real, but since we favor real so much a few votes goes a long way
            label = "Real"
            avg_pred = sum_real_pred / votes_real
        else:
            avg_pred = sum_fake_pred / (10 - votes_real)
        
        # return all model results and the combined prediction score and label
        return results_map, {
            "prediction": avg_pred,
            "label": label,
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
                # return "Out of credits"
        return result['data']['output']

    def identify_artist(self, file_path):
        # pass vocal to match
        # artist = "Taylor Swift | 76% match"

        name, score = self.identifier.identify_artist(file_path)
        return f"{name} | {int(score*100)}% match"

    def process_file(self, id):
        if not self.file_ids.exists(id):
            self.file_ids.update_state(id, 'Does not exist')
            return
        # set to processing state
        self.file_ids.update_state(id, 'processing')
        result_json = {}
        # get path
        path = self.file_ids.get_path(id)

        def identify(path):
            nonlocal result_json
            result_json['artist_id'] = self.identify_artist(path)
        # identification_thread = threading.Thread(target=self.identify_artist, args=(path,)).start()
        identification_thread = threading.Thread(target=identify, args=(path,))
        identification_thread.start()

        # Separate the file
        self.file_ids.update_substate(id, 'separating')
        sep_file = separate_file(path, os.path.join(UPLOADS_FOLDER, "separated-uploads"), mp3=True)
        print("\nseparated\n")
        # get model eval results
        self.file_ids.update_substate(id, 'evaluating')
        result_json['model_results'], result_json['combined'] = self.get_model_results(path, sep_file)
        print("\nevaluating\n")


        # add ai explaination
        self.file_ids.update_substate(id, 'explaining')
        result_json['explaination'] = self.explain_results(result_json)
        print("\nexplaining\n")

        # match the vocals to an artist
        self.file_ids.update_substate(id, 'identifying')
        # result_json['artist_id'] = self.identify_artist(sep_file)
        print("\nidentifying\n")

        # wait for identification thread to finish and get return
        identification_thread.join()
        # result_json['artist_id'] = identification_thread.result
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
        self.delete_mp3_file(id)
    
    def delete_mp3_file(self, id):
        print(f"I want to delete ", self.file_ids[id]['filename'])

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
