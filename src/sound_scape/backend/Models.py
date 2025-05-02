import yaml, torch, json, logging

from .Evaluate import get_best_device, evaluate_nn
from sound_scape.backend.Evaluate import DeepfakeClassificationModel
from sound_scape.backend.Results import DfResultHandler
from Base_Path import get_path_relative_base
from sound_scape.backend.whisper_specrnet import WhisperSpecRNet, set_seed
from .xlsr_model import xlsr_model_eval
from .vocoder_eval import vocoder_model
from numba import config
from .CladModel import CladModel


logging.getLogger("numba").setLevel(logging.WARNING)


logging.basicConfig(level=logging.WARNING)


class vocoder:
    def __init__(self, device=None, name="vocoder", model_path=None):
        self.name = name
        self.device = device
        # if device is None:
        # self.device = get_best_device()
        # detect mps, use cpu as fallback if needed
        if not device:
            device = "mps" if torch.backends.mps.is_available() else get_best_device()
        self.device = device
        # debug check
        # print(f"Vocoder initializing on device: {self.device}")
        self.model_path = model_path
        if not model_path:
            self.model_path = get_path_relative_base(
                "pretrained_models/vocoder/librifake_pretrained_lambda0.5_epoch_25.pth"
            )
        self.yaml_path = get_path_relative_base(
            "pretrained_models/vocoder/model_config_RawNet.yaml"
        )
        self.model = vocoder_model(
            self.model_path, device=device, yaml_path=self.yaml_path
        )

    def evaluate(self, file_path):
        multi, binary = self.model.eval(file_path)

        label = None
        pred = None

        # TODO pick between multi and binary and whether should have 1 > 0 or 0 > 1
        # if(multi[0] > multi[1]):
        if binary[0] > binary[1]:
            # print("Real")
            label = "Real"
            pred = binary[0]
            # pred = multi[0]
        else:
            # print("Fake")
            label = "Fake"
            pred = binary[1]
            # pred = multi[1]

        # print('Multi classification result : gt:{}, wavegrad:{}, diffwave:{}, parallel wave gan:{}, wavernn:{}, wavenet:{}, melgan:{}'.format(multi[0], multi[1], multi[2], multi[3], multi[4], multi[5], multi[6]))
        # sum all the multi classification results
        # sum_multi = sum(multi)
        # print('Binary classification result : real:{}, fake:{}'.format(binary[0], binary[1]))

        # print(multi, binary)
        return pred, label

    # def raw_eval(self, file_path):
    #     multi, binary = self.model.eval(file_path)
    #     label = None
    #     pred = None
    #     # Flipped
    #     if(binary[0] > binary[1]):
    #         # print("Real")
    #         label = "Real"
    #         pred = binary[0]
    #         # pred = multi[0]
    #     else:
    #         # print("Fake")
    #         label = "Fake"
    #         pred = binary[1]
    #         # pred = multi[1]

    #     return multi, binary, label, pred


class xlsr:
    def __init__(self, device=None, name="xlsr", model_path=None):
        self.name = name
        self.device = device
        self.model_path = model_path
        if not device:
            self.device = get_best_device()
        self.model = xlsr_model_eval(device=self.device, path=model_path)

    def evaluate(self, file_path):
        pred = self.model.eval_file(file_path)[0]
        return abs(pred / 100) * 50, "Real" if pred > 0 else "Fake"

    def raw_eval(self, file_path):
        pred = self.model.eval_file(file_path)[0]
        return pred, "Real" if pred > 0 else "Fake"


class whisper_specrnet:
    def __init__(
        self,
        name="whisper_specrnet",
        device="",
        model_path="",
        config_path="",
        threshold=0.45,
        reval_threshold=0,
        no_sep_threshold=0,
    ):
        self.name = name
        self.device = device
        self.weights_path = model_path
        self.model_path = self.weights_path
        self.threshold = threshold
        self.reval_threshold = reval_threshold
        self.no_sep_threshold = no_sep_threshold

        if device == "":
            self.device = get_best_device()

        get_best_device()
        if config_path == "":
            self.config_path = get_path_relative_base(
                "pretrained_models/whisper_specrnet/config.yaml"
            )
        if model_path == "":
            self.weights_path = get_path_relative_base(
                "pretrained_models/whisper_specrnet/weights.pth"
            )
            self.model_path = self.weights_path

        self.config = yaml.safe_load(open(self.config_path, "r"))

        model_name, model_parameters = (
            self.config["model"]["name"],
            self.config["model"]["parameters"],
        )

        self.model = WhisperSpecRNet(
            input_channels=self.config.get("input_channels", 1),
            freeze_encoder=self.config.get("freeze_encoder", False),
            device=self.device,
        )

        self.model.load_state_dict(
            torch.load(self.weights_path, map_location=self.device)
        )

        self.seed = self.config["data"].get("seed", 42)
        set_seed(self.seed)

    def evaluate(self, file_path):
        # Evaluate a single file
        # LABEL 0 = FAKE 1 = REAL
        pred, label = evaluate_nn(
            model=self.model,
            model_config=self.config["model"],
            device=self.device,
            single_file=file_path,
        )
        return (
            pred,
            self._label_to_str(label),
        )

    def raw_eval(self, file_path):
        return self.evaluate(file_path)

    def _label_to_str(self, label):
        # LABEL 0 = FAKE 1 = REAL
        return "Fake" if label == 0 else "Real"


class rawgat:
    def __init__(
        self, result_handler=None, name="rawgat", model_path=None, device=None
    ):
        self.name = name
        self.model_path = model_path
        self.result_handler = result_handler
        if result_handler is None:
            self.result_handler = DfResultHandler(-3, "Fake", "Real", 10, 0.45)
        self.model = DeepfakeClassificationModel(
            result_handler=self.result_handler, model_path=model_path, device=device
        )

    def evaluate(self, file_path):
        res = self.model.evaluate_file(file_path)
        # TODO choose best shift value (if any)
        return res.raw_value, res.classification
        # return min(.95,res.percent_certainty), res.classification

    def raw_eval(self, file_path):
        res = self.model.evaluate_file(file_path)
        return res.raw_value, res.classification

    def raw_eval_multi(self, file_paths):
        results = self.model.evaluate_multi_files(file_paths, progress_bar=True)
        return results


class CLAD:
    def __init__(self, model_path=None, device=None, name="CLAD"):
        self.name = name
        self.model = CladModel(model_path=model_path)

    def evaluate(self, file_path, debug_print=False):
        try:
            raw_output = self.model.predict(file_path)

            if not raw_output:
                logging.error("CLAD subprocess returned no output.")
                return [0, "Error"]

            # Extract only the JSON part (last non-empty line)
            json_str = [line for line in raw_output.strip().splitlines() if line][-1]
            result = json.loads(json_str)

            if result.get("status") == "success":
                return [result["certainty"], result["label"]]
            else:
                return 0, "Error"

        except Exception as e:
            logging.exception(f"Failed running CLAD model: {e}")
            return 0, "Error"

    def raw_eval(self, file_path, debug_print=False):
        try:
            raw_output = self.model.predict(file_path)

            if not raw_output:
                logging.error("CLAD subprocess returned no output.")
                return [0, "Error"]

            # Extract only the JSON part (last non-empty line)
            json_str = [line for line in raw_output.strip().splitlines() if line][-1]
            result = json.loads(json_str)

            if result.get("status") == "success":
                return [result["certainty"], result["label"], result["score"]]
            else:
                return 0, "Error"

        except Exception as e:
            logging.exception(f"Failed running CLAD model: {e}")
            return 0, "Error"


if __name__ == "__main__":
    # print("Creating CLAD")
    print()
    print()
    clad = CLAD()
    path = "/Users/christiankilduff/Deepfake_Detection_Resources/SoundScape/src/DrakeDeepfake.mp3"
    # print("Evaluating CLAD")
    print(clad.evaluate(path))
    print(
        clad.evaluate(
            "/Users/christiankilduff/Deepfake_Detection_Resources/SoundScape/src/uploads/MarioDeepfake.mp3"
        )
    )
