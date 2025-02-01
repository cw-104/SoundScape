from sound_scape.backend.Evaluate import get_best_device, evaluate_nn
from sound_scape.backend.Evaluate import DeepfakeClassificationModel
from sound_scape.backend.Results import DfResultHandler
from Base_Path import get_path_relative_base
import Base_Path
from sound_scape.backend.whisper_specrnet import WhisperSpecRNet, set_seed
import yaml, torch


class whisper_specrnet:
    def __init__(self, device=None, weights_path="", config_path="", threshold=.45, reval_threshold=0, no_sep_threshold=0):
        self.device = device
        self.weights_path = weights_path
        self.config_path = config_path
        self.threshold = threshold
        self.reval_threshold = reval_threshold
        self.no_sep_threshold = no_sep_threshold
        
        if not device:
            self.device = get_best_device()
            
        get_best_device()
        if config_path == "":
            self.config_path = Base_Path.WHISPER_CONFIG_PATH
        if weights_path == "":
            self.weights_path = Base_Path.WHISPER_MODEL_WEIGHTS_PATH
        
        self.config = yaml.safe_load(open(self.config_path, "r"))
        
        model_name, model_parameters = self.config["model"]["name"], self.config["model"]["parameters"]

        print(f"loading model...\n")
        self.model = WhisperSpecRNet(
            input_channels=self.config.get("input_channels", 1),
            freeze_encoder=self.config.get("freeze_encoder", False),
            device=self.device,
        )
        
        self.model.load_state_dict(torch.load(self.weights_path, map_location=self.device, weights_only=False))


        self.seed = self.config["data"].get("seed", 42)
        set_seed(self.seed)


        
    def evaluate(self, file_path):
        # Evaluate a single file
        # LABEL 0 = FAKE 1 = REAL
        pred, label = evaluate_nn(
            model=self.model,
            model_config=self.config["model"],
            device=self.device,
            single_file=file_path
        )
        return pred, self._label_to_str(label),
    

    def _label_to_str(self, label):
        # LABEL 0 = FAKE 1 = REAL
        return "Fake" if label == 0 else "Real"


class rawgat:
    def __init__(self, result_handler=None):
        self.result_handler = result_handler
        if result_handler is None:
            self.result_handler = DfResultHandler(-3, "Fake", "Real", 10, .45)
        self.model = DeepfakeClassificationModel(result_handler=self.result_handler)
        
    def evaluate(self, file_path):
        res = self.model.evaluate_file(file_path)
        return res.percent_certainty, res.classification
    
    def evaluate_full_results(self, file_path):
        return self.model.evaluate_file(file_path)