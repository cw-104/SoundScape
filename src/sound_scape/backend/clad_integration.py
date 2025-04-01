from .CladModel import CladModel
import logging
import json

def run_clad(audio_path):
    try:
        clad_model = CladModel()
        raw_output = clad_model.predict(audio_path)

        if not raw_output:
            logging.error("CLAD subprocess returned no output.")
            return [ 0, "Error"]

        # Extract only the JSON part (last non-empty line)
        json_str = [line for line in raw_output.strip().splitlines() if line][-1]
        result = json.loads(json_str)

        if result.get("status") == "success":
            return [result["certainty"], result["label"]]
        else:
            return [0, "Error"]

    except Exception as e:
        logging.exception(f"Failed running CLAD model: {e}")
        return [0, "Error"]
