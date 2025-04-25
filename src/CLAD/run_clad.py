import sys
import os
import json
from pydub import AudioSegment
from pydub.utils import which


# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from sound_scape.backend.CladModel import CladModel

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_clad.py <audio_path>")
        sys.exit(1)

    audio_path = sys.argv[1]
    print(f"🔍 Running CLAD model on: {audio_path}")

    try:
        # ✅ Load audio with pydub for universal format support (MP3, WAV, etc.)
        AudioSegment.converter = which("ffmpeg")
         #debug print
        print(f"FFMPEG Path Set To: {AudioSegment.converter}")
        audio = AudioSegment.from_file(audio_path)
        duration_sec = len(audio) / 1000.0  # pydub duration is in milliseconds

        # ✅ Fake logic: longer than 5 seconds is considered 'Real', shorter is 'Deepfake'
        score = duration_sec - 5  # Negative means Deepfake, positive means Real

        percentage_divisor = 100.0 * 2 # will be a score in 100s we want to convert to a reasonable decimal range
        percentage_cap = 0.99
        # certainty = min(abs(score / percentage_divisor), percentage_cap)
        certainty = score / percentage_divisor
        certainty = min(abs(certainty), percentage_cap)  # Cap the certainty to a maximum of 0.99
        label = "Fake" if score < 0 else "Real"

        result = {
            "status": "success",
            "score": score,
            "certainty": certainty,
            "label": label
        }

        print(json.dumps(result))

    except Exception as e:
        error_result = {"status": "error", "message": str(e)}
        print(json.dumps(error_result))
        sys.exit(1)
