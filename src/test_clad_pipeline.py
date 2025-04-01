import os
from sound_scape.backend.Results_CLAD_sample import process_all_models

def test_clad():
    test_audio_file = os.path.abspath("src/DrakeDeepfake.mp3")
  # Simulated input
    print(f"🚀 Running full pipeline test with: {test_audio_file}")
    results = process_all_models(test_audio_file)

    print("\n✅ Full Model Results:")
    for model, output in results.items():
        print(f"{model}: {output}")

if __name__ == "__main__":
    test_clad()
