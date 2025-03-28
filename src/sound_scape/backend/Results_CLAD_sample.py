from .clad_integration import run_clad

def process_all_models(audio_path):
    results = {}
    # Example other models - placeholder
    # results['RawGAT'] = run_rawgat(audio_path)
    # results['Whisper'] = run_whisper(audio_path)

    # CLAD model integration
    clad_result = run_clad(audio_path)
    results['CLAD'] = clad_result

    return results
