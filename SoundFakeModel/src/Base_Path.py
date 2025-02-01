import os

def get_path_relative_base(path):
  """
  Converts a relative path to an absolute path, starting from the base layer of the project.
  """
  

  base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

 
  absolute_path = os.path.join(base_dir, path)

  return absolute_path


WHISPER_MODEL_WEIGHTS_PATH = get_path_relative_base("pretrained_models/whisper_specrnet/tiny_enc.en.pt")
MEL_FILTERS_PATH = get_path_relative_base("pretrained_models/whisper_specrnet/mel_filters.npz")
WHISPER_CONFIG_PATH = get_path_relative_base("pretrained_models/whisper_specrnet/config.yaml")

RAWGAT_CONFIG_PATH = get_path_relative_base("pretrained_models/RawGAT/model_config_RawGAT_ST.yaml")
RAWGAT_MODEL_WEIGHTS_PATH = get_path_relative_base("pretrained_models/RawGAT/RawGAT.pth")