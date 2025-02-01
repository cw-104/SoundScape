import io, os, shutil, argparse
from pathlib import Path
import select
import subprocess as sp
import sys
from typing import Dict, Tuple, Optional, IO
from sound_scape.backend.Isolate import separate_file

from sound_scape.backend.Evaluate import get_best_device

def separate(in_path, out_path, model="htdemucs", mp3=False, mp3_rate=128, float32=False, int24=False, two_stems=None):
    files = find_files(in_path)
    for file in files:
        separate_file(file, out_path, model=model, mp3=mp3, mp3_rate=mp3_rate, float32=float32, int24=int24, two_stems=two_stems)

def find_files(in_path):
    out = []
    for file in Path(in_path).iterdir():
        if file.suffix.lower().lstrip(".") in extensions:
            out.append(file)
    return out

# args
parser = argparse.ArgumentParser(description="Separate audio files")
parser.add_argument(
    "--in_path", type=str, required=True, help="Path to the input audio files"
)
parser.add_argument(
    "--out_path", type=str, required=True, help="Path to the output directory",
)
args = parser.parse_args()

model = "htdemucs"
extensions = ["mp3", "wav", "ogg", "flac"]  # we will look for all those file types.
two_stems = "vocals"   # only separate one stems from the rest, for instance
# two_stems = "vocals"

# Options for the output audio.
mp3 = True
mp3_rate = 320
float32 = False  # output as float 32 wavs, unsused if 'mp3' is True.
int24 = False    # output as int24 wavs, unused if 'mp3' is True.
# You cannot set both `float32 = True` and `int24 = True` !!

in_path = os.path.abspath(args.in_path)
out_path = os.path.abspath(args.out_path)

separate(in_path,out_path, model=model, mp3=mp3, mp3_rate=mp3_rate, float32=float32, int24=int24, two_stems=two_stems)

