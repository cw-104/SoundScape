---
# SoundScape
---

# How to run:

(Installation was tested on macos and linux ubuntu using python3.9, if you have a different OS additional steps or changing dependencies may be required)

## Requirements

- Install python3.9

1. clone the repo
2. cd SoundScape
3. create a venv for python3.9

```
python3.9 -m venv env
```

4. activate venv

```
source env/bin/activate
```

5. install dependencies

Per your operating system install
bzip2 libgmp-dev libblas-dev libffi-dev libgfortran5 libsqlite3-dev libz-dev libmpc-dev libmpfr-dev libncurses5-dev libopenblas-dev libssl-dev libreadline-dev tk-dev xz-utils

Macos

```
brew install bzip2 libgmp-dev libblas-dev libffi-dev libgfortran5 libsqlite3-dev libz-dev libmpc-dev libmpfr-dev libncurses5-dev libopenblas-dev libssl-dev libreadline-dev tk-dev xz-utils
```

Linux - apt

```
sudo apt install bzip2 libgmp-dev libblas-dev libffi-dev libgfortran5 libsqlite3-dev libz-dev libmpc-dev libmpfr-dev libncurses5-dev libopenblas-dev libssl-dev libreadline-dev tk-dev xz-utils libsox-dev
```

```
pip install -r requirements.txt
```

## Models

Most of the models are in the github, but there are a few you need to download

1. Trained XLSR
   https://drive.google.com/file/d/1PNF-lJgL2wRDSSXa6Hg2W_dyF0S7TLAn/view?usp=sharing
   This needs to be moved into src/trained_models

2. XLSR Wav2Vec dependency
   you need to download xlsr2_300m.pt which is from wav2vec for xlsr to work
   https://dl.fbaipublicfiles.com/fairseq/wav2vec/xlsr2_300m.pt

this needs to be put into src/pretrained_models/XLS-R/xlsr2_300m.pt

3. install fairsec

```
git clone https://github.com/pytorch/fairseq
pip install "pip<24.1"
pip install "omegaconf<2.1"
pip install fairseq/ && rm -rf fairseq
```

fairseq may require some dependency resolution, typically the above resolutions works, but may vary

If it says

```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
s3prl 0.4.17 requires omegaconf>=2.1.1, but you have omegaconf 2.0.6 which is incompatible.
```

Based on our testing, you should still be able to run with

```
py start_api.py
```

## Identification

you need to download the identification references from
https://drive.google.com/file/d/1Qy9iciryp_iKPAER2gn3uImkGo8t5Whm/view?usp=sharing

These should be in a folder
src/idenification_refrences
(inside the idenification_refrences/ folder should be folders of each artist name if you have an additional nested folder for some reason that must be removed)

## Run

API

```
cd src
py start_api.py
```

WEBSITE

```
cd src/site
npx http-server -p 8000
```

# Eval and Accuracy

Eval dataset non-isolated files located here
https://drive.google.com/file/d/17KnCrLA3uc2j8hMCDx0r5UcwxkTkOmLR/view?usp=sharing

```
Our accuracy after combining models
Real: 50/60 = 0.83
Fake: 33/54 = 0.61
Accuracy: 0.72
```

Deepfake detection targetted at detecting deepfake songs and music of popular artists.

# Open source projects utilized

## Demucs -- vocal isolation

Vocal isolation model to separate backing tracks

[demucs github](https://github.com/charzy/Demucs-v4-)

## Whisper-SpectRnet

[whisper-specRnet Github](https://github.com/piotrkawa/deepfake-whisper-features/tree/main?tab=readme-ov-file)

[Supporting Paper](https://www.isca-archive.org/interspeech_2023/kawa23b_interspeech.pdf)

## RawGAT-ST

[RawGAT-ST-code Github](https://github.com/eurecom-asp/RawGAT-ST-antispoofing)

@inproceedings{tak21_asvspoof,
author={Hemlata Tak and Jee-weon Jung and Jose Patino and Madhu Kamble and Massimiliano Todisco and Nicholas Evans},
title={{End-to-end spectro-temporal graph attention networks for speaker verification anti-spoofing and speech deepfake detection}},
year=2021,
booktitle={Proc. 2021 Edition of the Automatic Speaker Verification and Spoofing Countermeasures Challenge},
pages={1--8},
doi={10.21437/ASVSPOOF.2021-1}
}

@INPROCEEDINGS{Jung2021AASIST,
author={Jung, Jee-weon and Heo, Hee-Soo and Tak, Hemlata and Shim, Hye-jin and Chung, Joon Son and Lee, Bong-Jin and Yu, Ha-Jin and Evans, Nicholas},
booktitle={arXiv preprint arXiv:2110.01200},
title={AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks},
year={2021}

@INPROCEEDINGS{Tak2021End,
author={Tak, Hemlata and Patino, Jose and Todisco, Massimiliano and Nautsch, Andreas and Evans, Nicholas and Larcher, Anthony},
booktitle={Proc. ICASSP},
title={End-to-End anti-spoofing with RawNet2},
year={2021},
pages={6369-6373}
}
