---

# SoundScape

---

Deepfake detection targetted at detecting deepfake songs and music of popular artists.

# Documentation

## Command Line Interface
How to use command line interface

```bash
py cli.py -h
ex:
py cli.py --sep --fuke path/to/audio
```

arg: what to put


# Open source projects utilized


## Denucs -- vocal isolation
Vocal isolation model to separate backing tracks

[demucs github](https://github.com/charzy/Demucs-v4-)


## Whisper-spectRnet -- primary model
The primary model we are using for deepfake detection

[whisper-specRnet Github](https://github.com/piotrkawa/deepfake-whisper-features/tree/main?tab=readme-ov-file)

[Supporting Paper](https://www.isca-archive.org/interspeech_2023/kawa23b_interspeech.pdf )

## RawGAT-ST -- backup model
Initial model adapated from RawGAT code


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
