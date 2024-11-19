# SoundScape
Generative AI Music Detection

# Documentation
## SoundScapeModel

### Evaluate.py
### Class: `SoundScapeModel`

**How to Use**

```python
from Evaluate import SoundScapeModel

model = SoundScapeModel()
result = model.EvaluateFile(file_path)
```

---

#### `EvaluateFile(file_path)`
**Method**:  
Evaluates a single audio file for deepfake classification.

##### Parameters:
- **file_path** `{str}`: Path to the audio file.

##### Returns:
- **Result**:  
  - **classification** `{str}`:  
    "DF" (Deepfake) or "Authentic" (genuine audio).
  - **isDF** `{bool}`:  
    Indicates whether the audio file is a deepfake.
  - **certaintyClass** `{str}`:  
    Classification label of the audio file.
  - **rawcertainty** `{float}`:  
    Raw model evaluation results.
  - **shiftedcertainty** `{float}`:  
    Shifted evaluation results based on classification ranges.  
    *(See `Results.py` for more details.)*


## Command Line Interface
How to use command line interface



## RawGAT-ST
Initial model adapated from RawGAT code
`py file.py <-arg value>`

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
