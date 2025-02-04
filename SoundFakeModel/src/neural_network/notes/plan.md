# Steps

1. implement feature extraction
2. make neural network
3. Normalize features before input

# Model Architecture

# Core Features

- Root Mean Square (RMS) Energy Calculation
  - loudness and temporal envolope
- Cepstral Analysis
- Zero-Crossing Rate (ZCR) and Energy
- Spectral Analysis (FFT)
  - Pitch Tracking (using methods like autocorrelation or FFT)
- Wavelet Transform
- Energy Contour Analysis
- Silence Detection using Energy Thresholding
- Speech Rate Calculation
- Energy-Based Onset Detection
  - timings

## inputs

- model results from multiple models
  - 2 sets from each model sep and not sep
  - each set has label and conf
- features of the separated and unseparated audio file

## Feature Extraction Goals

- audio quality => general performance but also vocal isolation artificating
- volume => vocal isolation distinguishing
- music/singing features
  - a feature connecting to the idea of "flow" tempo may be more generalizable to music and speech
  - pitch BPM tone and other actual singing and music specific features would likely improve the accuracy for those files
- General possibilities
  - speech related
  - sequential-time

_needs to be low to moderate at max computation_

## Feature Ideas

music - Chroma Features => pitch related, would prob be rlly good for music - Melody Contour - Vibrato Rate and Depth => could help with vocal effects?

1. Zero-Crossing Rate
   - Differientiations in sounds, voiced vs unvoiced sounds, noise detection
2. Temporal Features (e.g., RMS Energy, Tempo)
   - sequential time, loudness, tempo _RMS Energy_
3. _Formant Frequencies_
   - singing and general speaking, Provides insights into vowel quality and articulation.
   - Resonant frequencies of the vocal tract that shape the sound of vowels and certain consonants.
4. Syllable Duration and Timing
   - The length of time each syllable is held during singing, as well as the timing of notes.
5. Harmonic-to-Noise Ratio (HNR)
   - The ratio of harmonics (periodic sound) to noise (aperiodic sound) in the singing voice.
   - Useful for assessing vocal quality and clarity
6. Temporal Envelope
   - sequential-time, loudness changes
7. Silence duration
8. Prosodic Features
   - Pitch Variation, Intonation
   - Variations in pitch, loudness, and duration that convey meaning and emotion in speech.
   - comututation medium to high
9. Rhythm and Timing Patterns
   - The regularity and patterns of speech sounds over time, including stress patterns.

### Formant Frequencies

1. Linear Predictive Coding (LPC)
   - vowel quality
2. Cepstral Analysis
   - distinguishing sounds, audio quality
3. Zero-Crossing Rate (ZCR) and Energy
   - speech vs silence, loudness
4. Spectral Analysis (FFT)
   - It helps the network learn the overall frequency profile of the audio, which can be used to classify different types of sounds (e.g., musical notes, speech, environmental sounds).
5. Wavelet-Transform
   - It helps the network recognize patterns that vary in both time and frequency, which is useful for classifying complex sounds like music or speech with varying intonations.

## Grouping

The model should have some hint that connect

sep audio features to sep model results
unsep audio features to unsep model results

_pairwise input representatioon_

_multi-input_

- layer for sep data
- layer for unsep data

1. Define inputs for each state from both models
2. Define branches for each state
3. Merge branches for each state

## Output

- weights for each model including weights for sep vs unsep
- using those weights can calc a prediction

## Refinement

- Adding noise sometimes to audio before processing each file
- Distinguishing between singing and non singing
  - If train the weighting is trained on all singing then it will likely overfit to music over speaking
