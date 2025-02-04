1. Mel-Frequency Cepstral Coefficients (MFCCs)

- Computation moderate to intensive
  Pros:
  Captures the timbral aspects of audio.
  Robust to noise and variations in volume.

  Cons:
  May not capture temporal dynamics well. (not good for sequential or timing based inference)
  Sensitive to the choice of parameters (e.g., number of coefficients).

  Recommended Input Size: Use a window size of 20-40 ms with a hop size of 10-20 ms. For 3-minute audio files, you can extract around 300-600 MFCC frames.

  _When to Choose: Good for capturing speech characteristics and general audio patterns._

2. Spectrogram

   - Computation High !especially for large audio files

   Description: A visual representation of the spectrum of frequencies in a sound signal as they vary with time.

   Pros:

   - good for when sequential data or content over time is wanted
   - Useful for identifying patterns in complex audio signals.

   Cons:
   High dimensionality can lead to overfitting with small datasets.
   Requires significant preprocessing and normalization.

   Recommended Input Size: Use a window size of 20-40 ms with a hop size of 10-20 ms. For 3-minute audio files, you can generate a spectrogram with dimensions of around 300x300 (depending on parameters).

   _When to Choose: Useful for analyzing complex audio with varying frequency content, such as music and singing._

3. Chroma Features

   - computation moderation
   - !good for music (represents pitch classes)
     Pros:
     Effective for music and harmonic content analysis.
     Robust to changes in timbre and instrumentation.
     _if not a music file then less effective or even could make infrence worse_

   Description: Chroma features represent the energy distribution across the 12 different pitch classes (C, C#, D, etc.) and are useful for music analysis.

   Recommended Input Size: Similar to MFCCs, use a window size of 20-40 ms. For 3-minute audio files, you can extract around 300-600 chroma frames.
   _When to Choose: Best for music-related tasks and analyzing harmonic content._

4. Zero-Crossing Rate (ZCR)

   - computation low
     Pros:
     Useful for distinguishing between voiced and unvoiced sounds.

     Cons:
     May not be effective for complex audio.

     Description: The rate at which the audio signal changes from positive to negative or back, indicating the frequency content.

   Recommended Input Size: Can be calculated over short frames (e.g., 20 ms). For 3-minute audio files, you can extract around 900-1800 ZCR values.

   _When to Choose: Good for basic analysis of speech and distinguishing silence from sound._

5. Root Mean Square Energy (RMSE)

   - computation low
   - !volume and loudness
     con: limited for complex audio

   Description: A measure of the energy of the audio signal, indicating loudness.
   Recommended Input Size: Can be calculated over short frames (e.g., 20 ms). For 3-minute audio files, you can extract around 900-1800 RMSE values.
   _When to Choose: Useful for analyzing volume and loudness variations._

6. Spectral Centroid

   - computational moderate
     Pros:
     Useful for distinguishing different types of sounds.

     Cons:
     May not provide enough information for complex audio signals.
     Sensitive to noise and may require smoothing.

   Description: Indicates where the center of mass of the spectrum is located, often associated with the brightness of a sound.
   Provides insight into the perceived brightness of the audio.

   Recommended Input Size: Can be calculated over short frames (e.g., 20 ms). For 3-minute audio files, you can extract around 900-1800 spectral centroid values.

   _When to Choose: Useful for distinguishing between different types of sounds, especially in music and speech analysis._

7. Spectral Bandwidth

   - computation moderate
   - texture and sound source distinguishing

   - cons: sensitive to noise, no effective to all audio, requires careful param tuning

   Description: Measures the width of the spectrum, indicating the range of frequencies present in the audio signal.

   Recommended Input Size: Can be calculated over short frames (e.g., 20 ms). For 3-minute audio files, you can extract around 900-1800 spectral bandwidth values.
   W
   _When to Choose: Useful for analyzing the texture and richness of sounds, particularly in music._

8. Temporal Features (e.g., Tempo, Rhythm)

   - computation moderate to intensive
   - timing and rhythm
     Description: Features that capture the timing and rhythm of the audio, such as beats per minute (BPM).
   - not good for non musical data

   Pros:
   Important for music analysis and understanding rhythmic patterns.
   Can help in distinguishing between different genres of music.

   Cons:
   May not be relevant for non-musical audio.
   Sensitive to the quality of the audio and the presence of noise.

   Recommended Input Size: Depends on the method used for detection. For 3-minute audio files, you can summarize the tempo over the entire track.
   _When to Choose: Best for music-related tasks where rhythm and tempo are important._

9. Harmonic-to-Noise Ratio (HNR)

   - computation moderate to intensive
   - ratio of harmonic sound to noise, quality of sound
     pros:
     Useful for analyzing vocal quality and distinguishing between singing and speaking.
     Can help identify artifacts in audio.

     Cons:
     May not be effective for all types of audio.
     Sensitive to noise and may require careful parameter tuning.

     Description: Measures the ratio of harmonic sound to noise in the audio signal, indicating the quality of the sound.

   Recommended Input Size: Can be calculated over short frames (e.g., 20 ms). For 3-minute audio files, you can extract around 900-1800 HNR values.

   _When to Choose: Useful for analyzing vocal quality, especially in singing and speech._

Summary of Feature Selection

When selecting features for your model, consider the following:

    Volume/Silence: Use features like RMSE and Zero-Crossing Rate to capture loudness and silence.
    Singing: MFCCs, Chroma Features, and HNR are effective for analyzing vocal characteristics and singing.
    Background Music Tracks: Spectrograms and Chroma Features can help analyze harmonic content and musical structure.
    Vocals: MFCCs, HNR, and Spectral Centroid are useful for capturing vocal characteristics.
    Artifacting Due to Isolating: Features like HNR and Spectral Bandwidth can help identify artifacts and quality issues.
    Vocal Effects: MFCCs and Spectral Centroid can capture the timbral changes introduced by vocal effects.

Computational Complexity Considerations

Given that you have a limited dataset (900 samples) and 3-minute audio files, it's important to balance feature extraction complexity with the amount of data available. Here are some recommendations:

    Feature Dimensionality: Keep the number of features manageable to avoid overfitting. You might want to start with a subset of features and gradually add more based on model performance.
    Data Augmentation: Consider augmenting your dataset through techniques like pitch shifting, time stretching, or adding noise to increase the diversity of your training data.
    Batch Processing: Process audio files in batches to reduce computational load and improve efficiency.

4. Harmonic-to-Noise Ratio (HNR)

   Description: Measures the ratio of harmonic sound to noise in the audio signal.
   Robustness: HNR can be relatively stable in the presence of background noise, especially when analyzing vocal quality.
   Use Case: Effective for analyzing vocal quality and distinguishing between singing and speaking.

5. Spectral Flatness

   Description: Measures how flat or peaky the spectrum is, indicating the presence of noise versus tonal components.
   Robustness: Spectral flatness is less sensitive to noise and can provide insights into the tonal quality of the audio.
   Use Case: Useful for distinguishing between musical and non-musical sounds.
