from scipy.fft import fft
import numpy as np
import librosa
import torch

def extract_features(audio_file, rawgat, whisper_specrnet_model):
    """
    Returns Features:
    - model results as tensors
    - rms as tensor
    - magnitude_spectrum as tensor

    Parameters:
    - audio_file: Path to the audio file to be processed.
    - rawgat: An instance of the rawgat model.
    - whisper_specrnet_model: An instance of the whisper_specrnet model.
    """
    audio_data, _ = librosa.load(audio_file, sr=None)

    # Extract model features
    model_features = process_model_features(audio_file, rawgat, whisper_specrnet_model)

    # Calculate RMS
    rms = calculate_rms(audio_data)

    # Calculate FFT features
    fft_features = calculate_fft(audio_data)

    # Convert features to tensors
    rms_tensor = torch.tensor(rms, dtype=torch.float32)  # Convert RMS to tensor
    magnitude_spectrum_tensor = torch.tensor(fft_features["magnitude_spectrum"], dtype=torch.float32)  # Convert magnitude spectrum to tensor
    fft_result_tensor = torch.tensor(fft_features["fft_result"], dtype=torch.complex64)  # Convert FFT result to tensor

    audio_featrures = torch.cat([rms_tensor.unsqueeze(0), magnitude_spectrum_tensor], dim=0)


    # Return structured features as tensors
    return model_features, audio_featrures


def process_model_features(audio_file, rawgat_model, whisper_specrnet_model):
    """
    Extract features from two models for a given audio file.

    Parameters:
    - audio_file: Path to the audio file to be processed.
    - rawgat_model: An instance of the rawgat model.
    - whisper_specrnet_model: An instance of the whisper_specrnet model.

    Returns:
    - features: A list of tuples containing the extracted features from both models.
    """
    rawgat_conf, rawgat_label = rawgat_model.evaluate(audio_file)
    whisper_conf, whisper_label = whisper_specrnet_model.evaluate(audio_file)
    
    # Map label prediction to negative/positive value
    rawgat_label = -1 if rawgat_label == "Fake" else 1
    whisper_label = -1 if whisper_label == "Fake" else 1

    # Create 2D tensors for each feature
    whisper_feature = torch.tensor([whisper_label * whisper_conf], dtype=torch.float32)
    rawgat_feature = torch.tensor([rawgat_conf * whisper_conf], dtype=torch.float32)

    # Concatenate along the second dimension (dim=1)
    features = torch.cat([whisper_feature, rawgat_feature], dim=0)

    return features

def calculate_rms(audio_data):
    """
    Calculate the Root Mean Square (RMS) energy of the audio signal.

    Returns:
    - rms: The RMS value of the audio signal.
    """
    rms = np.sqrt(np.mean(audio_data**2))
    return rms


def calculate_fft(audio_data):
    """
    Calculate the FFT of the audio signal and extract relevant features.

    Parameters:
    - audio_data: The audio signal as a NumPy array.

    Returns:
    - fft_features: A dictionary containing FFT features (e.g., magnitude spectrum).
    """
    # Perform FFT
    fft_result = fft(audio_data)
    # Get the magnitude spectrum
    magnitude_spectrum = np.abs(fft_result)

    # Return a dictionary with FFT features
    return {
        "magnitude_spectrum": magnitude_spectrum,
        "fft_result": fft_result  # You can include the raw FFT result if needed
    }
