import os
import librosa
import numpy as np
import joblib
from scipy.stats import skew, kurtosis

# Load the trained model
model_path = 'model.pkl'
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model file {model_path} not found. Please run train.py first.")

loaded_model = joblib.load(model_path)

def extract_features(file_path, n_mfcc=20):
    """
    Extract audio features for deepfake detection with enhanced robustness.
    
    Parameters:
        file_path (str): Path to the audio file
        n_mfcc (int): Number of MFCC coefficients to extract
        
    Returns:
        np.ndarray: Feature vector
    """
    try:
        # Load audio with error handling
        y, sr = librosa.load(file_path, sr=None)
        
        # Ensure minimum length (pad if needed)
        min_duration = 3  # seconds
        min_samples = min_duration * sr
        if len(y) < min_samples:
            y = np.pad(y, (0, min_samples - len(y)))
        
        # Trim silence for better feature extraction
        y, _ = librosa.effects.trim(y, top_db=30)
        
        # Extract MFCCs (mean and std)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        
        # Extract spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_centroid_mean = np.mean(spectral_centroid)
        
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spectral_bandwidth_mean = np.mean(spectral_bandwidth)
        
        # Extract temporal features
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zero_crossing_rate)
        
        # Extract statistical features
        amplitude_skew = skew(y)
        amplitude_kurtosis = kurtosis(y)
        
        # Extract pitch related features
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
        
        # Combine all features
        base_features = np.concatenate([
            mfcc_mean, 
            mfcc_std,
            [spectral_centroid_mean, spectral_bandwidth_mean, zcr_mean, 
             amplitude_skew, amplitude_kurtosis, pitch_mean]
        ])
        
        # Adapt features to match model input expectations
        expected_features = loaded_model.n_features_in_
        
        if len(base_features) > expected_features:
            # Truncate features
            return base_features[:expected_features]
        elif len(base_features) < expected_features:
            # Pad with zeros
            return np.pad(base_features, (0, expected_features - len(base_features)))
        else:
            return base_features
        
    except Exception as e:
        raise RuntimeError(f"Feature extraction failed: {str(e)}")

def validate_audio_file(file_path):
    """Validate if the file is a valid audio file"""
    try:
        if not os.path.exists(file_path):
            return False, "File does not exist"
        
        # Try to load the file with librosa
        y, sr = librosa.load(file_path, sr=None, duration=1)  # Just load 1 second to check
        
        # Check if audio data was loaded
        if len(y) == 0:
            return False, "No audio data found in file"
            
        return True, "Valid audio file"
    except Exception as e:
        return False, f"Invalid audio file: {str(e)}"

def runtest(file_path):
    """
    Analyze an audio file to detect if it's a deepfake
    
    Parameters:
        file_path (str): Path to the audio file
        
    Returns:
        str: "Fake Audio" or "Real Audio"
    """
    # Validate file
    is_valid, message = validate_audio_file(file_path)
    if not is_valid:
        raise ValueError(message)
    
    # Extract features
    try:
        features = extract_features(file_path)
        
        # Debug information
        print(f"Extracted {len(features)} features for analysis")
        print(f"Model expects {loaded_model.n_features_in_} features")
        
        # Predict
        prediction = loaded_model.predict([features])
        probability = loaded_model.predict_proba([features])
        
        print(f"Prediction: {prediction[0]}")
        print(f"Confidence: {np.max(probability) * 100:.2f}%")
        
        if prediction[0] == 1:
            return "Fake Audio"
        else:
            return "Real Audio"
            
    except Exception as e:
        raise RuntimeError(f"Analysis failed: {str(e)}")