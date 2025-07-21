import os
import librosa
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from scipy.stats import skew, kurtosis
import joblib
import time
import matplotlib.pyplot as plt
import seaborn as sns
import tempfile

def extract_features(file_path, n_mfcc=20):
    """
    Extract comprehensive audio features for deepfake detection
    
    Parameters:
        file_path (str): Path to the audio file
        n_mfcc (int): Number of MFCC coefficients to extract
        
    Returns:
        np.ndarray: Feature vector
    """
    try:
        # Load audio file using improved loader function
        y, sr = load_audio_file(file_path)
        
        # Ensure minimum length (pad if needed)
        min_duration = 3  # seconds
        min_samples = min_duration * sr
        if len(y) < min_samples:
            y = np.pad(y, (0, min_samples - len(y)))
            
        # Rest of your feature extraction code remains the same...
        # Trim silence for better feature extraction
        y, _ = librosa.effects.trim(y, top_db=30)
        
        # Extract MFCCs (mean and std)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        mfcc_mean = np.mean(mfcc, axis=1)
        mfcc_std = np.std(mfcc, axis=1)
        
        # Extract spectral features
        spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
        spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
        spectral_rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
        
        # Get means of spectral features
        spectral_centroid_mean = np.mean(spectral_centroid)
        spectral_bandwidth_mean = np.mean(spectral_bandwidth)
        spectral_rolloff_mean = np.mean(spectral_rolloff)
        
        # Extract temporal features
        zero_crossing_rate = librosa.feature.zero_crossing_rate(y)
        zcr_mean = np.mean(zero_crossing_rate)
        
        # Extract statistical features
        amplitude_skew = skew(y)
        amplitude_kurtosis = kurtosis(y)
        
        # Extract pitch related features (if possible)
        pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
        pitch_mean = np.mean(pitches[pitches > 0]) if np.any(pitches > 0) else 0
        
        # Combine all features
        features = np.concatenate([
            mfcc_mean, 
            mfcc_std,
            [spectral_centroid_mean, spectral_bandwidth_mean, spectral_rolloff_mean, 
             zcr_mean, amplitude_skew, amplitude_kurtosis, pitch_mean]
        ])
        
        return features
        
    except Exception as e:
        print(f"Error extracting features from {file_path}: {str(e)}")
        return None

def train_model():
    """Train the deepfake detection model with enhanced features"""
    print("Starting model training...")
    start_time = time.time()
    
    # Paths for real and fake audio
    real_path = "test/real"
    fake_path = "test/fake"
    
    # Check if directories exist
    if not os.path.exists(real_path) or not os.path.exists(fake_path):
        print(f"Error: Dataset directories not found. Make sure {real_path} and {fake_path} exist.")
        return
    
    # Get file counts
    real_files = os.listdir(real_path)
    fake_files = os.listdir(fake_path)
    
    print(f"Found {len(real_files)} real audio files and {len(fake_files)} fake audio files")
    
    X = []
    y = []
    
    # Process real audio files
    print("Processing real audio files...")
    for idx, file in enumerate(real_files):
        path = os.path.join(real_path, file)
        print(f"  Processing {idx+1}/{len(real_files)}: {file}")
        features = extract_features(path)
        
        if features is not None:
            X.append(features)
            y.append(0)  # Label 0 for real
    
    # Process fake audio files
    print("Processing fake audio files...")
    for idx, file in enumerate(fake_files):
        path = os.path.join(fake_path, file)
        print(f"  Processing {idx+1}/{len(fake_files)}: {file}")
        features = extract_features(path)
        
        if features is not None:
            X.append(features)
            y.append(1)  # Label 1 for fake
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Training with {len(X_train)} samples, testing with {len(X_test)} samples")
    
    # Train model with improved parameters
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    
    print("Training model...")
    clf.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print("\nModel Evaluation:")
    print(f"Accuracy: {accuracy * 100:.2f}%")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["Real", "Fake"]))
    
    # Feature importance visualization
    if len(clf.feature_importances_) > 0:
        plt.figure(figsize=(10, 6))
        
        # Get feature importances
        importances = clf.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # Plot the top 15 features
        top_n = min(15, len(importances))
        sns.barplot(x=importances[indices[:top_n]], y=[f"Feature {i}" for i in indices[:top_n]])
        plt.title('Feature Importances for Deepfake Detection')
        plt.tight_layout()
        plt.savefig('feature_importance.png')
        print("\nFeature importance chart saved as 'feature_importance.png'")
    
    # Save model
    joblib.dump(clf, 'model.pkl')
    print("\nModel saved as 'model.pkl'")
    
    total_time = time.time() - start_time
    print(f"\nTraining completed in {total_time:.2f} seconds")

def load_audio_file(file_path):
    """Enhanced audio loading with special handling for problematic files"""
    import os
    
    # Check if it's a WhatsApp audio file (common problem files)
    filename = os.path.basename(file_path).lower()
    if "whatsapp" in filename and file_path.lower().endswith(".mp3"):
        try:
            print("Using specialized loader for WhatsApp audio")
            return load_problematic_audio(file_path)
        except Exception as e:
            print(f"Specialized loader failed: {str(e)}")
            # Continue with regular attempts if specialized loader fails
    
    # Regular loading attempts
    try:
        # First attempt: standard loading 
        y, sr = librosa.load(file_path, sr=None)
        return y, sr
    except Exception as e1:
        print(f"Standard loading failed: {str(e1)}")
        try:
            # Second attempt with different parameters
            y, sr = librosa.load(file_path, sr=22050, mono=True, res_type='kaiser_fast')
            return y, sr
        except Exception as e2:
            print(f"Second loading attempt failed: {str(e2)}")
            try:
                # Third attempt - special handling for any file
                return load_problematic_audio(file_path)
            except Exception as e3:
                print(f"All loading attempts failed for {file_path}")
                print(f"Error details: {str(e3)}")
                raise
def load_problematic_audio(file_path):
 
        import os
        import numpy as np
    
        try:
        # Step 1: Use pydub to convert the file directly
            from pydub import AudioSegment
        
        # Create temporary WAV file
            temp_dir = tempfile.gettempdir()
            temp_wav = os.path.join(temp_dir, "converted_audio.wav")
        
            print(f"Converting problematic file: {file_path}")
        
        # Force format to mp3 for WhatsApp files
            if "whatsapp" in file_path.lower():
                 audio = AudioSegment.from_file(file_path, format="mp3")
            else:
                audio = AudioSegment.from_file(file_path)
            
        # Convert to standard WAV format
            audio = audio.set_channels(1)  # Convert to mono
            audio = audio.set_frame_rate(22050)  # Standard sample rate
            audio.export(temp_wav, format="wav")
            print(f"Successfully converted to WAV: {temp_wav}")
        
        # Step 2: Use soundfile (not librosa) to load the wav
            import soundfile as sf
            y, sr = sf.read(temp_wav)
        
        # Step 3: Clean up temp file
            try:
                os.remove(temp_wav)
            except:
                pass
            return y, sr
        
        except Exception as e:
            print(f"Error in specialized loader: {str(e)}")
            raise
if __name__ == "__main__":
    train_model()