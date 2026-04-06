import os 
import numpy as np
import pandas as pd
import sys
import librosa
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # Add parent directory
from utils import GENRES, SAMPLE_RATE, DURATION


def safe_scalar(value, default=0.0):
    scalar = float(np.mean(np.atleast_1d(value)))
    if not np.isfinite(scalar):
        return float(default)
    return scalar

def compute_mel_spectrogram(audio, sr = SAMPLE_RATE, n_mels=128, hop_length=512):
    mel = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels, hop_length=hop_length)
    mel_db = librosa.power_to_db(mel, ref=np.max)   
    return mel_db

def compute_mfcc(audio, sr = SAMPLE_RATE, n_mfcc=40):
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfcc_mean = np.mean(mfccs, axis=1)
    mfcc_std = np.std(mfccs, axis=1)
    mfcc_max = np.max(mfccs, axis=1)
    mfcc_min = np.min(mfccs, axis=1)

    mfcc_features = np.concatenate([mfcc_mean, mfcc_std, mfcc_max, mfcc_min])
    return mfcc_features

def compute_additional_features(audio, sr=SAMPLE_RATE):
    # Chroma — which of 12 musical pitches are dominant
    # Set tuning to 0.0 to avoid per-clip auto-tuning warnings on low-energy clips.
    chroma = librosa.feature.chroma_stft(y=audio, sr=sr, tuning=0.0)
    chroma_mean = np.mean(chroma, axis=1)   # shape: (12,)
    
    # Spectral centroid — "brightness" of the sound
    centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
    centroid_mean = safe_scalar(centroid)       # single number
    
    # Tempo — how fast is the music in BPM
    tempo, _ = librosa.beat.beat_track(y=audio, sr=sr)
    tempo_value = safe_scalar(tempo)
    
    # Zero crossing rate — how often signal crosses zero
    zcr = librosa.feature.zero_crossing_rate(audio)
    zcr_mean = safe_scalar(zcr)
    
    features = np.concatenate([
        chroma_mean,                    # 12 numbers
        [centroid_mean],                # 1 number
        [tempo_value],                  # 1 number
        [zcr_mean]                      # 1 number
    ])
    return features

def extract_all_features(audio, sr=SAMPLE_RATE):
    mfcc_features        = compute_mfcc(audio, sr)           # 160 numbers
    additional_features  = compute_additional_features(audio, sr)  # 15 numbers
    
    all_features = np.concatenate([mfcc_features, additional_features])
    return all_features  # 175 numbers total

def build_feature_dataset(processed_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    X = []  # features
    y = []  # labels
    files = []  # track filenames
    
    for label, genre in enumerate(GENRES):
        genre_dir = os.path.join(processed_dir, genre)
        
        if not os.path.exists(genre_dir):
            print(f"Skipping {genre} — folder not found")
            continue
        
        clips = [f for f in os.listdir(genre_dir) if f.endswith('.wav')]
        print(f"\nProcessing {genre} ({len(clips)} clips)...")
        
        for clip_name in clips:
            clip_path = os.path.join(genre_dir, clip_name)
            try:
                audio, sr = librosa.load(clip_path, sr=SAMPLE_RATE, mono=True)
                features = extract_all_features(audio, sr)
                X.append(features)
                y.append(label)
                files.append(clip_path)
            except Exception as e:
                print(f"  ✗ Error: {clip_name} — {e}")
        
        print(f"  ✓ Done: {genre}")
    
    X = np.array(X)
    y = np.array(y)
    
    np.save(os.path.join(output_dir, "X_features.npy"), X)
    np.save(os.path.join(output_dir, "y_labels.npy"), y)
    
    print(f"\n{'='*40}")
    print(f"Feature extraction complete!")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"Features per clip: {X.shape[1]}")
    print(f"Saved to: {output_dir}")

if __name__ == "__main__":
    build_feature_dataset(processed_dir="data/processed", output_dir="data/features")