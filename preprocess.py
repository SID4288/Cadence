# new/preprocess.py — CQT & HPSS Advanced Feature Extraction
import os
import sys
from pathlib import Path
import numpy as np
import librosa
import soundfile as sf
from tqdm import tqdm

# Ensure parent directory is in path
sys.path.append(str(Path(__file__).resolve().parent))
from utils import (
    GENRES,
    SAMPLE_RATE,
    DURATION,
    N_BINS,
    BINS_PER_OCTAVE,
    HOP_LENGTH,
    LEGACY_SPLITS_DIR,
    PROCESSED_CQT_DIR,
)

def compute_sota_features(audio_path_or_array, sr=SAMPLE_RATE):
    """
    Computes SOTA audio representations using:
    1. Harmonic-Percussive Source Separation (HPSS)
    2. Constant-Q Transform (CQT) for both sources
    
    Returns a normalized 2-channel numpy array of shape (2, N_BINS, N_FRAMES).
    Channel 0: Harmonic CQT
    Channel 1: Percussive CQT
    """
    if isinstance(audio_path_or_array, (str, Path)):
        # Load audio with librosa
        audio, _ = librosa.load(str(audio_path_or_array), sr=sr, mono=True)
    else:
        audio = audio_path_or_array

    # 1. Harmonic-Percussive Source Separation
    harmonic, percussive = librosa.effects.hpss(audio)

    # 2. Compute Constant-Q Transform (CQT) for both
    # We use a try-except fallback in case the signal is too short or low-energy for CQT
    try:
        cqt_h = np.abs(librosa.cqt(
            harmonic,
            sr=sr,
            hop_length=HOP_LENGTH,
            n_bins=N_BINS,
            bins_per_octave=BINS_PER_OCTAVE
        ))
        cqt_p = np.abs(librosa.cqt(
            percussive,
            sr=sr,
            hop_length=HOP_LENGTH,
            n_bins=N_BINS,
            bins_per_octave=BINS_PER_OCTAVE
        ))
    except Exception as exc:
        # Fallback to Mel Spectrogram if CQT fails (extremely rare, for degenerate signals)
        mel_h = librosa.feature.melspectrogram(y=harmonic, sr=sr, n_mels=N_BINS, hop_length=HOP_LENGTH)
        mel_p = librosa.feature.melspectrogram(y=percussive, sr=sr, n_mels=N_BINS, hop_length=HOP_LENGTH)
        cqt_h = np.abs(mel_h)
        cqt_p = np.abs(mel_p)

    # 3. Convert to dB log scale
    cqt_h_db = librosa.amplitude_to_db(cqt_h, ref=np.max)
    cqt_p_db = librosa.amplitude_to_db(cqt_p, ref=np.max)

    # 4. Standard Normalize [0, 1] for stable neural network inputs
    def normalize(db_array):
        amin = db_array.min()
        amax = db_array.max()
        denominator = amax - amin
        if denominator < 1e-9:
            return np.zeros_like(db_array)
        return (db_array - amin) / denominator

    norm_h = normalize(cqt_h_db)
    norm_p = normalize(cqt_p_db)

    # Stack to create 2-channel representation (2, N_BINS, N_FRAMES)
    # CQT typically produces ~1292 frames for 30s audio with hop_length=512
    features = np.stack([norm_h, norm_p], axis=0)
    return features


def preprocess_all_dataset_splits():
    """
    Precomputes CQT features for all wav files in dataset/splits/ (train and test)
    and saves them in dataset/processed_cqt/ as .npy files.
    """
    splits = ["train", "test"]
    print("Starting precomputation of HPSS + CQT SOTA features...")
    
    for split in splits:
        input_split_dir = LEGACY_SPLITS_DIR / split
        output_split_dir = PROCESSED_CQT_DIR / split
        
        if not input_split_dir.exists():
            print(f"Directory not found: {input_split_dir}. Skipping split: {split}")
            continue
            
        print(f"\nProcessing '{split}' split...")
        
        for genre in GENRES:
            input_genre_dir = input_split_dir / genre
            output_genre_dir = output_split_dir / genre
            
            if not input_genre_dir.exists():
                continue
                
            os.makedirs(str(output_genre_dir), exist_ok=True)
            files = [f for f in os.listdir(input_genre_dir) if f.endswith(".wav")]
            print(f"  Genre: '{genre}' ({len(files)} files)")
            
            for file in tqdm(files, desc=f"    {genre}"):
                input_file_path = input_genre_dir / file
                output_file_name = file.replace(".wav", ".npy")
                output_file_path = output_genre_dir / output_file_name
                
                # Check if already processed to save time
                if output_file_path.exists():
                    continue
                    
                try:
                    features = compute_sota_features(input_file_path)
                    np.save(str(output_file_path), features)
                except Exception as exc:
                    print(f"  ✗ Error processing {file}: {exc}")

    print("\nPrecomputation successfully completed!")
    print(f"Processed SOTA dataset stored in: {PROCESSED_CQT_DIR}")


if __name__ == "__main__":
    preprocess_all_dataset_splits()
