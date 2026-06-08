import os
import librosa #great for reading audio files and resampling
import numpy as np
import soundfile as sf #for writing audio files (librosa only reads, not writes)
import pandas as pd
import subprocess
import sys
import tempfile
from sklearn.model_selection import GroupShuffleSplit
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))  # Add parent directory to path
from utils import GENRES, SAMPLE_RATE, DURATION

FALLBACK_AUDIO_EXTENSIONS = {".webm", ".m4a", ".mp4", ".ogg", ".opus"}


def load_audio_with_ffmpeg(filepath):
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError(
            "Missing dependency: imageio-ffmpeg. Install it with `pip install imageio-ffmpeg`."
        ) from exc

    fd, temporary_wav = tempfile.mkstemp(suffix=".wav")
    os.close(fd)
    ffmpeg_executable = imageio_ffmpeg.get_ffmpeg_exe()

    try:
        subprocess.run(
            [
                ffmpeg_executable,
                "-y",
                "-i", filepath,
                "-ac", "1",
                "-ar", str(SAMPLE_RATE),
                temporary_wav,
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        audio, sr = librosa.load(temporary_wav, sr=SAMPLE_RATE, mono=True)
        return audio, sr
    finally:
        if os.path.exists(temporary_wav):
            os.remove(temporary_wav)

def load_audio(filepath):
    if not isinstance(filepath, str) or not filepath.strip():
        raise ValueError(f"Invalid filepath: {filepath}")

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filepath}")

    extension = os.path.splitext(filepath)[1].lower()

    if extension in FALLBACK_AUDIO_EXTENSIONS:
        try:
            return load_audio_with_ffmpeg(filepath)
        except Exception as fallback_error:
            raise RuntimeError(
                f"Failed to load '{filepath}' with ffmpeg fallback: {fallback_error}"
            ) from fallback_error

    try:
        audio, sr = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
        return audio, sr
    except Exception as original_error:
        raise RuntimeError(f"Failed to load '{filepath}': {original_error}") from original_error

def chunk_audio(audio):
    samples_per_clip = int(SAMPLE_RATE * DURATION)
    total_samples = len(audio)
    chunks = []
    
    for start in range(0, total_samples, samples_per_clip):
        end = start + samples_per_clip
        chunk = audio[start:end]
        
        # If the last chunk is shorter than the desired length, pad it with zeros
        if len(chunk) < samples_per_clip:
            padding = np.zeros(samples_per_clip - len(chunk))
            chunk = np.concatenate((chunk, padding))
        
        chunks.append(chunk)
    
    return chunks

def normalize_audio(audio):
    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val
    return audio

def clear_output_dir(output_dir):
    """Remove all .wav files from the output directory"""
    if os.path.exists(output_dir):
        for filename in os.listdir(output_dir):
            if filename.endswith('.wav'):
                filepath = os.path.join(output_dir, filename)
                try:
                    os.remove(filepath)
                except Exception as e:
                    print(f"Warning: Could not delete {filepath}: {e}")

def preprocess_and_save(filepath, output_dir):
    audio, sr = load_audio(filepath)
    chunks = chunk_audio(audio)
    
    base_name = os.path.basename(filepath)
    file_stem, _ = os.path.splitext(base_name)

    for i, chunk in enumerate(chunks):
        normalized_chunk = normalize_audio(chunk)
        filename = f'{file_stem}_chunk{i}.wav'
        output_path = os.path.join(output_dir, filename)
        sf.write(output_path, normalized_chunk, sr)

    return len(chunks)


def split_song_filepaths(filepaths, test_size=0.2, random_state=42):
    if len(filepaths) <= 1:
        return filepaths, []

    # Group by original song path to guarantee song-level split.
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    indices = list(range(len(filepaths)))
    train_idx, test_idx = next(splitter.split(indices, groups=filepaths))
    train_files = [filepaths[i] for i in train_idx]
    test_files = [filepaths[i] for i in test_idx]
    return train_files, test_files

if __name__ == "__main__":
    df = pd.read_csv("data/dataset.csv")
    processed_files = 0
    processed_chunks = 0
    skipped_files = 0
    train_song_count = 0
    test_song_count = 0

    split_root = "data/splits"
    train_root = os.path.join(split_root, "train")
    test_root = os.path.join(split_root, "test")
    
    for genre in GENRES:
        train_genre_dir = os.path.join(train_root, genre)
        test_genre_dir = os.path.join(test_root, genre)
        os.makedirs(train_genre_dir, exist_ok=True)
        os.makedirs(test_genre_dir, exist_ok=True)
        clear_output_dir(train_genre_dir)
        clear_output_dir(test_genre_dir)
        
        genre_files = [
            filepath for filepath in df[df["genre"] == genre]["filepath"].tolist()
            if isinstance(filepath, str) and filepath.strip()
        ]
        train_files, test_files = split_song_filepaths(genre_files)
        train_song_count += len(train_files)
        test_song_count += len(test_files)

        for filepath in train_files:
            try:
                chunk_count = preprocess_and_save(filepath, train_genre_dir)
                processed_files += 1
                processed_chunks += chunk_count
            except Exception as exc:
                skipped_files += 1
                print(f"Skipping file due to decode error: {filepath}\n  Reason: {exc}")

        for filepath in test_files:
            try:
                chunk_count = preprocess_and_save(filepath, test_genre_dir)
                processed_files += 1
                processed_chunks += chunk_count
            except Exception as exc:
                skipped_files += 1
                print(f"Skipping file due to decode error: {filepath}\n  Reason: {exc}")
    
    print("Preprocessing and splitting complete. Processed audio saved in 'data/splits/'.")
    print(f"Train songs: {train_song_count}")
    print(f"Test songs: {test_song_count}")
    print(f"Processed files: {processed_files}")
    print(f"Generated chunks: {processed_chunks}")
    print(f"Skipped files: {skipped_files}")