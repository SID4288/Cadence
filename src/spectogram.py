import os
import numpy as np
import librosa
import shutil
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import GENRES, SAMPLE_RATE


def audio_to_spectrogram(filepath, n_mels=128, hop_length=512):
    audio, sr = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
    
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        hop_length=hop_length
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    # Normalize to [0, 1]
    mel_min = mel_db.min()
    mel_max = mel_db.max()
    mel_norm = (mel_db - mel_min) / (mel_max - mel_min + 1e-9)
    
    return mel_norm


def clear_npy_files(output_dir):
    if not os.path.exists(output_dir):
        return

    for root, _, files in os.walk(output_dir):
        for filename in files:
            if filename.endswith('.npy'):
                os.remove(os.path.join(root, filename))


def save_all_spectrograms(split_root_dir, output_dir):
    train_input_dir = os.path.join(split_root_dir, 'train')
    test_input_dir = os.path.join(split_root_dir, 'test')
    train_output_dir = os.path.join(output_dir, 'train')
    test_output_dir = os.path.join(output_dir, 'test')

    os.makedirs(train_output_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)
    clear_npy_files(train_output_dir)
    clear_npy_files(test_output_dir)
    
    total = 0
    
    for split_name, split_input_dir, split_output_dir in (
        ('train', train_input_dir, train_output_dir),
        ('test', test_input_dir, test_output_dir),
    ):
        print(f"\nProcessing {split_name} split...")

        for genre in GENRES:
            genre_input_dir = os.path.join(split_input_dir, genre)
            genre_output_dir = os.path.join(split_output_dir, genre)
            if not os.path.exists(genre_input_dir):
                continue

            os.makedirs(genre_output_dir, exist_ok=True)
            clips = [f for f in os.listdir(genre_input_dir) if f.endswith('.wav')]
            print(f"  {genre} ({len(clips)} clips)")

            for clip_name in clips:
                clip_path = os.path.join(genre_input_dir, clip_name)
                try:
                    mel_norm = audio_to_spectrogram(clip_path)

                    base_name = clip_name.replace('.wav', '')
                    out_name = f"{genre}_{base_name}.npy"
                    out_path = os.path.join(genre_output_dir, out_name)
                    np.save(out_path, mel_norm)
                    total += 1
                except Exception as e:
                    print(f"    ✗ Error: {clip_name} — {e}")
            
            print(f"    ✓ Done: {genre}")
    
    print(f"\nTotal spectrograms saved: {total}")
    print(f"Saved to: {output_dir}")


def validate_spectrogram_outputs(output_dir):
    shapes = set()
    missing_genre_names = []
    checked_files = 0

    for root, _, files in os.walk(output_dir):
        for filename in files:
            if not filename.endswith('.npy'):
                continue

            file_path = os.path.join(root, filename)
            array = np.load(file_path)
            shapes.add(array.shape)
            checked_files += 1

            if not any(genre in filename for genre in GENRES):
                missing_genre_names.append(file_path)

    print(f"\nValidated files: {checked_files}")
    print(f"Unique shapes found: {sorted(shapes)}")
    if missing_genre_names:
        print(f"Files missing a genre name in the filename: {len(missing_genre_names)}")
        for file_path in missing_genre_names[:10]:
            print(f"  - {file_path}")
    else:
        print("All filenames include a genre name.")

    return shapes, missing_genre_names

if __name__ == "__main__":
    save_all_spectrograms("data/splits", "data/spectrograms")
    validate_spectrogram_outputs("data/spectrograms")