import os
import numpy as np
import librosa
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
def save_all_spectrograms(processed_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    total = 0
    
    for label, genre in enumerate(GENRES):
        genre_input_dir  = os.path.join(processed_dir, genre)
        genre_output_dir = os.path.join(output_dir, genre)
        os.makedirs(genre_output_dir, exist_ok=True)
        
        clips = [f for f in os.listdir(genre_input_dir) if f.endswith('.wav')]
        print(f"\nProcessing {genre} ({len(clips)} clips)...")
        
        for clip_name in clips:
            clip_path = os.path.join(genre_input_dir, clip_name)
            try:
                mel_norm = audio_to_spectrogram(clip_path)
                
                # Save as .npy with same name
                out_name = clip_name.replace('.wav', '.npy')
                out_path = os.path.join(genre_output_dir, out_name)
                np.save(out_path, mel_norm)
                total += 1
            except Exception as e:
                print(f"  ✗ Error: {clip_name} — {e}")
        
        print(f"  ✓ Done: {genre}")
    
    print(f"\nTotal spectrograms saved: {total}")
    print(f"Saved to: {output_dir}")

if __name__ == "__main__":
    save_all_spectrograms("data/processed", "data/spectrograms")