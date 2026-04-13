import os
import numpy as np
import librosa
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils import GENRES, SAMPLE_RATE

class FolkMusicDataset(Dataset):
    def __init__(self, filepaths, labels, n_mels=128, hop_length=512):
        self.filepaths   = filepaths
        self.labels      = labels
        self.n_mels      = n_mels
        self.hop_length  = hop_length
    
    def __len__(self):
        return len(self.filepaths)
    
    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        label    = self.labels[idx]
        
        # Load audio
        audio, sr = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
        
        # Compute mel spectrogram
        mel = librosa.feature.melspectrogram(
            y=audio,
            sr=sr,
            n_mels=self.n_mels,
            hop_length=self.hop_length
        )
        mel_db = librosa.power_to_db(mel, ref=np.max)
        
        # Normalize to [0, 1]
        mel_min = mel_db.min()
        mel_max = mel_db.max()
        mel_norm = (mel_db - mel_min) / (mel_max - mel_min + 1e-9)
        
        # Convert to tensor with 3 channels (CNN expects RGB-like input)
        mel_tensor = torch.tensor(mel_norm, dtype=torch.float32)
        mel_tensor = mel_tensor.unsqueeze(0).repeat(3, 1, 1)
        
        return mel_tensor, label
    

def create_dataloaders(processed_dir, batch_size=32):
    filepaths = []
    labels    = []
    
    for label, genre in enumerate(GENRES):
        genre_dir = os.path.join(processed_dir, genre)
        if not os.path.exists(genre_dir):
            continue
        clips = [f for f in os.listdir(genre_dir) if f.endswith('.wav')]
        for clip in clips:
            filepaths.append(os.path.join(genre_dir, clip))
            labels.append(label)
    
    # Split 80/20
    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        filepaths, labels,
        test_size=0.2,
        random_state=42,
        stratify=labels
    )
    
    train_dataset = FolkMusicDataset(X_train, y_train)
    test_dataset  = FolkMusicDataset(X_test,  y_test)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Test samples:  {len(test_dataset)}")
    
    return train_loader, test_loader

if __name__ == "__main__":
    train_loader, test_loader = create_dataloaders("data/processed")
    
    # Peek at one batch
    spectrograms, labels = next(iter(train_loader))
    print(f"Batch shape: {spectrograms.shape}")
    print(f"Labels: {labels}")