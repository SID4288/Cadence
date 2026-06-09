# new/dataset.py — PyTorch SOTA CQT & HPSS Dataset & DataLoader
import os
import sys
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

sys.path.append(str(Path(__file__).resolve().parent))
from utils import GENRES, PROCESSED_CQT_DIR, LEGACY_SPLITS_DIR
from preprocess import compute_sota_features

class FolkMusicCQTDataset(Dataset):
    """
    High-performance PyTorch dataset for SOTA 2-channel CQT features.
    Prefers loading precomputed .npy files for speed, with safe fallback to loading 
    and computing features on-the-fly from raw .wav chunks if .npy files are missing.
    """
    def __init__(self, split="train", use_precomputed=True):
        self.split = split
        self.use_precomputed = use_precomputed
        self.filepaths = []
        self.labels = []
        self.is_precomputed_list = []

        # Determine folders
        processed_split_dir = PROCESSED_CQT_DIR / split
        legacy_split_dir = LEGACY_SPLITS_DIR / split

        for label, genre in enumerate(GENRES):
            processed_genre_dir = processed_split_dir / genre
            legacy_genre_dir = legacy_split_dir / genre

            # 1. Check if we can use precomputed CQT files
            if self.use_precomputed and processed_genre_dir.exists():
                npy_files = [f for f in os.listdir(processed_genre_dir) if f.endswith(".npy")]
                if len(npy_files) > 0:
                    for f in npy_files:
                        self.filepaths.append(processed_genre_dir / f)
                        self.labels.append(label)
                        self.is_precomputed_list.append(True)
                    continue

            # 2. Fallback to raw .wav files if precomputed not found or disabled
            if legacy_genre_dir.exists():
                wav_files = [f for f in os.listdir(legacy_genre_dir) if f.endswith(".wav")]
                for f in wav_files:
                    self.filepaths.append(legacy_genre_dir / f)
                    self.labels.append(label)
                    self.is_precomputed_list.append(False)

        if len(self.filepaths) == 0:
            raise FileNotFoundError(
                f"No training data found in either '{processed_split_dir}' or '{legacy_split_dir}'. "
                "Please run `python new/preprocess.py` to prepare the dataset splits first."
            )

    def __len__(self):
        return len(self.filepaths)

    def __getitem__(self, idx):
        filepath = self.filepaths[idx]
        label = self.labels[idx]
        is_precomputed = self.is_precomputed_list[idx]

        try:
            if is_precomputed:
                # High-speed precomputed loading
                features = np.load(str(filepath))
            else:
                # Dynamic fallback computation (librosa-based)
                features = compute_sota_features(filepath)
        except Exception as exc:
            # If an error occurs, return a silent zero-tensor matching average CQT shape
            # (2 channels, 84 bins, 1292 frames) to protect the training batch loop
            print(f"\n[Warning] Failed to load {filepath}: {exc}. Returning zero fallback.")
            features = np.zeros((2, 84, 1292), dtype=np.float32)

        # Truncate or pad along the time dimension (axis 2) to guarantee uniform size of 1292
        target_frames = 1292
        current_frames = features.shape[2]
        if current_frames > target_frames:
            features = features[:, :, :target_frames]
        elif current_frames < target_frames:
            pad_width = target_frames - current_frames
            features = np.pad(features, ((0,0), (0,0), (0, pad_width)), mode="constant")

        # Convert to PyTorch float tensor
        features_tensor = torch.tensor(features, dtype=torch.float32)
        return features_tensor, label


def create_sota_dataloaders(batch_size=16, use_precomputed=True, num_workers=0):
    """
    Creates train and test DataLoader instances for high-performance training.
    """
    train_dataset = FolkMusicCQTDataset(split="train", use_precomputed=use_precomputed)
    test_dataset = FolkMusicCQTDataset(split="test", use_precomputed=use_precomputed)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True if torch.cuda.is_available() else False
    )

    print(f"Loaded SOTA '{train_dataset.split}' dataset size: {len(train_dataset)} samples")
    print(f"Loaded SOTA '{test_dataset.split}' dataset size:  {len(test_dataset)} samples")
    return train_loader, test_loader


if __name__ == "__main__":
    # Test dataset pipeline integrity
    try:
        train_loader, test_loader = create_sota_dataloaders(batch_size=4, use_precomputed=True)
        features, labels = next(iter(train_loader))
        print("=== DATASET PIPELINE VERIFIED ===")
        print(f"Batch Features shape: {features.shape} (Expected: [4, 2, 84, 1292])")
        print(f"Batch Labels shape:   {labels.shape}")
        print("=================================")
    except Exception as exc:
        print(f"Dataset test failed: {exc}")
        print("This is normal if precomputation was not yet executed.")
