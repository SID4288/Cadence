"""
predict_song.py  —  Nepali Music Genre Classifier
==================================================
Usage:
    python predict_song.py path/to/song.mp3
    python predict_song.py path/to/song.wav --model best_genre_model.pth
"""

import sys
import argparse

import numpy as np
import librosa
import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights


# ─────────────────────────────────────────────────────────────────────────────
# 1. CONSTANTS  (must match training exactly)
# ─────────────────────────────────────────────────────────────────────────────

GENRES = ["bhajan", "deuda", "lok_dohori", "newari", "tamang_selo", "tharu"]
NUM_CLASSES = len(GENRES)
IDX_TO_GENRE = {i: g for i, g in enumerate(GENRES)}

# Global normalisation stats computed from training data
GLOBAL_MEAN = 0.4551
GLOBAL_STD  = 0.1986

# Spectrogram / chunking parameters
SR          = 22050
N_MELS      = 128
HOP_LENGTH  = 512
N_FFT       = 2048
CHUNK_SECS  = 30                          # 30 s → 1292 frames at sr=22050
CHUNK_SAMP  = CHUNK_SECS * SR            # 661 500 samples

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ─────────────────────────────────────────────────────────────────────────────
# 2. MODEL  (exact copy of training architecture)
# ─────────────────────────────────────────────────────────────────────────────

class MixPool(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim * 2, in_dim)

    def forward(self, x):
        avg = x.mean(dim=1)
        mx  = x.max(dim=1).values
        return self.proj(torch.cat([avg, mx], dim=-1))


class EfficientNetBiLSTMClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        backbone      = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.features = backbone.features
        self.bilstm   = nn.LSTM(
            input_size=1280, hidden_size=192, num_layers=2,
            batch_first=True, bidirectional=True, dropout=0.4,
        )
        self.lstm_drop = nn.Dropout(0.4)
        self.pool      = MixPool(384)
        self.classifier = nn.Sequential(
            nn.Linear(384, 128),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)          # (B, 1280, 4, T//32)
        x = x.mean(dim=2)             # (B, 1280, T//32)
        x = x.permute(0, 2, 1)        # (B, T//32, 1280)
        x, _ = self.bilstm(x)         # (B, T//32, 384)
        x = self.lstm_drop(x)
        x = self.pool(x)              # (B, 384)
        return self.classifier(x)     # (B, num_classes)


# ─────────────────────────────────────────────────────────────────────────────
# 3. PREPROCESSING  (must mirror how the Kaggle .npy files were built)
# ─────────────────────────────────────────────────────────────────────────────

def audio_to_mel_npy(audio_segment: np.ndarray) -> np.ndarray:
    """
    Convert a raw audio segment (1-D float32 at SR=22050) to the same
    format as the pre-computed .npy files in the training dataset.

    Pipeline:
        power mel → dB (ref=max) → min-max normalise to [0, 1]

    The resulting mean≈0.45 / std≈0.20 matches GLOBAL_MEAN / GLOBAL_STD
    from the training logs, confirming this is correct.
    """
    mel = librosa.feature.melspectrogram(
        y=audio_segment, sr=SR, n_fft=N_FFT,
        hop_length=HOP_LENGTH, n_mels=N_MELS,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)            # (128, T)  range ~[-80, 0]
    # min-max normalise to [0, 1]  ← this is why GLOBAL_MEAN ≈ 0.45
    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    return mel_norm.astype(np.float32)                        # (128, T)


def mel_to_3channel(mel: np.ndarray) -> np.ndarray:
    """
    Exact replica of load_3channel() from the training notebook.
    Input : mel (128, T) — the .npy-equivalent values
    Output: (3, 128, T)  float32
    """
    d1 = librosa.feature.delta(mel, order=1)
    d2 = librosa.feature.delta(mel, order=2)

    mel_g = (mel - GLOBAL_MEAN) / (GLOBAL_STD + 1e-8)       # global norm

    def norm(arr):
        return (arr - arr.mean()) / (arr.std() + 1e-8)

    x = np.stack([norm(mel_g), norm(d1), norm(d2)], axis=0)  # (3, 128, T)
    return x.astype(np.float32)


def audio_to_chunks(audio_path: str) -> list[np.ndarray]:
    """
    Load audio, pad/trim to full 30-second chunks, return list of (3,128,T).
    Short songs (< 30 s) are zero-padded to one full chunk.
    """
    y, _ = librosa.load(audio_path, sr=SR, mono=True)

    # Build non-overlapping 30-second chunks
    chunks = []
    for start in range(0, max(len(y), CHUNK_SAMP), CHUNK_SAMP):
        segment = y[start : start + CHUNK_SAMP]
        if len(segment) < CHUNK_SAMP * 0.25:      # skip tiny tail (< 7.5 s)
            break
        # zero-pad the last chunk if shorter than 30 s
        if len(segment) < CHUNK_SAMP:
            segment = np.pad(segment, (0, CHUNK_SAMP - len(segment)))

        mel   = audio_to_mel_npy(segment)
        x     = mel_to_3channel(mel)
        chunks.append(x)

    return chunks


# ─────────────────────────────────────────────────────────────────────────────
# 4. INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

def load_model(model_path: str) -> EfficientNetBiLSTMClassifier:
    model = EfficientNetBiLSTMClassifier(NUM_CLASSES).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    model.eval()                       # ← disables dropout for stable output
    return model


def predict(audio_path: str, model: EfficientNetBiLSTMClassifier) -> dict:
    chunks = audio_to_chunks(audio_path)
    if not chunks:
        raise ValueError(f"Could not extract any audio chunks from {audio_path}")

    all_logits = []
    with torch.no_grad():
        for x in chunks:
            tensor = torch.tensor(x).unsqueeze(0).to(DEVICE)  # (1,3,128,T)
            logits = model(tensor)                              # (1, 6)
            all_logits.append(logits)

    # Soft voting: average logits across all chunks → one prediction per song
    avg_logits = torch.stack(all_logits).mean(dim=0)           # (1, 6)
    probs      = torch.softmax(avg_logits, dim=1)[0]           # (6,)
    pred_idx   = probs.argmax().item()

    return {
        "predicted_genre": IDX_TO_GENRE[pred_idx],
        "confidence":      probs[pred_idx].item(),
        "all_probs":       {IDX_TO_GENRE[i]: probs[i].item() for i in range(NUM_CLASSES)},
        "num_chunks":      len(chunks),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Predict Nepali music genre")
    parser.add_argument("audio",  help="Path to audio file (.mp3 / .wav / .flac …)")
    parser.add_argument(
    "--model",
    default="models/classifier/best_genre_model.pth",
    help="Path to model weights"
)
    args = parser.parse_args()

    print(f"\nLoading model from: {args.model}")
    model = load_model(args.model)

    print(f"Predicting:        {args.audio}")
    result = predict(args.audio, model)

    print(f"\n{'─'*40}")
    print(f"  Predicted genre : {result['predicted_genre'].upper()}")
    print(f"  Confidence      : {result['confidence']*100:.1f}%")
    print(f"  Chunks analysed : {result['num_chunks']}")
    print(f"\n  All probabilities:")
    for genre, prob in sorted(result["all_probs"].items(), key=lambda x: -x[1]):
        bar = "█" * int(prob * 30)
        print(f"    {genre:<15} {prob*100:5.1f}%  {bar}")
    print(f"{'─'*40}\n")


if __name__ == "__main__":
    main()