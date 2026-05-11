import argparse
import os
import random
import sys
from pathlib import Path

import librosa
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if SRC_DIR.exists():
    sys.path.insert(0, str(SRC_DIR))

try:
    from utils import GENRES, SAMPLE_RATE, DURATION
    from preprocess import load_audio, chunk_audio
except Exception:
    GENRES = ["tamang_selo", "deuda", "bhajan", "newari", "tharu", "lok_dohori"]
    SAMPLE_RATE = 22050
    DURATION = 30

    def load_audio(filepath):
        audio, sr = librosa.load(filepath, sr=SAMPLE_RATE, mono=True)
        return audio, sr

    def chunk_audio(audio):
        samples_per_clip = int(SAMPLE_RATE * DURATION)
        total_samples = len(audio)
        chunks = []
        for start in range(0, total_samples, samples_per_clip):
            end = start + samples_per_clip
            chunk = audio[start:end]
            if len(chunk) < samples_per_clip:
                padding = np.zeros(samples_per_clip - len(chunk))
                chunk = np.concatenate((chunk, padding))
            chunks.append(chunk)
        return chunks


class FolkMusicClassifier(nn.Module):
    def __init__(self, num_classes=len(GENRES), pretrained=False):
        super().__init__()
        weights = models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = models.efficientnet_b0(weights=weights)
        num_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(num_features, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.backbone(x)


def audio_to_mel(audio, sr, n_mels=128, hop_length=512):
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_mels=n_mels,
        hop_length=hop_length,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-9)
    return mel_norm


def build_batch(chunks, sr, transform):
    tensors = []
    for chunk in chunks:
        mel_spec = audio_to_mel(chunk, sr)
        mel_tensor = torch.tensor(mel_spec, dtype=torch.float32)
        mel_tensor = mel_tensor.unsqueeze(0).repeat(3, 1, 1)
        if transform:
            mel_tensor = transform(mel_tensor)
        tensors.append(mel_tensor)
    return torch.stack(tensors, dim=0)


def pick_random_audio_file(root_dir):
    extensions = {".wav", ".mp3", ".flac", ".webm", ".m4a", ".mp4", ".ogg", ".opus"}
    candidates = []
    for base, _, files in os.walk(root_dir):
        for filename in files:
            if Path(filename).suffix.lower() in extensions:
                candidates.append(Path(base) / filename)
    if not candidates:
        raise FileNotFoundError(f"No audio files found under: {root_dir}")
    return random.choice(candidates)


def predict_song(
    file_path,
    weights_path,
    device="auto",
    top_k=3,
    min_confidence=0.5,
    min_margin=0.15,
    max_entropy=1.2,
    min_chunk_confidence=0.6,
    min_chunk_ratio=0.6,
    show_chunks=False,
):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    audio, sr = load_audio(str(file_path))
    chunks = chunk_audio(audio)

    transform = transforms.Compose([
        transforms.Resize((128, 256)),
    ])

    batch = build_batch(chunks, sr, transform).to(device)

    model = FolkMusicClassifier(num_classes=len(GENRES), pretrained=False).to(device)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()

    with torch.no_grad():
        logits = model(batch)
        probs = torch.softmax(logits, dim=1).cpu().numpy()

    avg_probs = probs.mean(axis=0)
    top_indices = np.argsort(avg_probs)[::-1][:top_k]
    best_idx = int(top_indices[0])
    best_conf = float(avg_probs[best_idx])
    second_conf = float(avg_probs[top_indices[1]]) if len(top_indices) > 1 else 0.0
    confidence_margin = best_conf - second_conf

    entropy = float(-np.sum(avg_probs * np.log(avg_probs + 1e-12)))
    chunk_confidence = probs.max(axis=1)
    chunk_ratio = float(np.mean(chunk_confidence >= min_chunk_confidence))

    accepted = (
        best_conf >= min_confidence
        and confidence_margin >= min_margin
        and entropy <= max_entropy
        and chunk_ratio >= min_chunk_ratio
    )

    print(f"File: {file_path}")
    print(f"Chunks: {len(chunks)}")
    if not accepted:
        print("Genre not found for this song")
        print(
            "Rejection stats: max_conf={:.2f}%, margin={:.2f}%, entropy={:.3f}, chunk_ratio={:.2f}".format(
                best_conf * 100.0,
                confidence_margin * 100.0,
                entropy,
                chunk_ratio,
            )
        )
    print("Top predictions:")
    for idx in top_indices:
        print(f"  {GENRES[idx]}: {avg_probs[idx] * 100:.2f}%")

    if show_chunks:
        for i, row in enumerate(probs, start=1):
            chunk_top = np.argmax(row)
            print(f"Chunk {i}: {GENRES[chunk_top]} ({row[chunk_top] * 100:.2f}%)")

    return {
        "file": str(file_path),
        "avg_probs": avg_probs,
        "predicted_genre": None if not accepted else GENRES[best_idx],
        "confidence": best_conf,
        "confidence_margin": confidence_margin,
        "entropy": entropy,
        "chunk_ratio": chunk_ratio,
        "accepted": accepted,
        "top_indices": top_indices,
    }


def main():
    parser = argparse.ArgumentParser(description="Classify an audio file with the CNN model.")
    parser.add_argument("--file", type=str, help="Path to an audio file to classify.")
    parser.add_argument(
        "--random-dir",
        type=str,
        help="Pick a random audio file from this directory (recursive).",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default=str(PROJECT_ROOT / "models" / "classifier" / "best_cnn_model.pth"),
        help="Path to the CNN weights file.",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Number of top classes to show.")
    parser.add_argument(
        "--min-confidence",
        type=float,
        default=0.5,
        help="Minimum confidence required to accept a genre prediction.",
    )
    parser.add_argument(
        "--min-margin",
        type=float,
        default=0.15,
        help="Minimum margin between top-1 and top-2 to accept a prediction.",
    )
    parser.add_argument(
        "--max-entropy",
        type=float,
        default=1.2,
        help="Maximum entropy allowed to accept a prediction.",
    )
    parser.add_argument(
        "--min-chunk-confidence",
        type=float,
        default=0.6,
        help="Minimum chunk confidence for chunk consistency check.",
    )
    parser.add_argument(
        "--min-chunk-ratio",
        type=float,
        default=0.6,
        help="Minimum fraction of chunks above min-chunk-confidence to accept.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run inference on.",
    )
    parser.add_argument(
        "--show-chunks",
        action="store_true",
        help="Print per-chunk predictions.",
    )

    args = parser.parse_args()

    if args.file:
        file_path = Path(args.file)
    else:
        base_dir = Path(args.random_dir) if args.random_dir else PROJECT_ROOT / "data" / "raw"
        file_path = pick_random_audio_file(base_dir)
        print(f"Random pick: {file_path}")

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    predict_song(
        file_path=file_path,
        weights_path=args.weights,
        device=args.device,
        top_k=args.top_k,
        min_confidence=args.min_confidence,
        min_margin=args.min_margin,
        max_entropy=args.max_entropy,
        min_chunk_confidence=args.min_chunk_confidence,
        min_chunk_ratio=args.min_chunk_ratio,
        show_chunks=args.show_chunks,
    )


if __name__ == "__main__":
    main()
