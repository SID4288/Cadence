"""
app/ml/preprocessing.py
~~~~~~~~~~~~~~~~~~~~~~~~
Audio-to-tensor preprocessing — exact mirror of the pipeline used during
training. Any change here must also be reflected in the training notebook.
"""

import numpy as np
import librosa

# ─────────────────────────────────────────────────────────────────────────────
# Constants  (must match training exactly)
# ─────────────────────────────────────────────────────────────────────────────

SR          = 22050
N_MELS      = 128
HOP_LENGTH  = 512
N_FFT       = 2048
CHUNK_SECS  = 30
CHUNK_SAMP  = CHUNK_SECS * SR          # 661 500 samples

GLOBAL_MEAN = 0.4551
GLOBAL_STD  = 0.1986


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _audio_to_mel_npy(audio_segment: np.ndarray) -> np.ndarray:
    """
    power mel → dB (ref=max) → min-max normalise to [0, 1]
    Mirrors the pre-computed .npy format from the Kaggle dataset.
    """
    mel = librosa.feature.melspectrogram(
        y=audio_segment, sr=SR, n_fft=N_FFT,
        hop_length=HOP_LENGTH, n_mels=N_MELS,
    )
    mel_db   = librosa.power_to_db(mel, ref=np.max)
    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)
    return mel_norm.astype(np.float32)                        # (128, T)


def _mel_to_3channel(mel: np.ndarray) -> np.ndarray:
    """
    Exact replica of load_3channel() from the training notebook.
    Returns (3, 128, T) float32.
    """
    d1   = librosa.feature.delta(mel, order=1)
    d2   = librosa.feature.delta(mel, order=2)
    mel_g = (mel - GLOBAL_MEAN) / (GLOBAL_STD + 1e-8)

    def _norm(arr: np.ndarray) -> np.ndarray:
        return (arr - arr.mean()) / (arr.std() + 1e-8)

    x = np.stack([_norm(mel_g), _norm(d1), _norm(d2)], axis=0)
    return x.astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def audio_to_chunks(audio_path: str) -> list[np.ndarray]:
    """
    Load audio file and return a list of 3-channel mel-spectrogram tensors,
    one per 30-second chunk.  Short files are zero-padded to one full chunk.

    Parameters
    ----------
    audio_path : str
        Path to any librosa-compatible audio file.

    Returns
    -------
    list of np.ndarray, each shape (3, 128, T)
    """
    y, _ = librosa.load(audio_path, sr=SR, mono=True)

    chunks: list[np.ndarray] = []
    for start in range(0, max(len(y), CHUNK_SAMP), CHUNK_SAMP):
        segment = y[start : start + CHUNK_SAMP]

        # Skip tiny tail fragments (< 7.5 s)
        if len(segment) < CHUNK_SAMP * 0.25:
            break

        # Zero-pad the last chunk if it's shorter than 30 s
        if len(segment) < CHUNK_SAMP:
            segment = np.pad(segment, (0, CHUNK_SAMP - len(segment)))

        mel = _audio_to_mel_npy(segment)
        x   = _mel_to_3channel(mel)
        chunks.append(x)

    return chunks