"""
app/ml/model.py
~~~~~~~~~~~~~~~
Houses:
  • the exact model architecture from predict_song.py (EfficientNet-BiLSTM)
  • ModelManager — a singleton that loads the .pth once at startup and
    exposes a thread-safe inference method used by the API route.
"""

import logging
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

from app.core.config import settings

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Genre index mapping  (must match training order)
# ─────────────────────────────────────────────────────────────────────────────

GENRES = ["bhajan", "deuda", "lok_dohori", "newari", "tamang_selo", "tharu"]
NUM_CLASSES = len(GENRES)
IDX_TO_GENRE = {i: g for i, g in enumerate(GENRES)}


# ─────────────────────────────────────────────────────────────────────────────
# Architecture
# ─────────────────────────────────────────────────────────────────────────────

class MixPool(nn.Module):
    def __init__(self, in_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim * 2, in_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg = x.mean(dim=1)
        mx  = x.max(dim=1).values
        return self.proj(torch.cat([avg, mx], dim=-1))


class EfficientNetBiLSTMClassifier(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        backbone       = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
        self.features  = backbone.features
        self.bilstm    = nn.LSTM(
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)          # (B, 1280, 4, T//32)
        x = x.mean(dim=2)             # (B, 1280, T//32)
        x = x.permute(0, 2, 1)        # (B, T//32, 1280)
        x, _ = self.bilstm(x)         # (B, T//32, 384)
        x = self.lstm_drop(x)
        x = self.pool(x)              # (B, 384)
        return self.classifier(x)     # (B, num_classes)


# ─────────────────────────────────────────────────────────────────────────────
# Singleton manager
# ─────────────────────────────────────────────────────────────────────────────

class ModelManager:
    """
    Loads the trained model once at application startup.
    Provides a single `.predict_from_chunks()` method used by the route.
    """

    def __init__(self):
        self._model: EfficientNetBiLSTMClassifier | None = None
        self._device: str = "cuda" if torch.cuda.is_available() else "cpu"

    # ── lifecycle ──────────────────────────────────────────────────────────

    def load(self) -> None:
        model_path = Path(settings.MODEL_PATH)
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model weights not found at '{model_path}'. "
                "Place best_genre_model.pth in models/classifier/ "
                "or set MODEL_PATH in your .env file."
            )
        logger.info("Loading model from %s on %s …", model_path, self._device)
        model = EfficientNetBiLSTMClassifier(NUM_CLASSES).to(self._device)
        model.load_state_dict(
            torch.load(model_path, map_location=self._device, weights_only=True)
        )
        model.eval()
        self._model = model
        logger.info("Model loaded successfully.")

    def unload(self) -> None:
        self._model = None
        if self._device == "cuda":
            torch.cuda.empty_cache()
        logger.info("Model unloaded.")

    # ── properties ─────────────────────────────────────────────────────────

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def device(self) -> str:
        return self._device

    # ── inference ──────────────────────────────────────────────────────────

    def predict_from_chunks(self, chunks: list) -> dict:
        """
        Run soft-voted inference over all 30-second chunks.

        Parameters
        ----------
        chunks : list of np.ndarray, shape (3, 128, T) each

        Returns
        -------
        dict with keys: predicted_genre, confidence, all_probs, num_chunks
        """
        import numpy as np

        if self._model is None:
            raise RuntimeError("Model is not loaded. Call ModelManager.load() first.")

        all_logits = []
        with torch.no_grad():
            for x in chunks:
                tensor = torch.tensor(x).unsqueeze(0).to(self._device)  # (1, 3, 128, T)
                logits = self._model(tensor)                              # (1, 6)
                all_logits.append(logits)

        avg_logits = torch.stack(all_logits).mean(dim=0)  # (1, 6)
        probs      = torch.softmax(avg_logits, dim=1)[0]  # (6,)
        pred_idx   = int(probs.argmax().item())

        return {
            "predicted_genre": IDX_TO_GENRE[pred_idx],
            "confidence":      round(float(probs[pred_idx].item()), 4),
            "all_probs": {
                IDX_TO_GENRE[i]: round(float(probs[i].item()), 4)
                for i in range(NUM_CLASSES)
            },
            "num_chunks": len(chunks),
        }


# Module-level singleton
model_manager = ModelManager()