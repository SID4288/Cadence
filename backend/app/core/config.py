"""
app/core/config.py — Application settings.
Edit values here or override via environment variables / .env file.
No external settings library required.
"""

import os
from pathlib import Path


class Settings:
    # ── Model ──────────────────────────────────────────────────────────────
    MODEL_PATH: str = os.getenv(
        "MODEL_PATH", "models/classifier/best_genre_model.pth"
    )

    # ── Upload limits ──────────────────────────────────────────────────────
    MAX_FILE_SIZE_MB: int = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
    ALLOWED_AUDIO_EXTENSIONS: set[str] = {
        ".mp3", ".wav", ".flac", ".ogg", ".m4a", ".aac"
    }

    # ── Temp directory (auto-cleaned after each request) ───────────────────
    TEMP_UPLOAD_DIR: str = os.getenv("TEMP_UPLOAD_DIR", "temp_uploads")

    # ── CORS ───────────────────────────────────────────────────────────────
    # Change to your frontend URL in production, e.g. ["https://myapp.com"]
    ALLOWED_ORIGINS: list[str] = ["*"]


settings = Settings()

# Ensure the temp upload directory exists at import time
Path(settings.TEMP_UPLOAD_DIR).mkdir(parents=True, exist_ok=True)