"""
app/api/routes/classify.py
~~~~~~~~~~~~~~~~~~~~~~~~~~
POST /api/v1/classify

Accepts a multipart audio file upload, runs it through the genre classifier,
and returns a structured JSON response.
"""

import logging
import uuid
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile, status

from app.core.config import settings
from app.ml.model import model_manager
from app.ml.preprocessing import audio_to_chunks
from app.schemas.classify import ClassifyResponse, GenreProbabilities

logger = logging.getLogger(__name__)
router = APIRouter()

TEMP_DIR = Path(settings.TEMP_UPLOAD_DIR)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _validate_upload(file: UploadFile) -> None:
    """Raise HTTPException for unsupported extension or oversized file."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in settings.ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=(
                f"Unsupported file type '{suffix}'. "
                f"Allowed: {sorted(settings.ALLOWED_AUDIO_EXTENSIONS)}"
            ),
        )


async def _save_upload(file: UploadFile) -> Path:
    """Stream upload to a uniquely named temp file; return its path."""
    suffix     = Path(file.filename or "audio").suffix.lower() or ".mp3"
    temp_path  = TEMP_DIR / f"{uuid.uuid4().hex}{suffix}"

    max_bytes   = settings.MAX_FILE_SIZE_MB * 1024 * 1024
    bytes_read  = 0

    with temp_path.open("wb") as f:
        while chunk := await file.read(1024 * 256):   # 256 KB at a time
            bytes_read += len(chunk)
            if bytes_read > max_bytes:
                temp_path.unlink(missing_ok=True)
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail=(
                        f"File exceeds the {settings.MAX_FILE_SIZE_MB} MB limit."
                    ),
                )
            f.write(chunk)

    return temp_path


# ─────────────────────────────────────────────────────────────────────────────
# Endpoint
# ─────────────────────────────────────────────────────────────────────────────

@router.post(
    "/classify",
    response_model=ClassifyResponse,
    summary="Classify a Nepali folk music audio file",
    description=(
        "Upload an audio file (MP3, WAV, FLAC, OGG, M4A, AAC — up to "
        f"{settings.MAX_FILE_SIZE_MB} MB). "
        "The model splits it into 30-second chunks, runs soft-voted "
        "inference, and returns the predicted genre with probabilities."
    ),
)
async def classify_audio(
    file: UploadFile = File(..., description="Audio file to classify"),
) -> ClassifyResponse:
    if not model_manager.is_loaded:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not ready yet. Please retry in a moment.",
        )

    _validate_upload(file)

    temp_path: Path | None = None
    try:
        # 1. Persist upload to disk
        temp_path = await _save_upload(file)
        logger.info("Saved upload to %s (%s)", temp_path.name, file.filename)

        # 2. Preprocess audio → list of (3, 128, T) chunks
        chunks = audio_to_chunks(str(temp_path))
        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail=(
                    "Could not extract any audio from the uploaded file. "
                    "Ensure the file is a valid audio and at least 7.5 seconds long."
                ),
            )

        # 3. Inference
        result = model_manager.predict_from_chunks(chunks)

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Unexpected error during classification: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An unexpected error occurred during classification.",
        ) from exc
    finally:
        # Always clean up the temp file
        if temp_path and temp_path.exists():
            temp_path.unlink()

    return ClassifyResponse(
        predicted_genre=result["predicted_genre"],
        confidence=result["confidence"],
        all_probabilities=GenreProbabilities(**result["all_probs"]),
        num_chunks_analysed=result["num_chunks"],
        filename=file.filename or "unknown",
    )