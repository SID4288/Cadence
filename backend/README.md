# Nepali Music Genre Classifier — FastAPI Backend

## Project Structure

```
backend/
├── app/
│   ├── api/
│   │   └── routes/
│   │       └── classify.py        # POST /api/v1/classify
│   ├── core/
│   │   └── config.py              # Settings via .env
│   ├── ml/
│   │   ├── model.py               # Model architecture + singleton manager
│   │   └── preprocessing.py       # Audio → mel chunks pipeline
│   ├── schemas/
│   │   └── classify.py            # Pydantic request/response models
│   └── main.py                    # FastAPI app factory
├── models/
│   └── classifier/
│       └── best_genre_model.pth   # ← place your .pth file here
├── temp_uploads/                  # auto-cleaned after each request
├── .env.example
├── requirements.txt
└── README.md
```

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Place your model weights
cp /path/to/best_genre_model.pth models/classifier/best_genre_model.pth

# 4. Configure environment (optional — defaults are fine for local dev)
cp .env.example .env

# 5. Start the server
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

## API

### `POST /api/v1/classify`

Upload an audio file and receive genre predictions.

**Request** — `multipart/form-data`

| Field  | Type | Description                    |
|--------|------|--------------------------------|
| `file` | File | Audio file (MP3/WAV/FLAC/etc.) |

**Response** — `200 OK`

```json
{
  "predicted_genre": "lok_dohori",
  "confidence": 0.8732,
  "all_probabilities": {
    "bhajan":      0.0321,
    "deuda":       0.0412,
    "lok_dohori":  0.8732,
    "newari":      0.0198,
    "tamang_selo": 0.0215,
    "tharu":       0.0122
  },
  "num_chunks_analysed": 3,
  "filename": "song.mp3"
}
```

**Error responses**

| Status | Meaning                                      |
|--------|----------------------------------------------|
| 413    | File exceeds the size limit (default 50 MB)  |
| 415    | Unsupported audio format                     |
| 422    | Audio too short or unreadable                |
| 503    | Model not loaded yet (rare, retry in a moment) |

### `GET /health`

Returns `{ "status": "ok", "model_loaded": true, "device": "cpu" }`.

## Interactive Docs

Once running, visit:
- Swagger UI: http://localhost:8000/docs
- ReDoc:       http://localhost:8000/redoc
