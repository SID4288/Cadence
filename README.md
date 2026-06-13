# Cadence — Nepali Music Genre Classifier

Classifies Nepali folk music into genres using an EfficientNet-BiLSTM deep learning model.

## Project Structure

```
Cadence/
├── backend/                   # FastAPI + PyTorch classification API
│   ├── app/
│   │   ├── api/routes/        # POST /api/v1/classify endpoint
│   │   ├── core/              # Config via .env
│   │   ├── ml/                # Model architecture & audio preprocessing
│   │   ├── schemas/           # Pydantic request/response models
│   │   └── main.py            # FastAPI app entrypoint
│   ├── models/classifier/     # Trained model weights (.pth)
│   ├── temp_uploads/          # Auto-cleaned upload directory
│   ├── requirements.txt
│   └── .env.example
├── frontend/                  # React + Vite UI
│   ├── src/
│   │   ├── components/        # Header, HeroSection, ResultCard, FileUploader, etc.
│   │   ├── lib/               # Utility functions
│   │   ├── App.jsx            # Main app logic
│   │   └── main.jsx           # Entry point
│   ├── package.json
│   └── vite.config.js
├── LICENSE
└── README.md
```

## Tech Stack

| Layer    | Technology                                                           |
| -------- | -------------------------------------------------------------------- |
| Backend  | FastAPI, PyTorch, Librosa, NumPy, Uvicorn                            |
| Frontend | React 19, Vite, Tailwind CSS 4, Lucide React, Radix UI               |

## Backend Setup

```bash
cd backend
python -m venv .venv
.venv\Scripts\activate      # Windows
# source .venv/bin/activate  # Linux/macOS
pip install -r requirements.txt
```

Place your trained model weights at `backend/models/classifier/best_genre_model.pth`, copy `.env.example` to `.env`, then start the server:

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

API docs are available at `http://localhost:8000/docs`.

## Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

The dev server runs at `http://localhost:5173` and proxies classification requests to the backend at `http://127.0.0.1:8000`.

## API

### `POST /api/v1/classify`

Upload an audio file (MP3/WAV/FLAC) and receive genre predictions.

```json
{
  "predicted_genre": "lok_dohori",
  "confidence": 0.8732,
  "all_probabilities": {
    "bhajan": 0.0321, "deuda": 0.0412, "lok_dohori": 0.8732,
    "newari": 0.0198, "tamang_selo": 0.0215, "tharu": 0.0122
  },
  "num_chunks_analysed": 3,
  "filename": "song.mp3"
}
```

### `GET /health`

Returns model status and device info.
