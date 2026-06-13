"""
main.py — FastAPI application entrypoint
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import classify
from app.core.config import settings
from app.ml.model import model_manager


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, release on shutdown."""
    model_manager.load()
    yield
    model_manager.unload()


app = FastAPI(
    title="Nepali Music Genre Classifier API",
    description="Classifies Nepali folk music into genres using an EfficientNet-BiLSTM model.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(classify.router, prefix="/api/v1", tags=["classify"])


@app.get("/health", tags=["health"])
async def health_check():
    return {
        "status": "ok",
        "model_loaded": model_manager.is_loaded,
        "device": model_manager.device,
    }