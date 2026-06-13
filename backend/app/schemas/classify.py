"""
app/schemas/classify.py — Request / response models for the classify endpoint.
"""

from pydantic import BaseModel, Field


class GenreProbabilities(BaseModel):
    bhajan:       float = Field(..., ge=0.0, le=1.0, description="Bhajan genre probability")
    deuda:        float = Field(..., ge=0.0, le=1.0, description="Deuda genre probability")
    lok_dohori:   float = Field(..., ge=0.0, le=1.0, description="Lok Dohori genre probability")
    newari:       float = Field(..., ge=0.0, le=1.0, description="Newari genre probability")
    tamang_selo:  float = Field(..., ge=0.0, le=1.0, description="Tamang Selo genre probability")
    tharu:        float = Field(..., ge=0.0, le=1.0, description="Tharu genre probability")


class ClassifyResponse(BaseModel):
    predicted_genre: str = Field(
        ...,
        description="Top predicted genre label",
        examples=["lok_dohori"],
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Probability of the predicted genre (0–1)",
        examples=[0.8732],
    )
    all_probabilities: GenreProbabilities = Field(
        ...,
        description="Softmax probability for every genre",
    )
    num_chunks_analysed: int = Field(
        ...,
        ge=1,
        description="Number of 30-second audio chunks processed",
    )
    filename: str = Field(..., description="Original uploaded filename")