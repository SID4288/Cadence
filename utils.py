# new/utils.py — SOTA configuration and shared settings
import os
from pathlib import Path

# Project-wide settings
PROJECT_NAME = "Homie Minister SOTA"
GENRES = ["tamang_selo", "deuda", "bhajan", "newari", "tharu", "lok_dohori"]
SAMPLE_RATE = 22050
DURATION = 30  # seconds per clip

# Constant-Q Transform (CQT) Hyperparameters
# 12 bins per octave matching semitones, spanning 7 octaves = 84 bins
N_BINS = 84
BINS_PER_OCTAVE = 12
HOP_LENGTH = 512

# File paths
# The dataset lives alongside the source files in this `new/` directory,
# so resolve all paths relative to NEW_ROOT (the directory containing utils.py).
NEW_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = NEW_ROOT.parent

DATASET_DIR = NEW_ROOT / "dataset"
LEGACY_SPLITS_DIR = DATASET_DIR / "splits"
PROCESSED_CQT_DIR = DATASET_DIR / "processed_cqt"

# Ensure directories exist (relative to NEW_ROOT so the workflow works
# regardless of the caller's current working directory).
os.makedirs(str(NEW_ROOT / "models"), exist_ok=True)
os.makedirs(str(NEW_ROOT / "results"), exist_ok=True)
os.makedirs(str(NEW_ROOT / "dataset" / "processed_cqt" / "train"), exist_ok=True)
os.makedirs(str(NEW_ROOT / "dataset" / "processed_cqt" / "test"), exist_ok=True)

def print_settings():
    print(f"=== {PROJECT_NAME} ===")
    print(f"Genres: {GENRES}")
    print(f"Sample Rate: {SAMPLE_RATE} Hz")
    print(f"CQT Bins: {N_BINS} (bins/octave: {BINS_PER_OCTAVE})")
    print(f"CQT Hop Length: {HOP_LENGTH}")
    print(f"=========================")

if __name__ == "__main__":
    print_settings()
