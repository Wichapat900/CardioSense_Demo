import os
import numpy as np

# =========================
# LOAD TRAIN NORMALIZATION
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")

TRAIN_MEAN = np.load(os.path.join(MODEL_DIR, "ecg_mean.npy"))
TRAIN_STD  = np.load(os.path.join(MODEL_DIR, "ecg_std.npy"))

# =========================
# NORMALIZATION (OPTION B)
# =========================
def normalize_beat(beat: np.ndarray) -> np.ndarray:
    """
    Normalize using TRAINING dataset statistics
    (must match training exactly)
    """
    return (beat - TRAIN_MEAN) / (TRAIN_STD + 1e-8)

# =========================
# BEAT EXTRACTION
# =========================
def extract_beats(ecg: np.ndarray, r_peaks, window=64):
    """
    Extract beats centered on R-peaks

    Returns:
        beats: shape (num_beats, 128)
    """
    beats = []

    for r in r_peaks:
        if r - window < 0 or r + window >= len(ecg):
            continue  # skip edge beats

        beat = ecg[r - window : r + window]
        beat = normalize_beat(beat)
        beats.append(beat)

    return np.array(beats)
