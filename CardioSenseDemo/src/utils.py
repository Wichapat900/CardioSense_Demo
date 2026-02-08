import os
import numpy as np

# ================= PATH SAFE =================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")

TRAIN_MEAN_PATH = os.path.join(MODEL_DIR, "train_mean.npy")
TRAIN_STD_PATH  = os.path.join(MODEL_DIR, "train_std.npy")

# Debug (optional but useful)
print("Loading mean from:", TRAIN_MEAN_PATH)
print("Loading std from :", TRAIN_STD_PATH)

TRAIN_MEAN = np.load(TRAIN_MEAN_PATH)
TRAIN_STD  = np.load(TRAIN_STD_PATH)

# ================= FUNCTIONS =================
def normalize_beat(beat):
    return (beat - TRAIN_MEAN) / (TRAIN_STD + 1e-8)


def extract_beats(signal, r_peaks, window=128):
    half = window // 2
    beats = []

    for r in r_peaks:
        if r - half >= 0 and r + half < len(signal):
            beat = signal[r - half : r + half]
            beats.append(beat)

    return np.array(beats)
