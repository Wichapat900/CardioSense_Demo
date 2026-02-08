import os
import numpy as np

# Absolute path to utils.py
THIS_DIR = os.path.dirname(os.path.abspath(__file__))

# models folder must be NEXT TO utils.py
MODEL_DIR = os.path.join(THIS_DIR, "models")

TRAIN_MEAN = np.load(os.path.join(MODEL_DIR, "train_mean.npy"))
TRAIN_STD  = np.load(os.path.join(MODEL_DIR, "train_std.npy"))

def normalize_beat(beat):
    return (beat - TRAIN_MEAN) / (TRAIN_STD + 1e-8)

def extract_beats(signal, r_peaks, window=128):
    half = window // 2
    beats = []
    for r in r_peaks:
        if r - half >= 0 and r + half < len(signal):
            beats.append(signal[r-half:r+half])
    return np.array(beats)
