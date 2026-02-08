import os
import numpy as np

# Path to src/
SRC_DIR = os.path.dirname(os.path.abspath(__file__))

# Go up to CardioSenseDemo/
BASE_DIR = os.path.dirname(SRC_DIR)

# models folder
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Load normalization stats (MATCH YOUR FILENAMES)
ECG_MEAN = np.load(os.path.join(MODEL_DIR, "ecg_mean.npy"))
ECG_STD  = np.load(os.path.join(MODEL_DIR, "ecg_std.npy"))

def normalize_beat(beat):
    return (beat - ECG_MEAN) / (ECG_STD + 1e-8)

def extract_beats(signal, r_peaks, window=128):
    half = window // 2
    beats = []
    for r in r_peaks:
        if r - half >= 0 and r + half < len(signal):
            beats.append(signal[r-half:r+half])
    return np.array(beats)
