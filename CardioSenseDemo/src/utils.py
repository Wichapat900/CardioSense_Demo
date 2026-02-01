import numpy as np
from scipy.signal import find_peaks

def normalize_beat(beat: np.ndarray) -> np.ndarray:
    beat = beat - np.mean(beat)
    beat = beat / (np.std(beat) + 1e-8)
    return beat

def detect_r_peaks(ecg: np.ndarray, fs: int) -> np.ndarray:
    ecg_norm = ecg / (np.max(np.abs(ecg)) + 1e-8)
    min_distance = int(0.2 * fs)

    r_peaks, _ = find_peaks(
        ecg_norm,
        distance=min_distance,
        height=0.4
    )
    return r_peaks

def extract_beats(ecg: np.ndarray, r_peaks: np.ndarray, window: int = 64) -> np.ndarray:
    beats = []
    for r in r_peaks:
        if r - window >= 0 and r + window < len(ecg):
            beats.append(ecg[r - window : r + window])
    return np.array(beats)