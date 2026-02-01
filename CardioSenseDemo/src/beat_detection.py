import numpy as np
from scipy.signal import find_peaks

def detect_r_peaks(ecg, fs=360):
    """
    Proven R-peak detector copied from CardioSense (stable).
    Do not modify unless absolutely necessary.
    """
    ecg = ecg - np.mean(ecg)

    squared = ecg ** 2

    peaks, _ = find_peaks(
        squared,
        distance=int(0.3 * fs),          # ≥300 ms refractory period
        height=np.percentile(squared, 95)  # adaptive threshold
    )

    return peaks
