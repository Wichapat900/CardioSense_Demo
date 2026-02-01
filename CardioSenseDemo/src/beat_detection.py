import numpy as np
from scipy.signal import find_peaks

def detect_r_peaks(ecg, fs):
    """
    Simple and robust R-peak detection.
    Works well for demo + MIT-BIH-like ECG.
    """

    ecg = ecg - np.mean(ecg)
    ecg = ecg / (np.std(ecg) + 1e-8)

    # Distance between heartbeats (≈ 200 ms)
    min_distance = int(0.2 * fs)

    # Peak detection
    peaks, _ = find_peaks(
        ecg,
        distance=min_distance,
        prominence=0.6
    )

    return peaks
