import numpy as np
from scipy.signal import find_peaks

def detect_r_peaks(ecg, fs):
    """
    Robust R-peak detection for Normal + PAC + PVC.
    Inspired by Pan-Tompkins (simplified).
    """

    ecg = np.array(ecg)

    # 1. Remove baseline
    ecg = ecg - np.mean(ecg)

    # 2. Emphasize QRS (energy)
    ecg_energy = ecg ** 2

    # 3. Smooth (moving average ~120 ms)
    window_size = int(0.12 * fs)
    window_size = max(1, window_size)
    ecg_smooth = np.convolve(
        ecg_energy,
        np.ones(window_size) / window_size,
        mode="same"
    )

    # 4. Adaptive threshold
    threshold = 0.35 * np.max(ecg_smooth)

    # 5. Minimum distance (200 ms)
    min_distance = int(0.25 * fs)

    peaks, _ = find_peaks(
        ecg_smooth,
        height=threshold,
        distance=min_distance
    )

    return peaks
