import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


def bandpass_filter(signal, fs, low=5, high=15, order=2):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, signal)


def detect_r_peaks(ecg, fs):
    """
    Robust R-peak detection for Normal / PAC / PVC
    """
    ecg = np.asarray(ecg)

    # 1. Bandpass filter (QRS emphasis)
    filtered = bandpass_filter(ecg, fs)

    # 2. Square the signal (Pan-Tompkins idea)
    squared = filtered ** 2

    # 3. Moving average (150 ms window)
    window_size = int(0.15 * fs)
    integrated = np.convolve(
        squared,
        np.ones(window_size) / window_size,
        mode="same"
    )

    # 4. Peak detection
    distance = int(0.3 * fs)  # minimum 300 ms between beats
    height = 0.3 * np.max(integrated)

    peaks, _ = find_peaks(
        integrated,
        distance=distance,
        height=height
    )

    return peaks
