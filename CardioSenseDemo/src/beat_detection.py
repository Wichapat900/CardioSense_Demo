import numpy as np
from scipy.signal import butter, filtfilt, find_peaks

def bandpass_filter(ecg, fs, low=0.5, high=40):
    nyq = 0.5 * fs
    b, a = butter(3, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, ecg)

def detect_r_peaks(ecg, fs):
    """
    Robust R-peak detection that works for:
    - Normal beats
    - PAC
    - PVC (including inverted PVCs)
    """
    # 1. Filter ECG
    filtered = bandpass_filter(ecg, fs)

    # 2. Normalize
    filtered = (filtered - np.mean(filtered)) / np.std(filtered)

    # 3. Detect peaks on absolute signal (IMPORTANT)
    peaks, _ = find_peaks(
        np.abs(filtered),
        distance=int(0.3 * fs),      # minimum 300 ms between beats
        prominence=0.6               # controls sensitivity
    )

    return peaks
