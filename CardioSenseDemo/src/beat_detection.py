import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


def bandpass_filter(ecg, fs, low=3, high=20, order=2):
    nyq = 0.5 * fs
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return filtfilt(b, a, ecg)


def detect_r_peaks(ecg, fs):
    ecg = np.asarray(ecg)

    # 0. Remove first 0.5s (startup artifact)
    start = int(0.5 * fs)
    ecg = ecg[start:]

    # 1. Bandpass (PVC-safe)
    filtered = bandpass_filter(ecg, fs)

    # 2. Absolute signal (PVC can be inverted)
    energy = np.abs(filtered)

    # 3. Smooth energy
    win = int(0.12 * fs)
    energy_smooth = np.convolve(
        energy,
        np.ones(win) / win,
        mode="same"
    )

    # 4. Adaptive threshold
    threshold = 0.35 * np.max(energy_smooth)

    # 5. Candidate peaks
    candidates, _ = find_peaks(
        energy_smooth,
        height=threshold,
        distance=int(0.25 * fs)  # 240 bpm max
    )

    # 6. Refine: find true R in raw ECG
    r_peaks = []
    search = int(0.08 * fs)

    for c in candidates:
        left = max(0, c - search)
        right = min(len(ecg), c + search)
        r = np.argmax(ecg[left:right]) + left
        r_peaks.append(r)

    # 7. Restore indices
    r_peaks = np.array(r_peaks) + start

    return r_peaks
