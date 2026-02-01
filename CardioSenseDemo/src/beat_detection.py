import numpy as np
import neurokit2 as nk
import scipy.signal as signal


def preprocess_ecg(ecg_signal, fs):
    """
    Clean ECG using NeuroKit's validated pipeline
    """
    ecg_cleaned = nk.ecg_clean(ecg_signal, sampling_rate=fs, method="neurokit")
    return ecg_cleaned


def detect_r_peaks(ecg_signal, fs=360):
    """
    Robust R-peak detection for Normal / PAC / PVC
    Returns:
        r_peaks (np.array): indices of R-peaks
        ecg_cleaned (np.array): cleaned ECG
    """

    # 1. Clean ECG
    ecg_cleaned = preprocess_ecg(ecg_signal, fs)

    # 2. Detect R-peaks (Pan-Tompkins + adaptive thresholds)
    _, rpeaks = nk.ecg_peaks(
        ecg_cleaned,
        sampling_rate=fs,
        method="neurokit"
    )

    r_peaks = rpeaks["ECG_R_Peaks"]

    return r_peaks, ecg_cleaned
