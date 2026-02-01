import os
import numpy as np
import torch
import streamlit as st
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

from model import ECG_CNN
from utils import normalize_beat, extract_beats

# =====================
# CONFIG
# =====================
FS = 360
CLASSES = ["Normal", "PAC", "PVC"]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "CardioSense_Model_3class.pth")
DEMO_DIR = os.path.join(BASE_DIR, "demo_ecg")

DEMO_FILES = {
    "Normal": "normal_real.npy",
    "PAC": "pac_real.npy",
    "PVC": "pvc_real.npy"
}

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# =====================
# MODEL
# =====================
@st.cache_resource
def load_model():
    model = ECG_CNN(num_classes=3)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# =====================
# ROBUST R-PEAK DETECTION
# (detects inverted PVCs)
# =====================
def detect_r_peaks(signal, fs):
    signal = signal - np.mean(signal)

    # Normal R peaks
    peaks_pos, _ = find_peaks(
        signal,
        distance=0.25 * fs,
        prominence=0.6 * np.std(signal)
    )

    # Inverted R peaks (PVC)
    peaks_neg, _ = find_peaks(
        -signal,
        distance=0.25 * fs,
        prominence=0.6 * np.std(signal)
    )

    peaks = np.sort(np.concatenate([peaks_pos, peaks_neg]))
    return peaks

# =====================
# PLOT
# =====================
def plot_ecg(signal, fs, r_peaks):
    t = np.arange(len(signal)) / fs
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, signal, color="black")
    ax.scatter(r_peaks / fs, signal[r_peaks], color="red", s=25)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("mV")
    ax.set_title("ECG Signal")
    return fig

# =====================
# STREAMLIT UI
# =====================
st.set_page_config(page_title="CardioSense", layout="centered")
st.title("🫀 CardioSense")
st.write("AI-based ECG screening (Normal vs Abnormal)")
st.markdown("---")

option = st.radio("Choose ECG input:", ["Use demo ECG signal", "Upload ECG (.npy)"])

if option == "Use demo ECG signal":
    demo_type = st.selectbox("Select ECG type:", ["Normal", "PAC", "PVC"])
    ecg = np.load(os.path.join(DEMO_DIR, DEMO_FILES[demo_type]))
else:
    uploaded = st.file_uploader("Upload ECG (.npy)", type=["npy"])
    if uploaded is None:
        st.stop()
    ecg = np.load(uploaded)

# =====================
# DETECTION
# =====================
r_peaks = detect_r_peaks(ecg, FS)

st.pyplot(plot_ecg(ecg, FS, r_peaks))
st.write(f"Detected **{len(r_peaks)} heartbeats**")

# =====================
# ANALYSIS
# =====================
if st.button("🔍 Analyze ECG"):
    beats = extract_beats(ecg, r_peaks)
    beats = np.array([normalize_beat(b) for b in beats if len(b) == 256])

    preds = []
    probs_all = []

    for beat in beats:
        x = torch.tensor(beat, dtype=torch.float32)[None, None, :].to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1).cpu().numpy()[0]

        preds.append(np.argmax(probs))
        probs_all.append(probs)

    preds = np.array(preds)
    probs_all = np.array(probs_all)

    # =====================
    # NORMAL vs ABNORMAL LOGIC
    # =====================
    abnormal_mask = preds != 0
    abnormal_ratio = abnormal_mask.mean()

    normal_conf = np.mean(probs_all[:, 0])
    abnormal_conf = 1.0 - normal_conf

    st.markdown("## 🩺 Final Result")

    if abnormal_ratio > 0.05:
        st.error("🚨 Abnormal heartbeat detected")
    else:
        st.success("✅ Normal heartbeat detected")

    st.markdown("### 📊 Confidence")
    st.write(f"**Normal:** {normal_conf*100:.1f}%")
    st.write(f"**Abnormal:** {abnormal_conf*100:.1f}%")
