import os
import numpy as np
import torch
import streamlit as st
import matplotlib.pyplot as plt

from model import ECG_CNN
from utils import normalize_beat, extract_beats
from beat_detection import detect_r_peaks   # single source of truth

# ================= CONFIG =================
CLASSES = ["Normal", "PAC", "PVC"]

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "CardioSense_Model_3class.pth")
DEMO_DIR = os.path.join(BASE_DIR, "demo_ecg")

DEMO_FILES = {
    "Normal": "normal_real.npy",
    "PAC": "pac_real.npy",
    "PVC": "pvc_real.npy"
}

FS = 360
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================= MODEL =================
@st.cache_resource
def load_model():
    model = ECG_CNN(num_classes=3)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# ================= ECG PLOT WITH GRID =================
def plot_ecg(signal, fs, r_peaks=None):
    t = np.arange(len(signal)) / fs

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, signal, color="black", linewidth=1.2)

    if r_peaks is not None:
        ax.scatter(
            r_peaks / fs,
            signal[r_peaks],
            color="red",
            s=25,
            zorder=5
        )

    # ECG grid
    ax.set_xlim(t[0], t[-1])
    y_min, y_max = np.min(signal) - 0.2, np.max(signal) + 0.2
    ax.set_ylim(y_min, y_max)

    ax.set_xticks(np.arange(0, t[-1], 0.04), minor=True)
    ax.set_yticks(np.arange(y_min, y_max, 0.1), minor=True)

    ax.set_xticks(np.arange(0, t[-1], 0.2))
    ax.set_yticks(np.arange(y_min, y_max, 0.5))

    ax.grid(which="minor", color="#f4cccc", linewidth=0.5)
    ax.grid(which="major", color="#e06666", linewidth=0.8)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("mV")
    ax.set_title("ECG Signal")

    return fig

# ================= UI =================
st.set_page_config(page_title="CardioSense", layout="centered")
st.title("🫀 CardioSense")
st.write("ECG Screening Demo — **Normal vs Abnormal**")
st.markdown("---")

option = st.radio(
    "Choose ECG input:",
    ["Use demo ECG signal", "Upload ECG (.npy)"]
)

if option == "Use demo ECG signal":
    demo_type = st.selectbox("Select ECG type:", ["Normal", "PAC", "PVC"])
    ecg = np.load(os.path.join(DEMO_DIR, DEMO_FILES[demo_type]))
else:
    uploaded = st.file_uploader("Upload ECG (.npy)", type=["npy"])
    if uploaded is None:
        st.stop()
    ecg = np.load(uploaded)

# ================= R-PEAK DETECTION =================
r_peaks = detect_r_peaks(ecg, FS)

st.pyplot(plot_ecg(ecg, FS, r_peaks))
st.write(f"Detected **{len(r_peaks)} heartbeats**")

# ================= ANALYSIS =================
if st.button("🔍 Analyze ECG"):
    beats = extract_beats(ecg, r_peaks)

    if len(beats) == 0:
        st.error("No valid heartbeats detected.")
        st.stop()

    probs_all = []

    for beat in beats:
        beat = normalize_beat(beat)
        x = torch.tensor(beat, dtype=torch.float32)[None, None, :].to(DEVICE)

        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1).cpu().numpy()[0]

        probs_all.append(probs)

    probs_all = np.array(probs_all)

    # ================= CONFIDENCE LOGIC =================
    avg_probs = np.mean(probs_all, axis=0)

    normal_conf = avg_probs[0]
    abnormal_conf = avg_probs[1] + avg_probs[2]

    # ================= FINAL RESULT =================
    st.markdown("## 🩺 Final Result")

    if abnormal_conf >= 0.5:
        st.error("🚨 **Abnormal rhythm detected**")
    else:
        st.success("✅ **Normal rhythm detected**")

    # ================= CONFIDENCE DISPLAY =================
    st.markdown("### 📊 Model Confidence")
    st.write(f"**Normal:** {normal_conf * 100:.1f}%")
    st.write(f"**Abnormal:** {abnormal_conf * 100:.1f}%")
