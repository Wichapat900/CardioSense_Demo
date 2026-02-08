import os
import numpy as np
import torch
import streamlit as st
import matplotlib.pyplot as plt

from model import ECG_CNN
from utils import extract_beats
from beat_detection import detect_r_peaks  # single source of truth

# ================= CONFIG =================
CLASSES = ["Normal", "PAC", "PVC"]

# Resolve paths safely (works locally + Streamlit Cloud)
SRC_DIR = os.path.dirname(os.path.abspath(__file__))          # CardioSenseDemo/src
BASE_DIR = os.path.dirname(SRC_DIR)                           # CardioSenseDemo
MODEL_DIR = os.path.join(BASE_DIR, "models")
DEMO_DIR = os.path.join(BASE_DIR, "demo_ecg")

MODEL_PATH = os.path.join(MODEL_DIR, "CardioSense_Model.pth")
TRAIN_MEAN_PATH = os.path.join(MODEL_DIR, "ecg_mean.npy")
TRAIN_STD_PATH  = os.path.join(MODEL_DIR, "ecg_std.npy")

DEMO_FILES = {
    "Normal": "normal_real.npy",
    "PAC": "pac_real.npy",
    "PVC": "pvc_real.npy",
}

FS = 360
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ================= LOAD NORMALIZATION =================
if not os.path.exists(TRAIN_MEAN_PATH):
    st.error(f"Missing file: {TRAIN_MEAN_PATH}")
    st.stop()

if not os.path.exists(TRAIN_STD_PATH):
    st.error(f"Missing file: {TRAIN_STD_PATH}")
    st.stop()

TRAIN_MEAN = np.load(TRAIN_MEAN_PATH)
TRAIN_STD = np.load(TRAIN_STD_PATH)

# ================= MODEL =================
@st.cache_resource
def load_model():
    model = ECG_CNN(num_classes=3)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# ================= ECG PLOT =================
def plot_ecg(signal, fs, r_peaks=None):
    t = np.arange(len(signal)) / fs
    fig, ax = plt.subplots(figsize=(12, 4))

    ax.plot(t, signal, color="black", linewidth=1.2)

    if r_peaks is not None and len(r_peaks) > 0:
        ax.scatter(
            r_peaks / fs,
            signal[r_peaks],
            color="red",
            s=25,
            zorder=5,
        )

    ax.set_xlim(t[0], t[-1])
    y_min, y_max = np.min(signal) - 0.2, np.max(signal) + 0.2
    ax.set_ylim(y_min, y_max)

    # ECG-style grid
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
    ecg_path = os.path.join(DEMO_DIR, DEMO_FILES[demo_type])

    if not os.path.exists(ecg_path):
        st.error(f"Missing demo ECG file: {ecg_path}")
        st.stop()

    ecg = np.load(ecg_path)
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

    preds = []
    confidences = []

    for beat in beats:
        # 🔑 training-consistent normalization
        beat = (beat - TRAIN_MEAN) / (TRAIN_STD + 1e-8)

        x = torch.tensor(beat, dtype=torch.float32)[None, None, :].to(DEVICE)

        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1).cpu().numpy()[0]

        preds.append(np.argmax(probs))
        confidences.append(np.max(probs))

    preds = np.array(preds)
    confidences = np.array(confidences)

    normal_count = np.sum(preds == 0)
    pac_count = np.sum(preds == 1)
    pvc_count = np.sum(preds == 2)
    abnormal_count = pac_count + pvc_count

    st.markdown("## 🩺 Final Result")

    if abnormal_count > 0:
        st.error("🚨 **Abnormal rhythm detected**")

        detected = []
        if pac_count > 0:
            detected.append(f"PAC ({pac_count} beats)")
        if pvc_count > 0:
            detected.append(f"PVC ({pvc_count} beats)")

        st.write("**Detected:** " + " and ".join(detected))
        st.caption("⚕️ This screening tool cannot diagnose medical conditions.")
    else:
        st.success("✅ **Normal rhythm detected**")

    st.markdown("### 📊 Model Confidence")
    st.write(f"**Average confidence:** {np.mean(confidences) * 100:.1f}%")
    st.write(
        f"**Range:** "
        f"{np.min(confidences) * 100:.1f}% – {np.max(confidences) * 100:.1f}%"
    )
