import os
import numpy as np
import torch
import streamlit as st
import matplotlib.pyplot as plt

from model import ECG_CNN
from utils import normalize_beat, detect_r_peaks, extract_beats

# =========================
# Configuration
# =========================
CLASSES = ["Normal", "PAC", "PVC"]
FS = 360
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "models", "CardioSense_Model_3class.pth")
DEMO_DIR = os.path.join(BASE_DIR, "demo_ecg")

DEMO_FILES = {
    "Normal": "normal_real.npy",
    "PAC": "pac_real.npy",
    "PVC": "pvc_real.npy"
}

# =========================
# Load Model
# =========================
@st.cache_resource
def load_model():
    model = ECG_CNN(num_classes=3)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

# =========================
# Plot ECG
# =========================
def plot_ecg(signal, fs, r_peaks=None):
    t = np.arange(len(signal)) / fs
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, signal, color="black", linewidth=1)

    if r_peaks is not None and len(r_peaks) > 0:
        ax.scatter(
            r_peaks / fs,
            signal[r_peaks],
            color="red",
            s=25,
            zorder=3
        )

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("mV")
    ax.set_title("ECG Signal")
    ax.grid(alpha=0.2)
    return fig

# =========================
# Streamlit UI
# =========================
st.set_page_config(page_title="CardioSense", layout="centered")
st.title("🫀 CardioSense")
st.write("AI-based Heart Rhythm Screening")
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

# =========================
# Beat Detection
# =========================
r_peaks = detect_r_peaks(ecg, FS)

st.pyplot(plot_ecg(ecg, FS, r_peaks))
st.write(f"Detected **{len(r_peaks)} heartbeats**")

# =========================
# ECG Analysis
# =========================
if st.button("🔍 Analyze ECG"):

    if len(r_peaks) < 3:
        st.warning("Not enough heartbeats detected for analysis.")
        st.stop()

    beats = extract_beats(ecg, r_peaks)
    beats = np.array([normalize_beat(b) for b in beats])

    preds = []
    probs_all = []

    for beat in beats:
        x = torch.tensor(
            beat,
            dtype=torch.float32
        )[None, None, :].to(DEVICE)

        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1).cpu().numpy()[0]

        preds.append(np.argmax(probs))
        probs_all.append(probs)

    preds = np.array(preds)
    probs_all = np.array(probs_all)

    counts = np.bincount(preds, minlength=3)
    avg_probs = probs_all.mean(axis=0)

    abnormal_beats = counts[1] + counts[2]
    abnormal_ratio = abnormal_beats / len(preds)

    # =========================
    # Final Result (USER VIEW)
    # =========================
    st.markdown("## 🩺 Final Result")

    if abnormal_ratio >= 0.2:
        st.error("⚠️ Abnormal heartbeat detected")
        st.caption(
            "Irregular heartbeat patterns were detected. "
            "This tool is for screening only and not a diagnosis."
        )
    else:
        st.success("✅ Normal heartbeat detected")

    st.markdown("### 📊 Confidence")
    st.write(f"**Normal:** {avg_probs[0]*100:.1f}%")
    st.write(f"**Abnormal:** {(avg_probs[1]+avg_probs[2])*100:.1f}%")
