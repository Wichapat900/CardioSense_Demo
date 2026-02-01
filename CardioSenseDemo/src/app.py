import os
import numpy as np
import torch
import streamlit as st
import matplotlib.pyplot as plt

from model import ECG_CNN
from utils import normalize_beat, detect_r_peaks, extract_beats
from beat_detection import detect_r_peaks


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

@st.cache_resource
def load_model():
    model = ECG_CNN(num_classes=3)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

def plot_ecg(signal, fs, r_peaks=None):
    t = np.arange(len(signal)) / fs
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, signal, color="black")
    if r_peaks is not None:
        ax.scatter(r_peaks / fs, signal[r_peaks], color="red", s=20)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("mV")
    ax.set_title("ECG Signal")
    return fig

st.set_page_config(page_title="CardioSense", layout="centered")
st.title("🫀 CardioSense")
st.write("AI-based Arrhythmia Detection (Normal / PAC / PVC)")
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

r_peaks = detect_r_peaks(ecg, FS)

st.pyplot(plot_ecg(ecg, FS, r_peaks))
st.write(f"Detected **{len(r_peaks)} heartbeats**")

if st.button("🔍 Analyze ECG"):
    beats = extract_beats(ecg, r_peaks)
    beats = np.array([normalize_beat(b) for b in beats])

    preds, probs_all = [], []

    for beat in beats:
        x = torch.tensor(beat, dtype=torch.float32)[None, None, :].to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1).cpu().numpy()[0]
        preds.append(np.argmax(probs))
        probs_all.append(probs)

    counts = np.bincount(preds, minlength=3)
    avg_probs = np.mean(probs_all, axis=0)

    st.markdown("## 🩺 Final Result")

    if np.argmax(counts) == 0:
        st.success("✅ Normal rhythm detected")
    elif np.argmax(counts) == 1:
        st.warning("⚠️ PAC detected")
    else:
        st.error("🚨 PVC detected")

    for i, label in enumerate(CLASSES):
        st.write(f"**{label}:** {avg_probs[i]*100:.1f}%")
