import os
import numpy as np
import torch
import streamlit as st
import matplotlib.pyplot as plt

from model import ECG_CNN
from utils import normalize_beat, extract_beats
from beat_detection import detect_r_peaks

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

@st.cache_resource
def load_model():
    model = ECG_CNN(num_classes=3)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

model = load_model()

def plot_ecg(ecg, fs, r_peaks):
    t = np.arange(len(ecg)) / fs
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(t, ecg, color="black")
    ax.scatter(r_peaks / fs, ecg[r_peaks], color="red", s=25)
    ax.set_title("ECG Signal (R-peaks detected)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("mV")
    return fig

st.set_page_config(page_title="CardioSense", layout="centered")
st.title("🫀 CardioSense")
st.write("AI-based ECG Analysis (Normal / Abnormal)")
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

    if len(beats) == 0:
        st.error("No valid heartbeats detected.")
        st.stop()

    beats = np.array([normalize_beat(b) for b in beats])

    preds = []

    for beat in beats:
        x = torch.tensor(beat, dtype=torch.float32)[None, None, :].to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1)
        preds.append(torch.argmax(probs).item())

    preds = np.array(preds)

    abnormal_ratio = np.mean(preds != 0)

    st.markdown("## 🩺 Final Result")

    if abnormal_ratio < 0.2:
        st.success("✅ **Normal heart rhythm detected**")
    else:
        st.error("🚨 **Abnormal heart rhythm detected**")

    st.caption("Abnormal includes irregular beats such as early or skipped heartbeats.")
