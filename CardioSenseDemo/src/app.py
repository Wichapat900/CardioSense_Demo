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
MODEL_PATH = os.path.join(BASE_DIR, "models", "CardioSense_Model.pth")
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
    # ---- ECG GRID ----
    ax.set_xlim(t[0], t[-1])
    y_min, y_max = np.min(signal) - 0.2, np.max(signal) + 0.2
    ax.set_ylim(y_min, y_max)
    # Small squares (0.04s, 0.1mV)
    ax.set_xticks(np.arange(0, t[-1], 0.04), minor=True)
    ax.set_yticks(np.arange(y_min, y_max, 0.1), minor=True)
    # Big squares (0.2s, 0.5mV)
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
    
    preds = []
    confidences = []
    for beat in beats:
        beat = normalize_beat(beat)
        x = torch.tensor(beat, dtype=torch.float32)[None, None, :].to(DEVICE)
        with torch.no_grad():
            probs = torch.softmax(model(x), dim=1).cpu().numpy()[0]
        preds.append(np.argmax(probs))
        confidences.append(np.max(probs))
    
    preds = np.array(preds)
    confidences = np.array(confidences)
    
    # ================= NORMAL vs ABNORMAL =================
    normal_count = np.sum(preds == 0)
    pac_count = np.sum(preds == 1)
    pvc_count = np.sum(preds == 2)
    abnormal_count = pac_count + pvc_count
    
    st.markdown("## 🩺 Final Result")
    
    if abnormal_count >= 1:
        st.error("🚨 **Abnormal rhythm detected**")
        
        # Specify what was detected
        detected = []
        if pac_count > 0:
            detected.append(f"PAC ({pac_count} beats)")
        if pvc_count > 0:
            detected.append(f"PVC ({pvc_count} beats)")
        
        st.write(f"**Detected:** {' and '.join(detected)}")
        st.caption("⚕️ *This screening tool cannot diagnose conditions. Consult a healthcare provider for evaluation.*")
    else:
        st.success("✅ **Normal rhythm detected**")
    
    # ================= MODEL CONFIDENCE =================
    st.markdown("### 📊 Model Confidence")
    avg_confidence = np.mean(confidences) * 100
    st.write(f"**Average Confidence:** {avg_confidence:.1f}%")
    st.write(f"**Range:** {np.min(confidences) * 100:.1f}% - {np.max(confidences) * 100:.1f}%")
