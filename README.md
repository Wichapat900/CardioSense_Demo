# CardioSense Demo 🫀

CardioSense is an ECG arrhythmia detection demo using a PyTorch CNN model.
It classifies ECG beats into:

- Normal
- PAC (Premature Atrial Contraction)
- PVC (Premature Ventricular Contraction)

## Project Structure

- `CardioSense/src/app.py` – Streamlit app
- `CardioSense/models/` – Trained PyTorch model
- `CardioSense/data/` – Preprocessed ECG beats
- `CardioSense/demo_ecg/` – Sample ECG signals

## Run Locally

```bash
pip install -r requirements.txt
streamlit run CardioSense/src/app.py
