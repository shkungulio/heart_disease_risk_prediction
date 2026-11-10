"""Interactive demo: enter patient values, see an estimated heart-disease probability.

Run:  streamlit run app/streamlit_app.py
Educational demo only - NOT a medical device.
"""
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data_prep import FEATURES  # noqa: E402

st.set_page_config(page_title="Heart Disease Risk Demo", page_icon="❤️", layout="centered")


@st.cache_resource
def load():
    model = joblib.load(ROOT / "models" / "best_model.joblib")
    metrics = json.loads((ROOT / "reports" / "metrics.json").read_text())
    return model, metrics


model, metrics = load()
tuned_key = next(k for k in metrics if k.endswith("_threshold"))
threshold = metrics[tuned_key]["threshold"]

st.title("❤️ Heart Disease Risk Estimator")
st.warning("Educational portfolio demo trained on 303 patients (UCI Cleveland data). **Not medical advice.**")

c1, c2 = st.columns(2)
with c1:
    age = st.slider("Age", 29, 77, 55)
    sex = st.radio("Sex", [0, 1], format_func=lambda x: "Male" if x else "Female", horizontal=True)
    cp = st.selectbox("Chest pain type", [1, 2, 3, 4],
                      format_func={1: "1 · Typical angina", 2: "2 · Atypical angina", 3: "3 · Non-anginal pain", 4: "4 · Asymptomatic"}.get, index=3)
    trestbps = st.slider("Resting blood pressure (mm Hg)", 90, 200, 130)
    chol = st.slider("Cholesterol (mg/dl)", 120, 570, 245)
    fbs = st.checkbox("Fasting blood sugar > 120 mg/dl")
    restecg = st.selectbox("Resting ECG", [0, 1, 2], format_func={0: "0 · Normal", 1: "1 · ST-T abnormality", 2: "2 · LV hypertrophy"}.get)
with c2:
    thalach = st.slider("Max heart rate achieved", 70, 205, 150)
    exang = st.checkbox("Exercise-induced angina")
    oldpeak = st.slider("ST depression (oldpeak)", 0.0, 6.5, 1.0, 0.1)
    slope = st.selectbox("Slope of peak-exercise ST", [1, 2, 3], format_func={1: "1 · Upsloping", 2: "2 · Flat", 3: "3 · Downsloping"}.get, index=1)
    ca = st.selectbox("Major vessels coloured (0-3)", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", [3, 6, 7], format_func={3: "3 · Normal", 6: "6 · Fixed defect", 7: "7 · Reversible defect"}.get)

row = pd.DataFrame([{
    "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol, "fbs": int(fbs), "restecg": restecg,
    "thalach": thalach, "exang": int(exang), "oldpeak": oldpeak, "slope": slope, "ca": float(ca), "thal": float(thal),
}])[FEATURES]

p = float(model.predict_proba(row)[0, 1])
st.subheader(f"Estimated probability: {p:.0%}")
st.progress(min(p, 1.0))
if p >= threshold:
    st.error(f"Above the screening threshold ({threshold:.2f}, tuned for ≥90% recall): flagged as **higher risk**.")
else:
    st.success(f"Below the screening threshold ({threshold:.2f}): flagged as **lower risk**.")
st.caption(f"Model: {metrics['model']} · test ROC-AUC {metrics['roc_auc']} (n = {metrics['n_test']}, 95% CI {metrics['roc_auc_95ci'][0]}-{metrics['roc_auc_95ci'][1]})")
