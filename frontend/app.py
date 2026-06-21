"""
Lung Risk Screener — Streamlit Frontend

Start with:
  streamlit run frontend/app.py

The API must be running first:
  uvicorn app.main:app --reload --port 8000
"""

import json
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Lung Risk Screener",
    page_icon="🫁",
    layout="centered",
)

# ── Header ─────────────────────────────────────────────────────────────────────

st.title("Lung Disease Risk Screener")
st.caption(
    "Radiomic feature extraction · Elastic Net + Linear SVM · "
    "Based on Kirby et al. 2023"
)
st.divider()

# ── Sidebar: model info ────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Model Information")
    try:
        resp  = requests.get(f"{API_URL}/stats", timeout=3)
        stats = resp.json()
        st.metric("Test AUC",  f"{stats.get('test_auc', 'N/A'):.3f}")
        st.metric("CV AUC",    f"{stats.get('cv_auc_mean', 'N/A'):.3f} ± {stats.get('cv_auc_std', 'N/A'):.3f}")
        st.metric("Features (selected)",  stats.get("n_features_selected", "N/A"))
        st.metric("Training samples",     stats.get("n_train", "N/A"))
    except Exception:
        st.warning("API offline — start the backend first.")
        st.code("uvicorn app.main:app --reload")

    st.divider()
    st.subheader("Methodology")
    st.markdown(
        "Features extracted using **PyRadiomics** (GLCM, GLRLM, GLSZM, GLDM, NGTDM). "
        "Cleaning: outliers >2 SD removed, correlated features (r>0.90) dropped. "
        "Selection: **Elastic Net** regression (α=0.01, l₁=0.5). "
        "Classifier: **Linear SVM**. "
        "\n\n*Kirby et al. 2023 — Radiomics for COPD risk stratification.*"
    )

# ── Upload panel ───────────────────────────────────────────────────────────────

st.subheader("Upload CT Scan")
st.markdown(
    "Upload a lung CT scan in **.nrrd**, **.nii**, **.nii.gz**, or **.mhd** format. "
    "The model auto-segments the lung and returns a risk score."
)

uploaded = st.file_uploader(
    "Drag and drop or browse",
    type=["nrrd", "nii", "gz", "mhd"],
    help="Single-file formats preferred: .nrrd or .nii.gz",
)

if uploaded is None:
    st.info("Upload a CT scan above to get a risk prediction.")
    st.stop()

# ── Prediction ─────────────────────────────────────────────────────────────────

with st.spinner("Segmenting lungs and extracting radiomic features…"):
    try:
        response = requests.post(
            f"{API_URL}/predict",
            files={"file": (uploaded.name, uploaded.getvalue(), "application/octet-stream")},
            timeout=300,   # feature extraction can take a few minutes
        )
    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot reach the API. Make sure the backend is running:\n\n"
            "```\nuvicorn app.main:app --reload\n```"
        )
        st.stop()

if response.status_code != 200:
    detail = response.json().get("detail", response.text)
    st.error(f"API error ({response.status_code}): {detail}")
    st.stop()

result = response.json()

# ── Results ────────────────────────────────────────────────────────────────────

st.divider()
st.subheader("Prediction Result")

score = result["risk_score"]
level = result["risk_level"]

colour_map = {"Low": "🟢", "Moderate": "🟡", "High": "🔴"}
icon       = colour_map.get(level, "⚪")

col1, col2 = st.columns(2)
with col1:
    st.metric(f"{icon}  Risk Level", level)
with col2:
    st.metric("Risk Score", f"{score:.1%}")

# Progress bar as visual indicator
bar_colour = {"Low": "green", "Moderate": "orange", "High": "red"}.get(level, "gray")
st.progress(score, text=f"Score: {score:.3f}")

# ── Feature importance chart ───────────────────────────────────────────────────

top_feats = result.get("top_features", {})
if top_feats:
    st.divider()
    st.subheader("Top Contributing Features")
    st.caption(
        "Feature contributions = SVC coefficient × scaled feature value. "
        "Positive → pushes toward high risk; negative → toward low risk."
    )

    import pandas as pd
    feat_df = (
        pd.DataFrame(list(top_feats.items()), columns=["Feature", "Contribution"])
        .sort_values("Contribution", key=abs, ascending=True)
        .set_index("Feature")
    )
    st.bar_chart(feat_df)

# ── Disclaimer ────────────────────────────────────────────────────────────────

st.divider()
st.caption(
    "⚠️  **Research tool only — not for clinical use.**  "
    "This model was trained on LUNA16 (subset1) and replicates Kirby et al. 2023 "
    "for educational purposes. Consult a qualified radiologist for clinical decisions."
)
