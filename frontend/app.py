"""
Lung Nodule Screener — Streamlit Frontend

Start with:
  python -m streamlit run frontend/app.py

The API must be running first:
  python -m uvicorn app.main:app --reload --port 8000
"""

import json
import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

# ── Page config ────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Lung Nodule Screener",
    page_icon="🫁",
    layout="centered",
)

# ── Header ─────────────────────────────────────────────────────────────────────

st.title("Lung Nodule Screener")
st.caption(
    "Radiomic feature extraction · Elastic Net + Linear SVM · LUNA16"
)
st.warning(
    "**Research tool only — not for clinical use.** "
    "This model was trained on LUNA16 CT scans to detect pulmonary nodules. "
    "It has not been validated for clinical decision-making. "
    "Consult a qualified radiologist for any clinical assessment.",
    icon="⚠️",
)
st.divider()

# ── Sidebar: model info ────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Model Information")
    try:
        resp  = requests.get(f"{API_URL}/stats", timeout=3)
        stats = resp.json()
        ci_lo = stats.get("test_auc_ci_lo", "")
        ci_hi = stats.get("test_auc_ci_hi", "")
        ci_str = f"  (95% CI {ci_lo:.3f}–{ci_hi:.3f})" if ci_lo and ci_hi else ""
        st.metric("Test AUC",  f"{stats.get('test_auc', 'N/A'):.3f}{ci_str}")
        st.metric("CV AUC",
                  f"{stats.get('cv_auc_mean', 'N/A'):.3f} ± {stats.get('cv_auc_std', 'N/A'):.3f}")
        st.metric("Features (selected)",  stats.get("n_features_selected", "N/A"))
        st.metric("Training samples",     stats.get("n_train", "N/A"))
    except Exception:
        st.warning("API offline — start the backend first.")
        st.code("python -m uvicorn app.main:app --reload")

    st.divider()
    st.subheader("Methodology")
    st.markdown(
        "Features extracted using **PyRadiomics** (GLCM, GLRLM, GLSZM, GLDM, NGTDM — 75 total). "
        "Cleaning: winsorize ±2 SD, drop correlated features (r>0.90). "
        "Selection: **Elastic Net** (α=0.01, L1=0.5). "
        "Classifier: **Linear SVM**."
    )

# ── Upload panel ───────────────────────────────────────────────────────────────

st.subheader("Upload CT Scan")
st.markdown(
    "Upload a lung CT scan in **.nrrd**, **.nii**, **.nii.gz**, or **.mhd** format. "
    "The model auto-segments the lung and returns a nodule probability."
)

uploaded = st.file_uploader(
    "Drag and drop or browse",
    type=["nrrd", "nii", "gz", "mhd"],
    help="Single-file formats preferred: .nrrd or .nii.gz",
)

if uploaded is None:
    st.info("Upload a CT scan above to get a nodule prediction.")
    st.stop()

# ── Prediction ─────────────────────────────────────────────────────────────────

with st.spinner("Segmenting lungs and extracting radiomic features…"):
    try:
        response = requests.post(
            f"{API_URL}/predict",
            files={"file": (uploaded.name, uploaded.getvalue(), "application/octet-stream")},
            timeout=300,
        )
    except requests.exceptions.ConnectionError:
        st.error(
            "Cannot reach the API. Make sure the backend is running:\n\n"
            "```\npython -m uvicorn app.main:app --reload\n```"
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

score      = result["nodule_probability"]
likelihood = result["nodule_likelihood"]

colour_map = {"Low": "🟢", "Moderate": "🟡", "High": "🔴"}
icon       = colour_map.get(likelihood, "⚪")

col1, col2 = st.columns(2)
with col1:
    st.metric(f"{icon}  Nodule Likelihood", likelihood)
with col2:
    st.metric("Nodule Probability", f"{score:.1%}")

st.progress(score, text=f"Score: {score:.3f}")

# ── Feature importance chart ───────────────────────────────────────────────────

top_feats = result.get("top_features", {})
if top_feats:
    st.divider()
    st.subheader("Top Contributing Features")
    st.caption(
        "Feature contributions = SVC coefficient × scaled feature value. "
        "Positive → pushes toward nodule-present; negative → toward nodule-absent."
    )

    import pandas as pd
    feat_df = (
        pd.DataFrame(list(top_feats.items()), columns=["Feature", "Contribution"])
        .sort_values("Contribution", key=abs, ascending=True)
        .set_index("Feature")
    )
    st.bar_chart(feat_df)
