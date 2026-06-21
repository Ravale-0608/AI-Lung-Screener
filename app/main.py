"""
Lung Risk Screener — FastAPI Backend

Endpoints:
  POST /predict   Upload a CT scan (.mhd / .nrrd / .nii / .nii.gz)
                  Returns: risk_score, risk_level, top_features
  GET  /health    Liveness probe
  GET  /stats     Training statistics

Auto-segmentation: lungmask (JoHof U-Net) generates the lung mask
automatically — no pre-computed mask required at inference time.

Start with:
  uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

import io
import json
import logging
import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import SimpleITK as sitk
import joblib

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from radiomics import featureextractor

log = logging.getLogger("uvicorn.error")

BASE_DIR  = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "model"

MODEL_PATH    = MODEL_DIR / "lung_screener.pkl"
FEATURES_PATH = MODEL_DIR / "feature_names.json"
STATS_PATH    = MODEL_DIR / "training_stats.json"
FI_PATH       = MODEL_DIR / "feature_importance.json"

EXTRACTOR_PARAMS = {
    "imageType": {"Original": {}},
    "featureClass": {
        "glcm":  [],
        "glrlm": [],
        "glszm": [],
        "gldm":  [],
        "ngtdm": [],
    },
    "setting": {
        "binWidth": 25,
        "resampledPixelSpacing": [1, 1, 1],
        "interpolator": "sitkBSpline",
        "label": 1,
    },
}

# ── App state (loaded once at startup) ────────────────────────────────────────

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    _load_model()
    yield


def _load_model():
    if not MODEL_PATH.exists():
        log.warning(
            f"Model not found at {MODEL_PATH}. "
            "Run 'python Main.py' first to train the model."
        )
        return

    state["pipeline"]      = joblib.load(MODEL_PATH)
    state["feature_names"] = json.loads(FEATURES_PATH.read_text()) if FEATURES_PATH.exists() else []
    state["extractor"]     = featureextractor.RadiomicsFeatureExtractor(**EXTRACTOR_PARAMS)
    state["stats"]         = json.loads(STATS_PATH.read_text()) if STATS_PATH.exists() else {}
    state["fi"]            = json.loads(FI_PATH.read_text()) if FI_PATH.exists() else {}
    log.info("Model and extractor loaded successfully.")


# ── FastAPI app ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Lung Risk Screener API",
    description=(
        "COPD/nodule risk scoring from CT scans. "
        "Replicates Kirby et al. 2023 — Elastic Net + Linear SVM."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Lung segmentation ──────────────────────────────────────────────────────────

def auto_segment(image: sitk.Image) -> sitk.Image:
    """
    Use lungmask (JoHof R231 U-Net) to generate a whole-lung binary mask.
    Falls back to a simple HU-threshold mask if lungmask is not installed.
    """
    try:
        from lungmask import LMInferer
        inferer     = LMInferer()
        seg_array   = inferer.apply(image)                    # 0=bg, 1=right, 2=left
        combined    = (seg_array > 0).astype(np.int16)
    except ImportError:
        log.warning("lungmask not installed — using HU threshold fallback.")
        arr      = sitk.GetArrayFromImage(image).astype(np.float32)
        combined = ((arr > -1000) & (arr < -200)).astype(np.int16)

    mask = sitk.GetImageFromArray(combined)
    mask.CopyInformation(image)
    return mask


# ── Feature extraction ─────────────────────────────────────────────────────────

def extract_features_from_image(
    image: sitk.Image,
    mask: sitk.Image,
    extractor,
    feature_names: list[str],
) -> pd.DataFrame:
    """Extract radiomic features and align to the training feature set."""
    mask = sitk.Cast(mask, sitk.sitkInt32)
    result = extractor.execute(image, mask)
    raw = {k: float(v) for k, v in result.items() if k.startswith("original_")}

    # Align to training features: fill missing with 0 (will be handled by scaler)
    row = {f: raw.get(f, 0.0) for f in feature_names}
    return pd.DataFrame([row])


# ── Feature importance for the prediction ─────────────────────────────────────

def get_top_features(
    pipeline,
    X: pd.DataFrame,
    feature_names: list[str],
    n: int = 5,
) -> dict:
    """
    Compute per-instance feature contributions using Linear SVC coefficients
    (coef * scaled_value → signed contribution, fast alternative to SHAP).
    """
    scaler      = pipeline.named_steps["scaler"]
    selector    = pipeline.named_steps["selector"]
    clf         = pipeline.named_steps["clf"]

    X_scaled    = scaler.transform(X)
    selected_ix = selector.get_support(indices=True)
    coefs       = clf.coef_[0]

    contributions = {
        feature_names[ix]: float(coefs[j] * X_scaled[0, ix])
        for j, ix in enumerate(selected_ix)
    }
    top = dict(sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:n])
    # Readable feature name: strip "original_" prefix and class prefix
    return {
        k.replace("original_", "").replace("glcm_", "glcm·")
         .replace("glrlm_", "glrlm·").replace("glszm_", "glszm·")
         .replace("gldm_", "gldm·").replace("ngtdm_", "ngtdm·"): round(v, 4)
        for k, v in top.items()
    }


# ── Endpoints ──────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "model_loaded": "pipeline" in state}


@app.get("/stats")
async def stats():
    if not state.get("stats"):
        raise HTTPException(503, "Model not trained yet. Run python Main.py first.")
    return state["stats"]


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Upload a CT scan file (.mhd, .nrrd, .nii, .nii.gz).
    Note: .mhd files require a companion .raw file — prefer .nrrd for uploads.
    Returns risk_score (0–1), risk_level, and top 5 contributing features.
    """
    if "pipeline" not in state:
        raise HTTPException(
            503,
            "Model not loaded. Train the model first: python Main.py"
        )

    suffix = Path(file.filename).suffix.lower()
    allowed = {".mhd", ".nrrd", ".nii", ".gz"}
    if suffix not in allowed:
        raise HTTPException(
            400,
            f"Unsupported format '{suffix}'. Use .nrrd, .nii, .nii.gz, or .mhd"
        )

    # Save upload to a temp file so SimpleITK can read it
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = os.path.join(tmpdir, file.filename)
        content  = await file.read()
        with open(tmp_path, "wb") as f:
            f.write(content)

        try:
            image = sitk.ReadImage(tmp_path)
        except Exception as exc:
            raise HTTPException(400, f"Could not read CT scan: {exc}")

        try:
            mask = auto_segment(image)
        except Exception as exc:
            raise HTTPException(500, f"Lung segmentation failed: {exc}")

        try:
            X = extract_features_from_image(
                image,
                mask,
                state["extractor"],
                state["feature_names"],
            )
        except Exception as exc:
            raise HTTPException(500, f"Feature extraction failed: {exc}")

    pipeline = state["pipeline"]

    try:
        risk_score  = float(pipeline.predict_proba(X)[0, 1])
        top_feats   = get_top_features(pipeline, X, state["feature_names"])
    except Exception as exc:
        raise HTTPException(500, f"Prediction failed: {exc}")

    if risk_score < 0.3:
        risk_level = "Low"
    elif risk_score < 0.6:
        risk_level = "Moderate"
    else:
        risk_level = "High"

    return JSONResponse({
        "risk_score":   round(risk_score, 3),
        "risk_level":   risk_level,
        "top_features": top_feats,
        "methodology":  "Elastic Net feature selection + Linear SVM (Kirby et al. 2023)",
    })
