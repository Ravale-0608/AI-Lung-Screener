"""
Lung Nodule Screener — FastAPI Backend

Endpoints:
  POST /predict   Upload a CT scan (.mhd / .nrrd / .nii / .nii.gz)
                  Returns: nodule_probability, nodule_likelihood, top_features
  GET  /health    Liveness probe
  GET  /stats     Training statistics

Auto-segmentation: lungmask (JoHof U-Net) generates the lung mask
automatically — no pre-computed mask required at inference time.

Start with:
  python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
"""

import io
import json
import logging
import os
import shutil
import tempfile
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
import pandas as pd
import SimpleITK as sitk
import joblib

from sklearn import set_config
set_config(transform_output="pandas")

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


app = FastAPI(
    title="Lung Nodule Screener API",
    description=(
        "Pulmonary nodule detection from CT scans. "
        "Elastic Net feature selection + Linear SVM on LUNA16 radiomic features. "
        "Research use only — not for clinical decisions."
    ),
    version="1.1.0",
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
    Falls back to an HU-threshold mask if lungmask is not installed.
    """
    try:
        from lungmask import LMInferer
        inferer   = LMInferer()
        seg_array = inferer.apply(image)
        combined  = (seg_array > 0).astype(np.int16)
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
    """
    Extract radiomic features and align to the full raw feature set used during
    training. Missing features are filled with 0.0; the pipeline's NaNDropper
    and CorrelationFilter handle any further cleaning at inference time.
    """
    mask   = sitk.Cast(mask, sitk.sitkInt32)
    result = extractor.execute(image, mask)
    raw    = {k: float(v) for k, v in result.items() if k.startswith("original_")}
    row    = {f: raw.get(f, 0.0) for f in feature_names}
    return pd.DataFrame([row])


# ── Feature importance for the prediction ─────────────────────────────────────

def get_top_features(pipeline, X: pd.DataFrame, n: int = 5) -> dict:
    """
    Per-instance feature contributions: SVC coefficient × scaled feature value.
    Uses the full pipeline's selector output so contributions are in the same
    space as the classifier's decision boundary.
    """
    selector = pipeline.named_steps["selector"]
    clf      = pipeline.named_steps["clf"]

    # pipeline[:-1] = all steps except clf; output shape (1, n_selected)
    X_sel = pipeline[:-1].transform(X)
    names = selector.get_feature_names_out()
    coefs = clf.coef_[0]

    contributions = {
        name: float(coefs[j] * X_sel.values[0, j])
        for j, name in enumerate(names)
    }
    top = dict(sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)[:n])
    return {
        k.replace("original_", "")
         .replace("glcm_",  "glcm·").replace("glrlm_", "glrlm·")
         .replace("glszm_", "glszm·").replace("gldm_",  "gldm·")
         .replace("ngtdm_", "ngtdm·"): round(v, 4)
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
    Returns nodule_probability (0–1), nodule_likelihood (Low/Moderate/High),
    and top 5 contributing radiomic features.
    Research use only — not for clinical decisions.
    """
    if "pipeline" not in state:
        raise HTTPException(
            503, "Model not loaded. Train the model first: python Main.py"
        )

    suffix  = Path(file.filename).suffix.lower()
    allowed = {".mhd", ".nrrd", ".nii", ".gz"}
    if suffix not in allowed:
        raise HTTPException(
            400, f"Unsupported format '{suffix}'. Use .nrrd, .nii, .nii.gz, or .mhd"
        )

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
                image, mask, state["extractor"], state["feature_names"]
            )
        except Exception as exc:
            raise HTTPException(500, f"Feature extraction failed: {exc}")

    pipeline = state["pipeline"]

    try:
        nodule_probability = float(pipeline.predict_proba(X)[0, 1])
        top_feats          = get_top_features(pipeline, X)
    except Exception as exc:
        raise HTTPException(500, f"Prediction failed: {exc}")

    if nodule_probability < 0.3:
        nodule_likelihood = "Low"
    elif nodule_probability < 0.6:
        nodule_likelihood = "Moderate"
    else:
        nodule_likelihood = "High"

    return JSONResponse({
        "nodule_probability": round(nodule_probability, 3),
        "nodule_likelihood":  nodule_likelihood,
        "top_features":       top_feats,
        "methodology":        "Elastic Net feature selection + Linear SVM (LUNA16)",
    })
