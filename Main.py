#!/usr/bin/env python3
"""
Lung Disease Risk Screener — Training Pipeline
Replicates Kirby et al. 2023: radiomic feature extraction, Elastic Net
feature selection, and Linear SVM classification on LUNA16 CT scans.

Usage:
    python Main.py                  # full run (slow — ~5–30 min/scan)
    python Main.py --limit 10       # quick dev run on 10 scans
    python Main.py --skip-shap      # skip SHAP (faster)
    python Main.py --demo           # generate synthetic labels if no CSV found
"""

import os
import glob
import json
import logging
import argparse
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import SimpleITK as sitk
import joblib

from radiomics import featureextractor
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
SUBSET_DIR = os.path.join(BASE_DIR, "subset1")
SEG_DIR    = os.path.join(BASE_DIR, "seg-lungs-LUNA16")
MODEL_DIR  = os.path.join(BASE_DIR, "model")
CACHE_CSV  = os.path.join(BASE_DIR, "features_cache.csv")

ANNOTATIONS_CSV = os.path.join(BASE_DIR, "annotations.csv")
CANDIDATES_CSV  = os.path.join(BASE_DIR, "candidates.csv")

os.makedirs(MODEL_DIR, exist_ok=True)

# ── PyRadiomics configuration (Kirby 2023 feature sets) ───────────────────────
EXTRACTOR_PARAMS = {
    "imageType": {"Original": {}},
    "featureClass": {
        "glcm":  [],   # 25 features — grey-level co-occurrence
        "glrlm": [],   # 16 features — run-length matrix
        "glszm": [],   # 16 features — size-zone matrix
        "gldm":  [],   # 16 features — dependence matrix
        "ngtdm": [],   #  5 features — neighbourhood grey-tone difference
    },
    "setting": {
        "binWidth": 25,
        "resampledPixelSpacing": [1, 1, 1],   # 1 mm isotropic
        "interpolator": "sitkBSpline",
        "label": 1,
    },
}


# ── Feature extraction ─────────────────────────────────────────────────────────

def build_extractor():
    extractor = featureextractor.RadiomicsFeatureExtractor(**EXTRACTOR_PARAMS)
    extractor.disableAllFeatures()
    for cls in ["glcm", "glrlm", "glszm", "gldm", "ngtdm"]:
        extractor.enableFeatureClassByName(cls)
    return extractor


def extract_scan(series_uid: str, extractor) -> dict | None:
    ct_path   = os.path.join(SUBSET_DIR, f"{series_uid}.mhd")
    mask_path = os.path.join(SEG_DIR,    f"{series_uid}.mhd")

    if not os.path.exists(ct_path) or not os.path.exists(mask_path):
        return None

    try:
        image = sitk.ReadImage(ct_path)
        mask  = sitk.ReadImage(mask_path)

        # LUNA16 lung masks use labels 3 (right lung) and 4 (left lung).
        # Binarize so PyRadiomics sees a single label=1 region covering both lungs.
        arr    = sitk.GetArrayFromImage(mask)
        binary = (arr > 0).astype(np.int16)
        mask   = sitk.GetImageFromArray(binary)
        mask.CopyInformation(sitk.ReadImage(mask_path))
        mask   = sitk.Cast(mask, sitk.sitkInt32)

        result = extractor.execute(image, mask)
        features = {
            k: float(v)
            for k, v in result.items()
            if k.startswith("original_")
        }
        features["series_uid"] = series_uid
        return features
    except Exception as exc:
        log.warning(f"  Skipped {series_uid[:40]}: {exc}")
        return None


def extract_all_features(limit: int | None = None) -> pd.DataFrame:
    if os.path.exists(CACHE_CSV):
        log.info(f"Loading cached features from {CACHE_CSV}")
        return pd.read_csv(CACHE_CSV)

    log.info("Building PyRadiomics extractor (first run may download models)…")
    extractor = build_extractor()

    ct_files    = glob.glob(os.path.join(SUBSET_DIR, "*.mhd"))
    series_uids = [os.path.splitext(os.path.basename(f))[0] for f in ct_files]

    if limit:
        series_uids = series_uids[:limit]
        log.info(f"Development mode: processing {limit} scans")

    log.info(f"Found {len(series_uids)} CT scans in subset1/")
    log.info("⚠  Extraction is slow (~2–10 min per scan). Go make tea.")

    records = []
    for i, uid in enumerate(series_uids, 1):
        log.info(f"[{i}/{len(series_uids)}] {uid[:50]}…")
        result = extract_scan(uid, extractor)
        if result:
            records.append(result)

    df = pd.DataFrame(records)
    df.to_csv(CACHE_CSV, index=False)
    log.info(f"Saved {len(df)} rows → {CACHE_CSV}")
    return df


# ── Label creation ─────────────────────────────────────────────────────────────

def create_labels(df: pd.DataFrame, demo: bool = False) -> pd.DataFrame:
    """
    1 = high risk (scan has a confirmed nodule), 0 = low risk.
    Source: annotations.csv > candidates.csv > demo synthetic labels.

    To get the CSV files:
      • Go to luna16.grand-challenge.org
      • Download annotations.csv and candidates.csv (listed under 'Data')
      • Place them in the project root alongside Main.py
    """
    if os.path.exists(ANNOTATIONS_CSV):
        log.info(f"Labels from {ANNOTATIONS_CSV}")
        ann         = pd.read_csv(ANNOTATIONS_CSV)
        nodule_uids = set(ann["seriesuid"].unique())
        df["label"] = df["series_uid"].isin(nodule_uids).astype(int)

    elif os.path.exists(CANDIDATES_CSV):
        log.info(f"Labels from {CANDIDATES_CSV}")
        cands       = pd.read_csv(CANDIDATES_CSV)
        nodule_uids = set(cands[cands["class"] == 1]["seriesuid"].unique())
        df["label"] = df["series_uid"].isin(nodule_uids).astype(int)

    elif demo:
        log.warning(
            "Demo mode: assigning synthetic labels (alternating 0/1). "
            "Results have NO clinical meaning — for pipeline testing only."
        )
        df["label"] = [i % 2 for i in range(len(df))]

    else:
        raise FileNotFoundError(
            "\n\nNo label file found.\n"
            "Download annotations.csv or candidates.csv from the LUNA16 challenge:\n"
            "  https://luna16.grand-challenge.org  →  Data\n"
            "Place the file in:  " + BASE_DIR + "\n"
            "Or run with --demo to test the pipeline with synthetic labels."
        )

    dist = df["label"].value_counts().to_dict()
    log.info(f"Label distribution → 0 (low risk): {dist.get(0,0)}, 1 (high risk): {dist.get(1,0)}")
    return df


# ── Data cleaning (Kirby 2023 method) ─────────────────────────────────────────

def clean_features(X: pd.DataFrame):
    """
    1. Drop NaN / Inf columns
    2. Remove constant features
    3. Remove outlier rows (any feature > 2 SD)
    4. Remove highly correlated features (Pearson r > 0.90)
    Returns (X_clean, outlier_mask) where outlier_mask is a boolean Series
    aligned with X's original index.
    """
    X = X.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
    X = X.loc[:, X.std() > 0]
    log.info(f"After NaN/constant removal: {X.shape}")

    # Remove features (columns) whose values contain outliers > 2 SD.
    # Kirby 2023 drops unstable features, not entire scans — removing rows
    # with 75+ features almost guarantees half the dataset gets wiped.
    z                = (X - X.mean()) / X.std()
    unstable_cols    = z.abs().gt(2).any(axis=0)
    X_clean          = X.loc[:, ~unstable_cols].copy()
    log.info(f"Removed {unstable_cols.sum()} unstable features (any value >2 SD) → {X_clean.shape[1]} features")

    corr      = X_clean.corr().abs()
    upper     = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
    to_drop   = [c for c in upper.columns if upper[c].gt(0.90).any()]
    X_clean   = X_clean.drop(columns=to_drop)
    log.info(f"Removed {len(to_drop)} correlated features → {X_clean.shape[1]} features remain")

    # No rows removed — return an all-False mask so caller logic is unchanged
    outlier_mask = pd.Series(False, index=X.index)
    return X_clean, outlier_mask


# ── Model training ─────────────────────────────────────────────────────────────

def train(X_train, y_train, X_test, y_test):
    """
    Elastic Net feature selection (treats labels as continuous 0/1 regression
    targets — drives sparse coefficients) followed by Linear SVM classifier.
    This is the Kirby 2023 winning combination.
    """
    pipeline = Pipeline([
        ("scaler",   StandardScaler()),
        ("selector", SelectFromModel(
            ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000),
            threshold="mean",
        )),
        ("clf",      SVC(kernel="linear", probability=True, C=1.0)),
    ])

    log.info("Fitting Elastic Net + Linear SVM…")
    pipeline.fit(X_train, y_train)

    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = pipeline.predict(X_test)
    auc    = roc_auc_score(y_test, y_prob)

    log.info(f"\nTest AUC : {auc:.3f}")
    log.info(f"\n{classification_report(y_test, y_pred, target_names=['Low risk','High risk'])}")

    n_selected = pipeline.named_steps["selector"].get_support().sum()
    log.info(f"Features selected by Elastic Net: {n_selected}")

    # 5-fold cross-validation on full dataset
    X_all = pd.concat([X_train, X_test]).reset_index(drop=True)
    y_all = pd.concat([y_train, y_test]).reset_index(drop=True)
    cv    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X_all, y_all, cv=cv, scoring="roc_auc")
    log.info(f"5-fold CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    return pipeline, auc, cv_scores


# ── SHAP explanations ──────────────────────────────────────────────────────────

def compute_shap(pipeline, X_train: pd.DataFrame, X_test: pd.DataFrame):
    try:
        import shap
    except ImportError:
        log.warning("shap not installed — skipping. pip install shap")
        return None

    log.info("Computing SHAP values (KernelExplainer — may take several minutes)…")
    cols = list(X_train.columns)
    bg   = shap.kmeans(X_train.sample(min(50, len(X_train)), random_state=42), 10)

    predict_fn = lambda x: pipeline.predict_proba(pd.DataFrame(x, columns=cols))[:, 1]
    explainer  = shap.KernelExplainer(predict_fn, bg)
    sample     = X_test.head(min(20, len(X_test)))
    sv         = explainer.shap_values(sample, nsamples=100)

    return explainer, sv, sample


# ── Feature importance from SVC coefficients (fast, no SHAP needed) ───────────

def svc_feature_importance(pipeline, feature_names: list[str]) -> dict:
    selector    = pipeline.named_steps["selector"]
    clf         = pipeline.named_steps["clf"]
    selected_ix = selector.get_support(indices=True)
    coefs       = clf.coef_[0]
    return {feature_names[i]: float(coefs[j]) for j, i in enumerate(selected_ix)}


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit",     type=int, default=None,
                        help="Process only N scans (development speed-up)")
    parser.add_argument("--skip-shap", action="store_true",
                        help="Skip SHAP computation")
    parser.add_argument("--demo",      action="store_true",
                        help="Use synthetic labels if no annotations.csv found")
    args = parser.parse_args()

    # ── 1. Feature extraction ──────────────────────────────────────────────────
    df = extract_all_features(limit=args.limit)
    if df.empty:
        log.error("No features extracted. Check subset1/ and seg-lungs-LUNA16/ exist.")
        return

    # ── 2. Labels ──────────────────────────────────────────────────────────────
    df = create_labels(df, demo=args.demo)

    series_uids = df["series_uid"].reset_index(drop=True)
    y           = df["label"].reset_index(drop=True)
    X_raw       = df.drop(columns=["series_uid", "label"]).reset_index(drop=True)

    # ── 3. Cleaning ────────────────────────────────────────────────────────────
    X_clean, outlier_mask = clean_features(X_raw)
    keep        = ~outlier_mask.values
    y_clean     = y[keep].reset_index(drop=True)
    X_clean     = X_clean.reset_index(drop=True)

    if y_clean.nunique() < 2:
        log.error(
            "Only one class after cleaning — cannot train.\n"
            "Ensure annotations.csv contains nodule-positive AND nodule-negative scans."
        )
        return

    log.info(f"\nFinal dataset: {X_clean.shape[0]} samples × {X_clean.shape[1]} features")

    # ── 4. Train / test split ──────────────────────────────────────────────────
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_clean, y_clean, test_size=0.2, random_state=42, stratify=y_clean
    )

    # ── 5. Train model ─────────────────────────────────────────────────────────
    pipeline, test_auc, cv_scores = train(X_tr, y_tr, X_te, y_te)

    # ── 6. Feature importance ──────────────────────────────────────────────────
    feature_names  = list(X_clean.columns)
    fi_dict        = svc_feature_importance(pipeline, feature_names)
    top5           = dict(sorted(fi_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:5])
    log.info(f"\nTop-5 features by SVC coefficient:\n" +
             "\n".join(f"  {k}: {v:+.4f}" for k, v in top5.items()))

    # ── 7. Optional SHAP ───────────────────────────────────────────────────────
    if not args.skip_shap and len(X_te) >= 5:
        shap_result = compute_shap(pipeline, X_tr, X_te)
        if shap_result:
            explainer, sv, sample = shap_result
            joblib.dump(explainer, os.path.join(MODEL_DIR, "shap_explainer.pkl"))
            np.save(os.path.join(MODEL_DIR, "shap_values.npy"), sv)
            sample.to_csv(os.path.join(MODEL_DIR, "shap_sample.csv"), index=False)
            log.info("SHAP explainer saved.")

    # ── 8. Save model artifacts ────────────────────────────────────────────────
    joblib.dump(pipeline, os.path.join(MODEL_DIR, "lung_screener.pkl"))

    with open(os.path.join(MODEL_DIR, "feature_names.json"), "w") as f:
        json.dump(feature_names, f)

    with open(os.path.join(MODEL_DIR, "feature_importance.json"), "w") as f:
        json.dump(fi_dict, f, indent=2)

    stats = {
        "test_auc":         round(test_auc, 4),
        "cv_auc_mean":      round(float(cv_scores.mean()), 4),
        "cv_auc_std":       round(float(cv_scores.std()), 4),
        "n_train":          len(X_tr),
        "n_test":           len(X_te),
        "n_features_raw":   X_raw.shape[1],
        "n_features_clean": X_clean.shape[1],
        "n_features_selected": int(pipeline.named_steps["selector"].get_support().sum()),
    }
    with open(os.path.join(MODEL_DIR, "training_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    log.info(f"\n{'='*55}")
    log.info(f"  Model saved  →  model/lung_screener.pkl")
    log.info(f"  Test AUC     →  {test_auc:.3f}")
    log.info(f"  CV AUC       →  {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    log.info(f"{'='*55}")
    log.info("Next steps:")
    log.info("  API    →  uvicorn app.main:app --reload")
    log.info("  UI     →  streamlit run frontend/app.py")
    log.info("  Docker →  docker-compose up --build")


if __name__ == "__main__":
    main()
