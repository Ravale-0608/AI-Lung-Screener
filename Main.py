"""
Lung Disease Risk Screener — Training Pipeline
Radiomic feature extraction, Elastic Net feature selection,
and Linear SVM classification on LUNA16 CT scans.

Usage:
    python Main.py                  # full run (loads from cache if available)
    python Main.py --limit 10       # quick dev run on 10 scans
    python Main.py --skip-shap      # skip SHAP (faster)
    python Main.py --demo           # use synthetic labels if no CSV found
    python Main.py --baselines      # also compute lung-volume / emphysema baselines
"""

import os
import glob
import json
import hashlib
import logging
import argparse
import shutil
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import SimpleITK as sitk
import joblib

from sklearn import set_config
set_config(transform_output="pandas")

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GroupKFold, cross_val_score,
)
from sklearn.metrics import roc_auc_score, classification_report
from radiomics import featureextractor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
SUBSET_DIRS = sorted(glob.glob(os.path.join(BASE_DIR, "subset*")))
SEG_DIR     = os.path.join(BASE_DIR, "seg-lungs-LUNA16")
MODEL_DIR   = os.path.join(BASE_DIR, "model")

ANNOTATIONS_CSV = os.path.join(BASE_DIR, "annotations.csv")
CANDIDATES_CSV  = os.path.join(BASE_DIR, "candidates.csv")

os.makedirs(MODEL_DIR, exist_ok=True)

# Actual feature counts (verified from extraction output):
# glcm=24, glrlm=16, glszm=16, gldm=14, ngtdm=5  →  total=75
EXTRACTOR_PARAMS = {
    "imageType": {"Original": {}},
    "featureClass": {
        "glcm":  [],   # 24 features — grey-level co-occurrence matrix
        "glrlm": [],   # 16 features — run-length matrix
        "glszm": [],   # 16 features — size-zone matrix
        "gldm":  [],   # 14 features — dependence matrix
        "ngtdm": [],   #  5 features — neighbourhood grey-tone difference
    },
    "setting": {
        "binWidth": 25,
        "resampledPixelSpacing": [1, 1, 1],
        "interpolator": "sitkBSpline",
        "label": 1,
    },
}

_params_hash    = hashlib.md5(
    json.dumps(EXTRACTOR_PARAMS, sort_keys=True).encode()
).hexdigest()[:8]
CACHE_CSV       = os.path.join(BASE_DIR, f"features_cache_{_params_hash}.csv")
BASELINES_CACHE = os.path.join(BASE_DIR, "baselines_cache.csv")


# ── Custom sklearn transformers ────────────────────────────────────────────────
# All three are inside the Pipeline so fit() sees only the training fold.

class NaNDropper(BaseEstimator, TransformerMixin):
    """Drop features with any NaN/Inf or zero variance (fitted on train only)."""

    def fit(self, X, y=None):
        Xdf = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        bad = Xdf.replace([np.inf, -np.inf], np.nan).isna().any() | (Xdf.std() == 0)
        self.cols_to_keep_ = Xdf.columns[~bad].tolist()
        return self

    def transform(self, X):
        Xdf = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        return Xdf[self.cols_to_keep_]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.cols_to_keep_)


class Winsorizer(BaseEstimator, TransformerMixin):
    """Clip feature values to ±n_std of the training distribution (fitted on train only)."""

    def __init__(self, n_std=2):
        self.n_std = n_std

    def fit(self, X, y=None):
        Xdf = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        self.mean_ = Xdf.mean()
        self.std_  = Xdf.std()
        return self

    def transform(self, X):
        Xdf = (
            pd.DataFrame(X, columns=self.mean_.index)
            if not isinstance(X, pd.DataFrame) else X
        )
        return Xdf.clip(
            lower=self.mean_ - self.n_std * self.std_,
            upper=self.mean_ + self.n_std * self.std_,
            axis=1,
        )

    def get_feature_names_out(self, input_features=None):
        return np.array(self.mean_.index.tolist())


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Drop features where |Pearson r| > threshold with any earlier feature (fitted on train only)."""

    def __init__(self, threshold=0.90):
        self.threshold = threshold

    def fit(self, X, y=None):
        Xdf = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        corr  = Xdf.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
        drop  = {c for c in upper.columns if upper[c].gt(self.threshold).any()}
        self.cols_to_keep_ = [c for c in Xdf.columns if c not in drop]
        self.n_dropped_    = len(drop)
        return self

    def transform(self, X):
        Xdf = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
        return Xdf[self.cols_to_keep_]

    def get_feature_names_out(self, input_features=None):
        return np.array(self.cols_to_keep_)


def build_pipeline(max_features=None, threshold="mean", probability=False) -> Pipeline:
    """
    Build the full cleaning + classification pipeline.
    probability=False (default) skips Platt scaling — use for CV.
    probability=True for the final saved model so the API can call predict_proba.
    """
    if max_features is not None:
        sel = SelectFromModel(
            ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000),
            max_features=max_features,
            threshold=-np.inf,
        )
    else:
        sel = SelectFromModel(
            ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000),
            threshold=threshold,
        )
    return Pipeline([
        ("nan_dropper", NaNDropper()),
        ("winsorizer",  Winsorizer(n_std=2)),
        ("corr_filter", CorrelationFilter(threshold=0.90)),
        ("scaler",      StandardScaler()),
        ("selector",    sel),
        ("clf",         SVC(kernel="linear", probability=probability, C=1.0)),
    ])


# ── Feature extraction ─────────────────────────────────────────────────────────

def build_extractor():
    extractor = featureextractor.RadiomicsFeatureExtractor(**EXTRACTOR_PARAMS)
    extractor.disableAllFeatures()
    for cls in ["glcm", "glrlm", "glszm", "gldm", "ngtdm"]:
        extractor.enableFeatureClassByName(cls)
    return extractor


def find_ct_path(series_uid: str) -> str | None:
    for d in SUBSET_DIRS:
        p = os.path.join(d, f"{series_uid}.mhd")
        if os.path.exists(p):
            return p
    return None


def extract_scan(series_uid: str, extractor) -> dict | None:
    ct_path   = find_ct_path(series_uid)
    mask_path = os.path.join(SEG_DIR, f"{series_uid}.mhd")

    if ct_path is None or not os.path.exists(mask_path):
        return None

    try:
        image = sitk.ReadImage(ct_path)
        mask  = sitk.ReadImage(mask_path)

        # LUNA16 masks use labels 3 (right lung) and 4 (left lung).
        # Binarize to label=1 covering both lungs.
        arr    = sitk.GetArrayFromImage(mask)
        binary = (arr > 0).astype(np.int16)
        mask   = sitk.GetImageFromArray(binary)
        mask.CopyInformation(sitk.ReadImage(mask_path))
        mask   = sitk.Cast(mask, sitk.sitkInt32)

        result   = extractor.execute(image, mask)
        features = {k: float(v) for k, v in result.items() if k.startswith("original_")}
        features["series_uid"] = series_uid
        return features
    except Exception as exc:
        log.warning(f"  Skipped {series_uid[:40]}: {exc}")
        return None


def extract_all_features(limit: int | None = None) -> pd.DataFrame:
    old_cache = os.path.join(BASE_DIR, "features_cache.csv")
    if not os.path.exists(CACHE_CSV) and os.path.exists(old_cache):
        shutil.copy(old_cache, CACHE_CSV)
        log.info(f"Migrated existing cache → {os.path.basename(CACHE_CSV)}")

    if os.path.exists(CACHE_CSV):
        log.info(f"Loading cached features from {CACHE_CSV}")
        df = pd.read_csv(CACHE_CSV)
        if limit:
            df = df.head(limit)
            log.info(f"(limited to {limit} scans from cache)")
        return df

    log.info("Building PyRadiomics extractor…")
    extractor = build_extractor()

    ct_files    = [f for d in SUBSET_DIRS for f in glob.glob(os.path.join(d, "*.mhd"))]
    series_uids = [os.path.splitext(os.path.basename(f))[0] for f in ct_files]
    if limit:
        series_uids = series_uids[:limit]
        log.info(f"Development mode: processing {limit} scans")

    log.info(f"Found {len(series_uids)} CT scans across {[os.path.basename(d) for d in SUBSET_DIRS]}")
    log.info("Extraction is slow (~2–10 min per scan).")

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
    Labels: 1 = nodule present (scan has ≥1 confirmed LUNA16 annotation ≥3 mm),
            0 = nodule absent.
    Source priority: annotations.csv > candidates.csv > synthetic demo labels.
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
        log.warning("Demo mode: synthetic labels (alternating 0/1). Not for clinical use.")
        df["label"] = [i % 2 for i in range(len(df))]

    else:
        raise FileNotFoundError(
            "\n\nNo label file found. Download annotations.csv from:\n"
            "  https://luna16.grand-challenge.org  →  Data\n"
            f"Place it in: {BASE_DIR}\n"
            "Or run with --demo for synthetic labels."
        )

    dist = df["label"].value_counts().to_dict()
    log.info(
        f"Label distribution → nodule-absent (0): {dist.get(0,0)}, "
        f"nodule-present (1): {dist.get(1,0)}"
    )
    return df


# ── Evaluation helpers ─────────────────────────────────────────────────────────

def bootstrap_auc_ci(y_true, y_score, n: int = 1000, random_state: int = 42):
    """Bootstrap 95% CI on AUC from n resamples of the test set."""
    rng  = np.random.RandomState(random_state)
    y_t  = np.asarray(y_true)
    y_s  = np.asarray(y_score)
    aucs = []
    for _ in range(n):
        ix = rng.choice(len(y_t), len(y_t), replace=True)
        if len(np.unique(y_t[ix])) < 2:
            continue
        aucs.append(roc_auc_score(y_t[ix], y_s[ix]))
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))


def read_scan_spacing(series_uid: str) -> dict | None:
    """Read ElementSpacing from .mhd header without loading pixel data."""
    ct_path = find_ct_path(series_uid)
    if ct_path is None:
        return None
    try:
        reader = sitk.ImageFileReader()
        reader.SetFileName(ct_path)
        reader.ReadImageInformation()
        sp = reader.GetSpacing()
        return {
            "series_uid": series_uid,
            "spacing_x":  sp[0],
            "spacing_y":  sp[1],
            "spacing_z":  sp[2],
        }
    except Exception:
        return None


def check_spacing_confound(series_uids: pd.Series, y: pd.Series) -> dict:
    """
    Report raw AUC of each spacing axis predicting the nodule label.
    AUC is NOT flipped: a value below 0.5 means the predictor anti-correlates
    with the label (high spacing → fewer nodules). The null expectation is
    AUC ≈ 0.50; deviation > 0.15 in either direction is flagged.
    LUNA16 subsets are random CV splits, not acquisition sites.
    """
    log.info("\nReading scan spacings from .mhd headers…")
    rows = []
    for uid, label in zip(series_uids, y):
        sp = read_scan_spacing(uid)
        if sp is not None:
            sp["label"] = label
            rows.append(sp)

    if not rows:
        log.warning("No spacing data found — skipping confound check.")
        return {}

    sp_df   = pd.DataFrame(rows)
    results = {}
    for col in ["spacing_x", "spacing_y", "spacing_z"]:
        auc  = roc_auc_score(sp_df["label"], sp_df[col])
        flag = "  ← deviates from null" if abs(auc - 0.5) > 0.15 else ""
        results[col] = round(auc, 3)
        log.info(f"  {col} AUC: {results[col]:.3f}  (null ≈ 0.50){flag}")

    return results


def run_spacing_group_cv(
    X: pd.DataFrame, y: pd.Series, series_uids: pd.Series, n_bins: int = 5
) -> dict:
    """
    Re-run CV using GroupKFold on binned spacing_z. Each fold's test set
    contains a different slice-thickness bin from the training folds.
    Per-fold try/except handles single-class test folds (common with small n
    and correlated spacing).
    """
    log.info("\nGroupKFold CV on binned spacing_z…")
    rows = []
    for uid, idx in zip(series_uids.values, X.index):
        sp = read_scan_spacing(uid)
        if sp is not None:
            rows.append({"orig_idx": idx, "spacing_z": sp["spacing_z"]})

    if not rows:
        log.warning("  No spacing data — skipping GroupKFold.")
        return {}

    sp_df   = pd.DataFrame(rows).set_index("orig_idx")
    valid   = sp_df.index.intersection(X.index)
    X_sub   = X.loc[valid].reset_index(drop=True)
    y_sub   = y.loc[valid].reset_index(drop=True)
    sp_vals = sp_df.loc[valid, "spacing_z"].values

    # Integer bin codes; duplicates="drop" merges ties
    cut = pd.qcut(sp_vals, q=n_bins, labels=False, duplicates="drop")
    # pd.qcut may return a Categorical; coerce to nullable then to int
    groups   = pd.array(cut, dtype="Int64").to_numpy(dtype=float, na_value=np.nan)
    not_nan  = ~np.isnan(groups)
    groups   = groups[not_nan].astype(int)
    X_sub    = X_sub.iloc[not_nan].reset_index(drop=True)
    y_sub    = y_sub.iloc[not_nan].reset_index(drop=True)

    n_actual = len(np.unique(groups))
    if n_actual < 2:
        log.warning("  Not enough distinct spacing groups — skipping GroupKFold.")
        return {}

    gkf         = GroupKFold(n_splits=n_actual)
    fold_scores = []
    for tr_idx, te_idx in gkf.split(X_sub, y_sub, groups=groups):
        try:
            p = build_pipeline()   # probability=False — no Platt scaling
            p.fit(X_sub.iloc[tr_idx], y_sub.iloc[tr_idx])
            score = roc_auc_score(
                y_sub.iloc[te_idx],
                p.decision_function(X_sub.iloc[te_idx]),
            )
            fold_scores.append(score)
        except ValueError:
            log.warning("  Skipped fold (single-class test set)")
            fold_scores.append(np.nan)

    valid_scores = [s for s in fold_scores if not np.isnan(s)]
    if not valid_scores:
        log.warning(
            f"  All {len(fold_scores)} GroupKFold folds failed (single-class test folds). "
            "This indicates the spacing_z bins are strongly correlated with the label."
        )
        return {
            "mean": None, "std": None,
            "n_bins": n_actual, "n": len(X_sub), "n_valid_folds": 0,
        }

    mean_auc = float(np.mean(valid_scores))
    std_auc  = float(np.std(valid_scores))
    log.info(
        f"  GroupKFold (spacing_z, {n_actual} bins) CV AUC: "
        f"{mean_auc:.3f} ± {std_auc:.3f}  "
        f"({len(valid_scores)}/{len(fold_scores)} valid folds, n={len(X_sub)})"
    )
    return {
        "mean":          round(mean_auc, 3),
        "std":           round(std_auc,  3),
        "n_bins":        n_actual,
        "n":             len(X_sub),
        "n_valid_folds": len(valid_scores),
    }


def compute_baseline_measurements(series_uids: pd.Series) -> pd.DataFrame:
    """
    Load or compute per-scan lung volume (mL) and emphysema density fraction
    (% voxels < −950 HU within the lung mask). Results are cached.
    """
    if os.path.exists(BASELINES_CACHE):
        log.info(f"Loading cached baselines from {BASELINES_CACHE}")
        return pd.read_csv(BASELINES_CACHE)

    log.info("Computing baselines (reading CT images and masks)…")
    records = []
    for i, uid in enumerate(series_uids, 1):
        ct_path   = find_ct_path(uid)
        mask_path = os.path.join(SEG_DIR, f"{uid}.mhd")
        if ct_path is None or not os.path.exists(mask_path):
            continue
        try:
            image = sitk.ReadImage(ct_path)
            mask  = sitk.ReadImage(mask_path)
            arr   = sitk.GetArrayFromImage(image).astype(np.float32)
            marr  = sitk.GetArrayFromImage(mask)
            lung  = (marr > 0)
            sp    = image.GetSpacing()
            vox   = sp[0] * sp[1] * sp[2]
            n_l   = int(lung.sum())
            n_e   = int(((arr < -950) & lung).sum())
            records.append({
                "series_uid":     uid,
                "lung_volume_ml": float(n_l * vox / 1000),
                "emphysema_frac": float(n_e / max(n_l, 1)),
            })
            if i % 20 == 0:
                log.info(f"  [{i}/{len(series_uids)}] done")
        except Exception as exc:
            log.warning(f"  Baseline skipped {uid[:40]}: {exc}")

    df = pd.DataFrame(records)
    df.to_csv(BASELINES_CACHE, index=False)
    log.info(f"Saved baselines → {BASELINES_CACHE}")
    return df


def evaluate_baselines(
    measurements: pd.DataFrame, series_uids: pd.Series, y: pd.Series
) -> dict:
    """
    Compute raw AUC of lung-volume and emphysema fraction vs labels for the
    given scan set. AUC is NOT flipped — values below 0.5 mean the predictor
    anti-correlates with the label. Null expectation is AUC ≈ 0.50.
    Evaluated on the same split as the model (series_uids and y must match).
    """
    uid_to_label = dict(zip(series_uids.values, y.values))
    df = measurements.copy()
    df["label"] = df["series_uid"].map(uid_to_label)
    df = df.dropna(subset=["label"])

    aucs = {}
    for col in ["lung_volume_ml", "emphysema_frac"]:
        auc = roc_auc_score(df["label"], df[col])
        aucs[col] = round(auc, 3)
        log.info(f"  Baseline AUC ({col}): {aucs[col]:.3f}  (null ≈ 0.50, n={len(df)})")
    return aucs


def compare_feature_counts(
    X_tr: pd.DataFrame, y_tr: pd.Series,
    X_te: pd.DataFrame, y_te: pd.Series,
) -> dict:
    """
    Fit each SelectFromModel configuration on the training split only and
    evaluate test-set AUC on the held-out test split (same split as the main model).
    """
    log.info("\nFeature-count comparison (same train/test split as main model):")
    results = {}
    for label, kw in [
        ("max_5",  {"max_features": 5}),
        ("max_10", {"max_features": 10}),
        ("mean",   {}),
    ]:
        p = build_pipeline(**kw)   # probability=False — no Platt scaling
        p.fit(X_tr, y_tr)
        y_score = p.decision_function(X_te)
        auc     = roc_auc_score(y_te, y_score)
        results[label] = {"test_auc": round(auc, 3)}
        log.info(f"  {label:7s}: test AUC = {auc:.3f}")
    return results


# ── Training ───────────────────────────────────────────────────────────────────

def train(X_tr: pd.DataFrame, y_tr: pd.Series,
          X_te: pd.DataFrame, y_te: pd.Series):
    # Final model: probability=True so the API can call predict_proba
    pipeline = build_pipeline(probability=True)
    log.info(
        "\nFitting pipeline: "
        "NaNDropper → Winsorizer → CorrelationFilter → StandardScaler "
        "→ ElasticNet selector → Linear SVM"
    )
    pipeline.fit(X_tr, y_tr)

    y_score = pipeline.decision_function(X_te)
    y_pred  = pipeline.predict(X_te)
    auc     = roc_auc_score(y_te, y_score)
    ci_lo, ci_hi = bootstrap_auc_ci(y_te, y_score)

    n_kept  = len(pipeline.named_steps["nan_dropper"].cols_to_keep_)
    n_clean = len(pipeline.named_steps["corr_filter"].cols_to_keep_)
    n_sel   = int(pipeline.named_steps["selector"].get_support().sum())

    log.info(f"\nTest AUC: {auc:.3f}  (95% CI {ci_lo:.3f}–{ci_hi:.3f})")
    log.info(f"Features after NaN/constant drop:  {n_kept}")
    log.info(f"Features after correlation filter: {n_clean}")
    log.info(f"Features selected by Elastic Net:  {n_sel}")
    log.info(
        f"\n{classification_report(y_te, y_pred, target_names=['Nodule-absent','Nodule-present'])}"
    )

    X_all = pd.concat([X_tr, X_te]).reset_index(drop=True)
    y_all = pd.concat([y_tr, y_te]).reset_index(drop=True)

    # CV pipeline: probability=False skips Platt scaling on every fold
    cv_pipeline = build_pipeline(probability=False)
    cv          = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores   = cross_val_score(cv_pipeline, X_all, y_all, cv=cv, scoring="roc_auc")
    log.info(f"5-fold CV AUC (StratifiedKFold): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    return pipeline, auc, ci_lo, ci_hi, cv_scores


# ── Feature importance from SVC coefficients ──────────────────────────────────

def svc_feature_importance(pipeline) -> dict:
    selected_names = pipeline.named_steps["selector"].get_feature_names_out()
    coefs          = pipeline.named_steps["clf"].coef_[0]
    return {str(name): float(coef) for name, coef in zip(selected_names, coefs)}


# ── SHAP (optional) ───────────────────────────────────────────────────────────

def compute_shap(pipeline, X_tr: pd.DataFrame, X_te: pd.DataFrame):
    try:
        import shap
    except ImportError:
        log.warning("shap not installed — skipping. pip install shap")
        return None

    log.info("Computing SHAP values…")
    cols       = list(X_tr.columns)
    bg         = shap.kmeans(X_tr.sample(min(50, len(X_tr)), random_state=42), 10)
    predict_fn = lambda x: pipeline.predict_proba(pd.DataFrame(x, columns=cols))[:, 1]
    explainer  = shap.KernelExplainer(predict_fn, bg)
    sample     = X_te.head(min(20, len(X_te)))
    sv         = explainer.shap_values(sample, nsamples=100)
    return explainer, sv, sample


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit",     type=int, default=None)
    parser.add_argument("--skip-shap", action="store_true")
    parser.add_argument("--demo",      action="store_true")
    parser.add_argument("--baselines", action="store_true",
                        help="Compute lung-volume and emphysema-fraction baselines "
                             "(requires reading all CT images; results cached)")
    args = parser.parse_args()

    # 1. Features
    df = extract_all_features(limit=args.limit)
    if df.empty:
        log.error("No features extracted. Check subset directories and seg-lungs-LUNA16/.")
        return

    # 2. Labels
    df = create_labels(df, demo=args.demo)

    series_uids = df["series_uid"].reset_index(drop=True)
    y           = df["label"].reset_index(drop=True)
    X_raw       = df.drop(columns=["series_uid", "label"]).reset_index(drop=True)

    if y.nunique() < 2:
        log.error("Only one class in labels — cannot train.")
        return

    log.info(f"\nDataset: {X_raw.shape[0]} samples × {X_raw.shape[1]} raw features")

    # 3. Train/test split — cleaning is inside the pipeline (no leakage)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_raw, y, test_size=0.2, random_state=42, stratify=y
    )
    # Track which series UIDs went to the test set (for baseline evaluation)
    series_uids_te = series_uids[X_te.index].reset_index(drop=True)
    y_te_reset     = y_te.reset_index(drop=True)

    # 4. Train
    pipeline, test_auc, ci_lo, ci_hi, cv_scores = train(X_tr, y_tr, X_te, y_te)

    # 5. Spacing confound check (raw AUC — NOT flipped)
    spacing_aucs = check_spacing_confound(series_uids, y)

    # 6. GroupKFold CV on binned spacing_z
    group_cv = run_spacing_group_cv(X_raw, y, series_uids)

    # 7. Baselines on the SAME test split as the model (n=36)
    baseline_aucs = {}
    if args.baselines:
        measurements  = compute_baseline_measurements(series_uids)
        baseline_aucs = evaluate_baselines(measurements, series_uids_te, y_te_reset)

    # 8. Random-label control (probability=False — no Platt scaling)
    log.info("\nRandom-label control (shuffled y)…")
    rng        = np.random.RandomState(42)
    y_shuffled = pd.Series(rng.permutation(y.values), index=y.index)
    ctrl_cv    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    ctrl_scores = cross_val_score(
        build_pipeline(), X_raw, y_shuffled, cv=ctrl_cv, scoring="roc_auc"
    )
    log.info(f"  Random-label CV AUC: {ctrl_scores.mean():.3f} ± {ctrl_scores.std():.3f}")

    # 9. Feature-count comparison on same train/test split as main model
    feat_cmp = compare_feature_counts(X_tr, y_tr, X_te, y_te)

    # 10. Feature importance
    fi_dict = svc_feature_importance(pipeline)
    top5    = dict(sorted(fi_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:5])
    log.info(
        "\nTop-5 features by SVC coefficient:\n" +
        "\n".join(f"  {k}: {v:+.4f}" for k, v in top5.items())
    )

    # 11. Optional SHAP
    if not args.skip_shap and len(X_te) >= 5:
        shap_result = compute_shap(pipeline, X_tr, X_te)
        if shap_result:
            explainer, sv, sample = shap_result
            joblib.dump(explainer, os.path.join(MODEL_DIR, "shap_explainer.pkl"))
            np.save(os.path.join(MODEL_DIR, "shap_values.npy"), sv)
            sample.to_csv(os.path.join(MODEL_DIR, "shap_sample.csv"), index=False)

    # 12. Save artifacts
    feature_names = list(X_raw.columns)
    joblib.dump(pipeline, os.path.join(MODEL_DIR, "lung_screener.pkl"))

    with open(os.path.join(MODEL_DIR, "feature_names.json"), "w") as f:
        json.dump(feature_names, f)

    with open(os.path.join(MODEL_DIR, "feature_importance.json"), "w") as f:
        json.dump(fi_dict, f, indent=2)

    stats = {
        "test_auc":                  round(test_auc, 4),
        "test_auc_ci_lo":            round(ci_lo,    4),
        "test_auc_ci_hi":            round(ci_hi,    4),
        "cv_auc_mean":               round(float(cv_scores.mean()), 4),
        "cv_auc_std":                round(float(cv_scores.std()),  4),
        "n_train":                   len(X_tr),
        "n_test":                    len(X_te),
        "n_features_raw":            X_raw.shape[1],
        "n_features_selected":       int(pipeline.named_steps["selector"].get_support().sum()),
        "random_label_cv_auc":       round(float(ctrl_scores.mean()), 4),
        "random_label_cv_std":       round(float(ctrl_scores.std()),  4),
        "spacing_confound":          spacing_aucs,
        "group_cv":                  group_cv,
        "baseline_aucs":             baseline_aucs,
        "feature_count_comparison":  feat_cmp,
    }
    with open(os.path.join(MODEL_DIR, "training_stats.json"), "w") as f:
        json.dump(stats, f, indent=2)

    log.info(f"\n{'='*60}")
    log.info(f"  Model saved    →  model/lung_screener.pkl")
    log.info(f"  Test AUC       →  {test_auc:.3f}  (95% CI {ci_lo:.3f}–{ci_hi:.3f})")
    log.info(f"  CV AUC (strat) →  {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    if group_cv:
        log.info(
            f"  CV AUC (group) →  {group_cv['mean']:.3f} ± {group_cv['std']:.3f}"
            f"  ({group_cv['n_bins']} spacing bins)"
        )
    log.info(f"  Random-label   →  {ctrl_scores.mean():.3f} ± {ctrl_scores.std():.3f}")
    log.info(f"{'='*60}")
    log.info("Next steps:")
    log.info("  API  →  python -m uvicorn app.main:app --reload")
    log.info("  UI   →  python -m streamlit run frontend/app.py")


if __name__ == "__main__":
    main()
