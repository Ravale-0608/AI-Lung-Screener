# Lung Nodule Screener

An end-to-end medical imaging ML pipeline that classifies CT scans as nodule-present or nodule-absent using radiomic texture features extracted from LUNA16 CT data.

> **Research tool only — not for clinical use.**

---

## Demo

Upload a CT scan → auto-segment lungs → extract 75 radiomic features → predict nodule probability → explain which features drove the decision.

```
API  →  http://localhost:8000/docs
UI   →  http://localhost:8501
```

---

## Results

Trained on **LUNA16 subsets 1–2** (178 CT scans, real nodule annotations).

| Metric | Value |
|---|---|
| Test AUC | 0.767 (before leakage fix) |
| 5-Fold CV AUC | 0.708 ± 0.095 (before leakage fix) |
| Features extracted | 75 |
| Training samples | 142 |
| Test samples | 36 |
| Label split | ~56 nodule-absent / ~122 nodule-present |

> **Note:** These are pre-fix numbers. The leakage fix (moving winsorization and correlation filtering inside the sklearn Pipeline so they are fitted on training data only) is expected to lower these metrics — that is the correct outcome, not a regression.

Adding subset2 (89 → 178 scans) raised CV AUC from 0.601 to 0.708 and cut variance by nearly half (±0.187 → ±0.095). This is consistent with more data reducing estimation noise, but does not rule out an acquisition confound also becoming easier to learn — see the confound check results in `model/training_stats.json`.

---

## How It Works

```
CT Scan Upload
    │
    ▼
lungmask (U-Net)          ← auto-segments lungs, no pre-computed mask needed
    │
    ▼
PyRadiomics               ← extracts 75 texture features
  GLCM  (24)  Grey-level co-occurrence matrix
  GLRLM (16)  Run-length matrix
  GLSZM (16)  Size-zone matrix
  GLDM  (14)  Dependence matrix
  NGTDM ( 5)  Neighbourhood grey-tone difference matrix
    │
    ▼ ← all steps below are fitted on the training split only (no leakage)
sklearn Pipeline
  NaN/constant drop        ← remove degenerate features
  Winsorise ±2 SD          ← tame extreme values without losing samples
  Drop correlated (r>0.90) ← reduce redundancy
  StandardScaler           ← zero-mean unit-variance
    │
    ▼
Elastic Net               ← sparse feature selection
    │
    ▼
Linear SVM                ← binary classifier (nodule-present vs. nodule-absent)
    │
    ▼
Nodule Probability + Top Features
```

---

## Methodology

**Feature extraction** uses PyRadiomics with 1 mm isotropic resampling and binWidth=25. Five texture matrix classes are extracted: GLCM, GLRLM, GLSZM, GLDM, and NGTDM (75 features total).

**Data cleaning**: outlier values are winsorised to ±2 SD per feature (fitted on the training split only), then highly correlated features (Pearson r > 0.90) are removed to reduce redundancy. Both steps live inside the sklearn Pipeline to prevent test-set leakage.

**Feature selection** uses Elastic Net regression (α=0.01, L1 ratio=0.5) via `SelectFromModel`, treating the binary labels as continuous targets. The L1 penalty drives most coefficients to zero, selecting a sparse, interpretable feature subset.

**Classification** uses a linear SVM trained on the Elastic Net-selected features.

**Labels** are derived from LUNA16 `annotations.csv`: any scan with at least one confirmed nodule annotation is labelled nodule-present (1), otherwise nodule-absent (0). This is a nodule-detection task trained on LUNA16 data — it is not a reproduction of any published spirometry or COPD cohort study.

**AUC** is computed from the SVM's `decision_function` (not `predict_proba`/Platt scaling) on the held-out test set, with 1000-bootstrap 95% CI. `predict_proba` is retained for the API's probability output.

---

## Project Structure

```
lung-screener/
  ├── Main.py                 # Training pipeline — run this first
  ├── app/
  │   └── main.py             # FastAPI backend — serves predictions
  ├── frontend/
  │   └── app.py              # Streamlit UI
  ├── model/                  # Saved after training (gitignored)
  │   ├── lung_screener.pkl
  │   ├── feature_names.json
  │   ├── feature_importance.json
  │   └── training_stats.json
  ├── subset1/                # LUNA16 CT scans (gitignored)
  ├── seg-lungs-LUNA16/       # Lung segmentation masks (gitignored)
  ├── Dockerfile              # API image
  ├── frontend/Dockerfile     # Frontend image
  ├── docker-compose.yml
  └── requirements.txt
```

---

## Setup

**1. Install dependencies**

```bash
pip install -r requirements.txt

# PyTorch (CPU) + lungmask for inference auto-segmentation
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install lungmask
```

**2. Get the data**

- Download LUNA16 from [luna16.grand-challenge.org](https://luna16.grand-challenge.org) (free registration)
- Place CT scans in `subset1/` and masks in `seg-lungs-LUNA16/`
- Download `annotations.csv` from the same page and place it in the project root

**3. Train the model**

```bash
python Main.py --skip-shap
```

Feature extraction takes ~30–60 minutes on first run and is cached to `features_cache_<hash>.csv`. Every subsequent run loads from cache and completes in seconds.

To test the pipeline before downloading real data:

```bash
python Main.py --demo --limit 10
```

To also compute lung-volume and emphysema-fraction baselines (requires reading all CT images; cached after first run):

```bash
python Main.py --skip-shap --baselines
```

---

## Running Locally

**API** (terminal 1):

```bash
python -m uvicorn app.main:app --reload
```

**UI** (terminal 2):

```bash
python -m streamlit run frontend/app.py
```

Open `http://localhost:8501`, upload a `.mhd` or `.nrrd` CT scan, and get a nodule probability.

---

## Running with Docker

```bash
docker-compose up --build
```

- UI → `http://localhost:8501`
- API docs → `http://localhost:8000/docs`

The `model/` directory is mounted as a volume so you do not need to rebuild the image after retraining.

---

## API Reference

### `POST /predict`

Upload a CT scan file (`.mhd`, `.nrrd`, `.nii`, `.nii.gz`).

**Response:**

```json
{
  "nodule_probability": 0.742,
  "nodule_likelihood": "High",
  "top_features": {
    "glszm·HighGrayLevelZoneEmphasis": 0.312,
    "glcm·ClusterTendency": -0.198,
    "ngtdm·Coarseness": 0.167
  },
  "methodology": "Elastic Net feature selection + Linear SVM (LUNA16)"
}
```

### `GET /stats`

Returns training metrics (AUC, bootstrap CI, feature counts, sample counts, confound check results).

### `GET /health`

Liveness probe — returns `{"status": "ok"}`.

---

## Training Pipeline Flags

`Main.py` accepts the following flags:

| Flag | Description |
|---|---|
| `--limit N` | Process only N scans (development speed-up) |
| `--skip-shap` | Skip SHAP computation |
| `--demo` | Use synthetic labels if no `annotations.csv` found |
| `--baselines` | Compute lung-volume and emphysema-fraction baselines (slow first run, cached after) |

Feature extraction results are cached to `features_cache_<params-hash>.csv`. Changing `EXTRACTOR_PARAMS` in `Main.py` automatically produces a new cache file; the old file is preserved.

---

## What I Would Do Next

- **More data** — add subsets 0 and 2–9 (~700 additional scans). This is the single change most likely to improve CV AUC.
- **Nodule-specific ROIs** — extract features from spherical masks around each annotated nodule rather than the whole lung, which is more consistent with nodule-characterisation literature.
- **GroupKFold on slice thickness** — if spacing_z AUC > 0.65, switch the CV split to GroupKFold on binned slice thickness to guard against acquisition confounds.
- **Multi-site validation** — test on LIDC-IDRI to assess generalisation across scanners and acquisition protocols.
- **Longitudinal model** — given two scans of the same patient, predict progression rather than point-in-time nodule presence.

---

## Dataset

**LUNA16** (LUng Nodule Analysis 2016)
888 CT scans from the LIDC-IDRI collection with standardised nodule annotations.
Available at [luna16.grand-challenge.org](https://luna16.grand-challenge.org) — free registration required.

**Important label semantics:** LUNA16 is a nodule *detection* benchmark. The label `1` means "this scan contains at least one confirmed pulmonary nodule ≥3 mm," not "this patient has cancer." LUNA16 carries no malignancy ratings. For malignancy scores, see the source LIDC-IDRI dataset, which includes radiologist malignancy ratings (1–5) per nodule.

LUNA16 subsets are random cross-validation splits, not acquisition sites.

The data is not included in this repository and must be downloaded separately. See the LUNA16 challenge page for terms of use.

---

## Stack

| Component | Library |
|---|---|
| Feature extraction | PyRadiomics 3.0, SimpleITK |
| Auto-segmentation | lungmask (JoHof R231 U-Net) |
| Feature selection | Elastic Net — scikit-learn |
| Classifier | Linear SVM — scikit-learn |
| Explainability | SVC coefficients / SHAP KernelExplainer |
| API | FastAPI + Uvicorn |
| UI | Streamlit |
| Containerisation | Docker + Docker Compose |
