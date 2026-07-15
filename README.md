# Lung Disease Risk Screener

An end-to-end medical imaging ML pipeline that classifies CT scans as high or low risk for lung disease using radiomic texture features extracted from LUNA16 CT data.

> **Research tool only — not for clinical use.**

---

## Demo

Upload a CT scan → auto-segment lungs → extract 75 radiomic features → predict risk score → explain which features drove the decision.

```
API  →  http://localhost:8000/docs
UI   →  http://localhost:8501
```

---

## Results

Trained on **LUNA16 subsets 1–2** (178 CT scans, real nodule annotations).

| Metric | Value |
|---|---|
| Test AUC | 0.767 |
| 5-Fold CV AUC | 0.708 ± 0.095 |
| Features extracted | 75 |
| Features after correlation filter | 29 |
| Features selected by Elastic Net | 14 |
| Training samples | 142 |
| Test samples | 36 |
| Label split | ~56 low risk / ~122 high risk |

Adding subset2 (89 → 178 scans) raised CV AUC from 0.601 to 0.708 and cut variance by nearly half (±0.187 → ±0.095), confirming the model is learning a real signal.

**Top 5 features by SVC coefficient:**

| Feature | Contribution |
|---|---|
| `gldm_DependenceEntropy` | +1.558 (→ high risk) |
| `glszm_GrayLevelNonUniformity` | +1.262 (→ high risk) |
| `glcm_SumEntropy` | −1.094 (→ low risk) |
| `glszm_HighGrayLevelZoneEmphasis` | +0.818 (→ high risk) |
| `glszm_LargeAreaHighGrayLevelEmphasis` | −0.765 (→ low risk) |

The CV AUC of 0.708 uses 2 of 10 available subsets (~178 of ~888 total scans). Adding the remaining subsets is the single highest-leverage improvement available.

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
  GLCM  (25)  Grey-level co-occurrence matrix
  GLRLM (16)  Run-length matrix
  GLSZM (16)  Size-zone matrix
  GLDM  (16)  Dependence matrix
  NGTDM ( 5)  Neighbourhood grey-tone difference matrix
    │
    ▼
Data Cleaning
  Winsorize ±2 SD         ← tame extreme values without losing samples
  Drop correlated (r>0.90)← 75 → 27 features
    │
    ▼
Elastic Net               ← sparse feature selection (27 → 10 features)
    │
    ▼
Linear SVM                ← binary classifier
    │
    ▼
Risk Score + Top Features ← 0–1 score, SVC coefficient contributions
```

---

## Methodology

**Feature extraction** uses PyRadiomics with 1mm isotropic resampling and binWidth=25. Five texture matrix classes are extracted: GLCM, GLRLM, GLSZM, GLDM, and NGTDM (75 features total).

**Data cleaning**: outlier values are winsorized to ±2 SD per feature, then highly correlated features (Pearson r > 0.90) are removed to reduce redundancy before model fitting.

**Feature selection** uses Elastic Net regression (α=0.01, L1 ratio=0.5) via `SelectFromModel`, treating the binary labels as continuous targets. The L1 penalty drives most coefficients to zero, selecting a sparse, interpretable feature subset.

**Classification** uses a linear SVM trained on the Elastic Net-selected features. The SVM's linear decision boundary pairs well with the Elastic Net's pre-selected, low-redundancy features.

**Labels** are derived from LUNA16 `annotations.csv`: any scan with at least one confirmed nodule annotation is labelled high risk (1), otherwise low risk (0).

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

Feature extraction takes ~30–60 minutes on first run and is cached to `features_cache.csv`. Every subsequent run loads from cache and completes in seconds.

To test the pipeline before downloading real data:

```bash
python Main.py --demo --limit 10
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

Open `http://localhost:8501`, upload a `.mhd` or `.nrrd` CT scan, and get a risk score.

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
  "risk_score": 0.742,
  "risk_level": "High",
  "top_features": {
    "glszm·HighGrayLevelZoneEmphasis": 0.312,
    "glcm·ClusterTendency": -0.198,
    "ngtdm·Coarseness": 0.167
  },
  "methodology": "Elastic Net feature selection + Linear SVM"
}
```

### `GET /stats`

Returns training metrics (AUC, feature counts, sample counts).

### `GET /health`

Liveness probe — returns `{"status": "ok"}`.

---

## Training Pipeline Details

`Main.py` accepts the following flags:

| Flag | Description |
|---|---|
| `--limit N` | Process only N scans (development speed-up) |
| `--skip-shap` | Skip SHAP computation (saves ~15 min) |
| `--demo` | Use synthetic labels if no `annotations.csv` found |

Feature extraction results are cached to `features_cache.csv`. Delete this file to force re-extraction (e.g. after changing extractor parameters).

---

## What I Would Do Next

- **More data** — add subsets 0 and 2–9 (~800 additional scans). This is the single change most likely to improve CV AUC.
- **Nodule-specific ROIs** — extract features from spherical masks around each annotated nodule rather than the whole lung, closer to the paper's actual methodology.
- **GLDZM features** — the grey-level distance-zone matrix is not currently enabled; PyRadiomics supports it and adding it may improve AUC.
- **Multi-site validation** — test on LIDC-IDRI to assess generalisation across scanners and acquisition protocols.
- **Longitudinal model** — given two scans of the same patient, predict progression rather than point-in-time risk.

---

## Dataset

**LUNA16** (LUng Nodule Analysis 2016)  
888 CT scans from the LIDC-IDRI collection with standardised nodule annotations.  
Available at [luna16.grand-challenge.org](https://luna16.grand-challenge.org) — free registration required.

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
