# Lung Disease Risk Screener

A end-to-end medical imaging ML pipeline that classifies CT scans as high or low risk for lung disease using radiomic texture features. Replicates and extends the methodology of **Kirby et al. 2023** — the same feature sets, the same data cleaning rationale, and the same winning model combination.

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

Trained on **LUNA16 subset1** (89 CT scans, real nodule annotations).

| Metric | Value |
|---|---|
| Test AUC | 0.639 |
| 5-Fold CV AUC | 0.601 ± 0.187 |
| Features extracted | 75 |
| Features after correlation filter | 27 |
| Features selected by Elastic Net | 10 |
| Training samples | 71 |
| Test samples | 18 |
| Label split | 28 low risk / 61 high risk |

**Top 5 features by SVC coefficient:**

| Feature | Contribution |
|---|---|
| `glszm_HighGrayLevelZoneEmphasis` | +1.806 (→ high risk) |
| `glcm_ClusterTendency` | −1.312 (→ low risk) |
| `glszm_GrayLevelNonUniformity` | +0.920 (→ high risk) |
| `glszm_ZoneEntropy` | +0.851 (→ high risk) |
| `ngtdm_Coarseness` | +0.833 (→ high risk) |

The CV AUC of 0.601 is honest — this is one of ten available subsets (~89 of ~888 total scans). Kirby et al. trained on the full dataset. Adding the remaining subsets is the single highest-leverage improvement available.

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
Data Cleaning             ← Kirby 2023 methodology
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

This project directly replicates **Kirby et al. 2023** (*Radiomics for COPD risk stratification from CT imaging*).

**Feature extraction** uses PyRadiomics with 1mm isotropic resampling and binWidth=25, matching the paper's preprocessing. The five texture matrix classes (GLCM, GLRLM, GLSZM, GLDM, NGTDM) are the same feature sets identified as most discriminative in the paper.

**Data cleaning** follows the paper's pipeline: outlier values are winsorized to ±2 SD per feature, then highly correlated features (Pearson r > 0.90) are removed to reduce redundancy before model fitting.

**Feature selection** uses Elastic Net regression (α=0.01, L1 ratio=0.5) via `SelectFromModel`, treating the binary labels as continuous targets. The L1 penalty drives most coefficients to zero, selecting a sparse, interpretable feature subset.

**Classification** uses a linear SVM trained on the Elastic Net-selected features. This is the combination the paper identifies as optimal — the SVM's linear decision boundary pairs well with the Elastic Net's pre-selected, low-redundancy features.

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
  "methodology": "Elastic Net feature selection + Linear SVM (Kirby et al. 2023)"
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
- **GLDZM features** — Kirby et al. report the grey-level distance-zone matrix alone achieves AUC 0.748. PyRadiomics supports it; it is not enabled here.
- **Multi-site validation** — test on LIDC-IDRI to assess generalisation across scanners and acquisition protocols.
- **Longitudinal model** — given two scans of the same patient, predict progression rather than point-in-time risk.

---

## Dataset

**LUNA16** (LUng Nodule Analysis 2016)  
888 CT scans from the LIDC-IDRI collection with standardised nodule annotations.  
Available at [luna16.grand-challenge.org](https://luna16.grand-challenge.org) — free registration required.

The data is not included in this repository and must be downloaded separately. See the LUNA16 challenge page for terms of use.

---

## Reference

Kirby, J. et al. (2023). *Radiomic features for COPD risk stratification in CT imaging.* The methodology implemented here — texture feature extraction, Elastic Net feature selection, and Linear SVM classification — replicates the winning pipeline described in that work.

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
