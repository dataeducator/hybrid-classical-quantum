# Reproducing the Results

This document describes how to reproduce all experiments and figures in the paper
"Hybrid Quantum-Classical Neural Networks for Personalized Survival Prediction in
Triple-Negative Breast Cancer."

Real SEER patient data cannot be redistributed under the SEER Research Data
Agreement. You must obtain your own SEER access to extract the cohort.

---

## 1. Prerequisites

### 1.1 SEER Data Access

1. Register for SEER access: https://seer.cancer.gov/data/
2. Sign the SEER Research Data Agreement (free, ~3-5 business days for approval).
3. Install SEER*Stat: https://seer.cancer.gov/seerstat/

### 1.2 Cohort Extraction

In SEER*Stat, build a case-listing session with the following filters:

- **Cancer site**: Breast (C50.0 - C50.9)
- **Behavior**: Malignant
- **First malignant primary indicator**: Yes
- **ER Status**: Negative
- **PR Status**: Negative
- **HER2 Status**: Negative

Export the following 23 variables to `tnbc.csv` and place it in the repository root:

```
Patient ID
Age recode (<60,60-69,70+)
Race recode (W, B, AI, API)
Year of diagnosis
Primary Site - labeled
ICD-O-3 Hist/behav
Grade Recode (thru 2017)
Grade Pathological (2018+)
Derived AJCC Stage Group, 7th ed (2010-2015)
Derived EOD 2018 Stage Group Recode (2018+)
ER Status Recode Breast Cancer (1990+)
PR Status Recode Breast Cancer (1990+)
Derived HER2 Recode (2010+)
CS tumor size (2004-2015)
Tumor Size Summary (2016+)
Laterality
RX Summ--Surg Prim Site (1998+)
Radiation recode
Chemotherapy recode (yes, no/unk)
Survival months
Vital status recode (study cutoff used)
Median household income inflation adj to 2023
Regional nodes positive (1988+)
```

Expected size: ~70,000-80,000 rows depending on the SEER cutoff date used.

---

## 2. Environment Setup

### 2.1 Python and Conda

This project requires Python 3.10 or newer.

```bash
git clone https://github.com/dataeducator/hybrid-classical-quantum
cd hybrid-classical-quantum
conda create -n tnbc python=3.11
conda activate tnbc
pip install -r requirements.txt
```

### 2.2 Pinned Dependencies

`requirements.txt` pins the exact versions used in the paper:

```
torch==2.10.0
pennylane==0.44.1
pennylane-lightning==0.44.0
xgboost==3.2.0
lightgbm==4.6.0
scikit-learn>=1.4
pandas>=2.0
numpy>=1.24
scipy>=1.11
lifelines==0.30.3
```

If you have an NVIDIA GPU, PyTorch will use it automatically for the classical
layers. The PennyLane `lightning.qubit` simulator runs on CPU regardless.

---

## 3. Data Preparation

Convert the raw SEER export into engineered features:

```bash
python preprocess_tnbc.py
```

This produces `breast_cancer_4quantum.csv` with derived columns:

- `numeric_age` (midpoint of age range)
- `years_since_2010`
- `income_encoded` (midpoint of income bracket)
- `income_age_ratio`
- `survival_60_months` (binary, used by `run_experiments.py`)
- `Tumor_Size_Summary_2016_` (backfilled from CS tumor size)
- `stage_cleaned` (derived in the experiment scripts at runtime)

The survival analysis scripts derive `_duration` and `_event` directly from raw
`Survival_months` and `Vital_status_recode_*` columns at runtime, applying the
right-censoring fix for the binary case (excluding alive patients with <60
months follow-up).

---

## 4. Running Experiments

### 4.1 Survival Analysis (Primary Results)

These three scripts produce the C-index numbers reported in the paper.

```bash
# Cox PH baseline + HybridSurvivalQ v1/v2/v3 (~40 min)
python run_survival_experiments.py

# v4: output scaling + classical pretrain + smaller init (~25 min)
python run_survival_v4.py

# Quantum residual learning: Cox + quantum correction (~30 min)
python run_survival_residual.py
```

### 4.2 Binary Classification (Ablation Comparison)

For comparison with the inflated leakage-contaminated baseline discussed in the
paper's discussion section:

```bash
python run_experiments.py   # ~40 min
```

### 4.3 Running in Parallel

If you have a multi-core machine, the experiment scripts can run in parallel
(each uses one CPU core for the quantum simulator):

```bash
python run_survival_v4.py > v4.log 2>&1 &
python run_survival_residual.py > residual.log 2>&1 &
wait
```

---

## 5. Generated Outputs

After running the scripts above, you will have:

```
breast_cancer_4quantum.csv          # Engineered features (do not commit; contains real data)
experiment_results.json             # Binary classification results
survival_results.json               # Cox + v1/v2/v3 survival results
survival_v4_results.json            # v4 results
survival_residual_results.json      # Quantum residual learning results
TNBC_Quantum_Ablation_Results.csv   # Binary ablation table
TNBC_Survival_Ablation_Results.csv  # Survival ablation table
```

---

## 6. Verifying Results

Compare your results files against the values reported in the paper:

| Model | Reported C-index | Tolerance |
|-------|------------------|-----------|
| Cox PH (lifelines, full data) | 0.73 | ±0.005 |
| HybridSurvivalQ_v1 | 0.59 | ±0.01 |
| HybridSurvivalQ_v2 | 0.58 | ±0.01 |
| HybridSurvivalQ_v3 | 0.59 | ±0.01 |
| HybridSurvivalQ_v3 (full data) | 0.67 | ±0.01 |
| HybridSurvivalQ_v4 | TBD | ±0.01 |
| Cox + Quantum Residual | TBD | ±0.005 |

> **Final v4 and Residual numbers will be filled in after the runs currently in
> progress complete. The values in the paper are the authoritative reference.**

### Expected Sources of Variance

Even with `random_state=42` set throughout, three sources of non-determinism
remain:

1. **PennyLane parameter-shift gradients** introduce ~1e-6 numerical noise.
2. **PyTorch CUDA non-determinism** for GPU operations (~1e-5).
3. **Library version sensitivity** (lifelines, scipy minor versions).

Combined, these typically produce ±0.005 C-index variance, occasionally up to
±0.01 for the smaller subsample experiments.

### Strict Reproduction

For bit-identical results:

```bash
# Force CPU-only PyTorch (eliminates CUDA non-determinism)
CUDA_VISIBLE_DEVICES="" python run_survival_residual.py
```

This adds ~5-10 minutes to runtime but guarantees deterministic output.

---

## 7. Hardware Requirements

- **Operating system**: Linux, macOS, or Windows
- **CPU**: 4+ cores recommended (PennyLane simulator is CPU-bound)
- **RAM**: 16 GB minimum, 32 GB recommended
- **GPU**: Optional (PyTorch will use it for classical layers)
- **Disk**: ~5 GB for SEER raw data + processed files
- **Total runtime**: ~2-3 hours for all experiments on a modern workstation

The reference hardware used to produce the paper's numbers:

```
GPU: NVIDIA GeForce RTX 5080 (17 GB)
CUDA: 12.8
PyTorch: 2.10.0
Platform: Windows 11
```

---

## 8. Troubleshooting

### "ModuleNotFoundError: No module named 'pennylane'"
You did not activate the conda environment. Run `conda activate tnbc`.

### "OMP: Error #15: Initializing libiomp5md.dll"
Set the environment variable before running:
```bash
# Linux/macOS
export KMP_DUPLICATE_LIB_OK=TRUE

# Windows
set KMP_DUPLICATE_LIB_OK=TRUE
```

### "ValueError: Input contains NaN"
Your SEER export has additional missing values not handled by the preprocessing
script. Check `breast_cancer_4quantum.csv` for unexpected NaN columns and either
drop them or extend the median-fill logic in the experiment scripts.

### Cox PH "ConvergenceWarning"
The `penalizer=0.01` argument should handle this, but if your cohort has highly
collinear features the warning may appear. The C-index should still match the
paper.

### Time budget reached early
Quantum models that finish under their `time_budget` parameter may have shorter
training than reported. Increase the budget in the script if needed:

```python
# In run_survival_v4.py
train_v4(..., time_budget=2400)  # Was 1200, now 40 minutes
```

### Different number of patients than reported
Your SEER cutoff date affects cohort size. Results scale with cohort size; expect
small (±0.005) differences if your cohort differs by <10% from ours.

---

## 9. Citing This Work

If you use this code or reproduce these results, please cite:

```
Claiborne J., Norwood T. (2026). Hybrid Quantum-Classical Neural Networks for
Personalized Survival Prediction in Triple-Negative Breast Cancer: A Fairness-
Aware, Agentic Modeling Framework. MSDS 730: Deep Learning, Meharry Medical
College.
```

---

## 10. Contact

For reproduction issues that aren't covered in this guide, open an issue at:
https://github.com/dataeducator/hybrid-classical-quantum/issues

Include:
- Your operating system and Python version
- Output of `pip list | grep -E "torch|pennylane|xgboost|lightgbm|lifelines"`
- The exact command that failed and its full error output
- Your `breast_cancer_4quantum.csv` shape (`df.shape` and `df.columns`)
