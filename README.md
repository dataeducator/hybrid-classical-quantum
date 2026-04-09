# Hybrid Quantum-Classical Neural Networks for TNBC Survival Prediction

A fairness-aware hybrid quantum-classical neural network for binary survival prediction in triple-negative breast cancer (TNBC), with systematic subgroup auditing across demographic groups.

**MSDS 730: Deep Learning | Spring 2026 | Meharry Medical College**

**Authors:** Jaclyn Claiborne and Tenicka Norwood

---

## Overview

This project investigates whether hybrid quantum-classical neural networks can improve survival prediction for TNBC patients while maintaining fairness across racial subgroups. We combine a 7-qubit variational quantum circuit (VQC) implemented in PennyLane with a classical fusion MLP in PyTorch, and benchmark against classical MLP, XGBoost, and LightGBM baselines.

Key contributions:
- Hybrid quantum-classical architecture for real-world clinical survival prediction
- Systematic ablation study comparing 7 model variants
- Automated data leakage audit (3-part: AUC scan, correlation, KS test)
- Fairness audit with demographic parity, equalized odds, and FPR parity metrics

## Architecture

![Architecture](figures/hybrid_architecture.pdf)

*HybridRealQ: 7 clinical features encoded via RY angle encoding into a variational quantum circuit with ring CNOT entanglement. The quantum scalar output is concatenated with classical features and processed by a fusion MLP.*

---

## Results

### Ablation Study

| Model | AUC | Precision | Recall | F1 | Time (s) |
|-------|-----|-----------|--------|-----|----------|
| HybridRealQ (7-qubit) | 0.7037 | 0.6299 | 0.5272 | 0.5740 | 200.1 |
| Classical MLP | 0.7942 | 0.5821 | 0.8478 | 0.6903 | <1 |
| Quantum Only | 0.7942 | 0.5839 | 0.9837 | 0.7328 | 194.5 |
| Hybrid (3-qubit) | 0.7271 | 0.5087 | 0.7935 | 0.6200 | 121.9 |
| Hybrid (Deep 2-layer) | 0.7899 | 0.6526 | 0.7554 | 0.7003 | 293.0 |
| **XGBoost** | **0.9051** | 0.6885 | 0.9728 | 0.8063 | 0.2 |
| **LightGBM** | 0.9006 | **0.7178** | 0.9402 | **0.8141** | 1.6 |

### Fairness Audit

| Metric | Value |
|--------|-------|
| Demographic Parity Difference | 0.2794 |
| Equal Opportunity (TPR gap) | 0.5734 |
| FPR Parity Difference | 0.1039 |

---

## Data

- **Source:** [SEER TNBC Registry](https://seer.cancer.gov/data/)
- **Size:** 76,846 raw records; 60,412 after preprocessing
- **Features:** 7 quantum + 16 classical
- **Target:** Binary 60-month survival

> Real patient data is excluded per SEER data use agreement. See `DATA_COMPLIANCE.md`.

---

## Quick Start

```bash
# Install dependencies
pip install torch pennylane xgboost lightgbm scikit-learn pandas numpy scipy

# Run experiments (leakage audit + training + ablation + fairness)
python run_experiments.py
```

## Project Structure

```
.
├── main.tex / main.pdf               # Final paper (LaTeX)
├── run_experiments.py                 # Standalone experiment pipeline
├── experiment_results.json            # Results (JSON)
├── TNBC_Quantum_Ablation_Results.csv  # Ablation table
├── figures/
│   ├── hybrid_architecture.tex        # TikZ architecture diagram
│   ├── hybrid_architecture.pdf        # Compiled figure
│   └── *.png                          # Result visualizations
├── codes/
│   ├── 1_data_harmonization/          # ETL pipelines
│   ├── 2_feature_engineering/         # Feature extraction
│   ├── 5_explainability/              # SHAP, LIME, Integrated Gradients
│   ├── 6_survival_modeling/           # Model architectures and training
│   ├── 7_disparity_analysis/          # Fairness metrics and subgroup analysis
│   └── tests/                         # Automated tests
├── *.ipynb                            # Jupyter notebook (full implementation)
└── DATA_COMPLIANCE.md                 # Data use compliance
```

## Hardware

- **GPU:** NVIDIA GeForce RTX 5080 (17 GB)
- **PyTorch:** 2.10.0 (CUDA 12.8)
- **PennyLane:** 0.44.1 (lightning.qubit)

---

**License:** [PolyForm Noncommercial 1.0.0](LICENSE.md)
