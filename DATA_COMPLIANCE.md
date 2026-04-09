# Data Compliance & Usage

This document describes how each data source is used in this project and
confirms compliance with all applicable data use agreements.

## What This Repository Contains

- **Model code only** (neural network definitions, training scripts)
- **Aggregate results** (AUC, fairness metrics, ablation tables)
- **No individual patient records** are included in this repository

## Data Source Compliance

### SEER (Surveillance, Epidemiology, and End Results)

- **Source**: NCI SEER Program (https://seer.cancer.gov/)
- **Access**: SEER*Stat public-use research data
- **Agreement**: SEER Research Data Agreement signed
- **What we use**: 76,846 TNBC patient records for model training (local only)
- **Not included**: Individual patient records are excluded via `.gitignore`
- **Citation**: Surveillance, Epidemiology, and End Results (SEER) Program
  (www.seer.cancer.gov) SEER*Stat Database.

## Data Handling Checklist

- [x] No SEER individual-level CSV files are in this repository
- [x] `.gitignore` excludes real data from version control
- [x] Only aggregate metrics and model performance results are shared

## License

PolyForm Noncommercial License 1.0.0
Copyright 2026 Tenicka Norwood, Jaclyn Claiborne.
