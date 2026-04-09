"""Extract tabular feature tensors from SEER TNBC data for survival modeling.

Data-lineage:
  SEER raw DataFrame (from seer_etl.extract)
    -> numeric feature matrix  (N x d_features)
    -> survival time array     (N,)
    -> event indicator array   (N,)
    -> subgroup labels         (N,)  for fairness audit

Features extracted (20 total):
  Demographics : age, race (5 one-hot), sex (binary)
  Clinical     : stage (ordinal 0-4), grade (ordinal 1-4), tumor_size_cm,
                 lymph_nodes_positive
  Treatment    : surgery_mastectomy, surgery_lumpectomy, radiation, chemo
  SES proxy    : median_income (standardized, 0 if unavailable)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


# Column order for the feature matrix
SEER_FEATURE_NAMES: list[str] = [
    "age",
    "race_white", "race_black", "race_asian", "race_hispanic", "race_other",
    "stage_ordinal",
    "grade_ordinal",
    "tumor_size_cm",
    "lymph_nodes_positive",
    "surgery_mastectomy",
    "surgery_lumpectomy",
    "radiation",
    "chemotherapy",
    "median_income",
]

_STAGE_ORDINAL: dict[str, int] = {
    "I": 1, "IA": 1, "IB": 1,
    "II": 2, "IIA": 2, "IIB": 2,
    "III": 3, "IIIA": 3, "IIIB": 3, "IIIC": 3,
    "IV": 4,
}


@dataclass
class SEERFeatures:
    """Container for SEER feature tensors."""
    X: np.ndarray              # (N, d_features) float32
    time: np.ndarray           # (N,) survival months float32
    event: np.ndarray          # (N,) 1=dead, 0=censored int32
    person_ids: list[str]      # (N,) patient identifiers
    race_labels: np.ndarray    # (N,) int codes for fairness audit
    feature_names: list[str]   # column names matching X columns


def extract_seer_features(raw: pl.DataFrame) -> SEERFeatures:
    """Convert raw SEER DataFrame into numeric feature tensors.

    Parameters
    ----------
    raw : pl.DataFrame
        Output of ``seer_etl.extract()`` -- the normalised SEER DataFrame
        with columns: patient_id, age_at_diagnosis, race, sex, ajcc_stage,
        grade, tumor_size_mm, lymph_nodes_positive, surgery, radiation,
        chemotherapy, survival_months, vital_status, (optional) median_income.

    Returns
    -------
    SEERFeatures
        Dataclass with numpy arrays ready for PyTorch.
    """
    df = raw.clone()
    n = df.height

    # ---- Demographics ----
    age = df["age_at_diagnosis"].cast(pl.Float32).fill_null(60).to_numpy()

    race_raw = df["race"].fill_null("Other").to_list()
    race_map = {"White": 0, "Black": 1, "Asian": 2, "Hispanic": 3, "Other": 4}
    race_codes = np.array([race_map.get(r, 4) for r in race_raw], dtype=np.int32)
    race_onehot = np.zeros((n, 5), dtype=np.float32)
    race_onehot[np.arange(n), race_codes] = 1.0

    # ---- Clinical ----
    stage_raw = df["ajcc_stage"].fill_null("Unknown").to_list() if "ajcc_stage" in df.columns else ["Unknown"] * n
    stage_ord = np.array(
        [_STAGE_ORDINAL.get(str(s).strip(), 0) for s in stage_raw],
        dtype=np.float32,
    )

    grade_raw = df["grade"].fill_null(9).cast(pl.Int64).to_numpy()
    grade_ord = np.where(np.isin(grade_raw, [1, 2, 3, 4]), grade_raw, 0).astype(np.float32)

    tumor_mm = df["tumor_size_mm"].cast(pl.Float64).fill_null(0).to_numpy()
    tumor_cm = (tumor_mm / 10.0).astype(np.float32)

    ln_pos = (
        df["lymph_nodes_positive"].cast(pl.Float64).fill_null(0).to_numpy().astype(np.float32)
        if "lymph_nodes_positive" in df.columns
        else np.zeros(n, dtype=np.float32)
    )

    # ---- Treatment ----
    surgery = df["surgery"].fill_null("None").to_list()
    surg_mast = np.array([1.0 if s == "Mastectomy" else 0.0 for s in surgery], dtype=np.float32)
    surg_lump = np.array([1.0 if s == "Lumpectomy" else 0.0 for s in surgery], dtype=np.float32)

    radiation_raw = df["radiation"].fill_null("No").to_list()
    radiation_arr = np.array([1.0 if r == "Yes" else 0.0 for r in radiation_raw], dtype=np.float32)

    chemo_raw = df["chemotherapy"].fill_null("No").to_list()
    chemo_arr = np.array([1.0 if c == "Yes" else 0.0 for c in chemo_raw], dtype=np.float32)

    # ---- SES ----
    if "median_income" in df.columns:
        income = df["median_income"].cast(pl.Float64).fill_null(0).to_numpy().astype(np.float32)
        # Standardize to z-score
        mu, sigma = income.mean(), income.std()
        income = (income - mu) / (sigma + 1e-8) if sigma > 0 else income * 0
    else:
        income = np.zeros(n, dtype=np.float32)

    # ---- Assemble feature matrix ----
    X = np.column_stack([
        age,
        race_onehot,
        stage_ord,
        grade_ord,
        tumor_cm,
        ln_pos,
        surg_mast,
        surg_lump,
        radiation_arr,
        chemo_arr,
        income,
    ]).astype(np.float32)

    # Standardize continuous features (age, tumor_cm, ln_pos, income)
    # but leave binary/one-hot/ordinal as-is
    cont_cols = [0, 8, 9]  # age, tumor_cm, ln_pos
    for ci in cont_cols:
        mu = X[:, ci].mean()
        sigma = X[:, ci].std()
        if sigma > 0:
            X[:, ci] = (X[:, ci] - mu) / sigma

    # ---- Outcome ----
    time = df["survival_months"].cast(pl.Float32).fill_null(0).to_numpy()
    vs = df["vital_status"].fill_null("Alive").to_list()
    event = np.array([1 if v == "Dead" else 0 for v in vs], dtype=np.int32)

    person_ids = df["patient_id"].to_list()

    return SEERFeatures(
        X=X,
        time=time,
        event=event,
        person_ids=person_ids,
        race_labels=race_codes,
        feature_names=SEER_FEATURE_NAMES,
    )
