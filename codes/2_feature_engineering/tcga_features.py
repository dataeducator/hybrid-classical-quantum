"""Feature engineering for TCGA-BRCA genomic data.

Data-lineage:
  TCGA patient_clinical.csv + mutations.csv
    -> genomic features (mutation count, TMB, fraction genome altered,
       key gene mutation flags, OS/DFS survival)

These features complement the SEER tabular and MIMIC temporal features
in the multi-tower fusion architecture.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl


# Key TNBC driver genes (from TCGA literature + our cBioPortal analysis)
TNBC_GENES = [
    "TP53", "PIK3CA", "BRCA1", "BRCA2", "RB1", "PTEN",
    "MYC", "EGFR", "CDKN2A", "NF1",
]


@dataclass
class TCGAFeatures:
    """Genomic feature tensors for TCGA-BRCA TNBC patients."""
    X: np.ndarray               # (N, d_features) genomic features
    time: np.ndarray            # (N,) survival time in months
    event: np.ndarray           # (N,) event indicator (1=deceased)
    patient_ids: list[str]      # (N,) patient IDs
    feature_names: list[str]    # column names
    race_labels: np.ndarray     # (N,) race codes for subgroup audit


def extract_tcga_features(
    patient_df: pl.DataFrame,
    mutations_df: pl.DataFrame | None = None,
) -> TCGAFeatures:
    """Extract genomic features from TCGA-BRCA clinical + mutation data.

    Parameters
    ----------
    patient_df : pl.DataFrame
        Patient-level clinical data (from tcga_etl.extract or CSV).
    mutations_df : pl.DataFrame, optional
        Mutation data (from cBioPortal). If None, gene flags are all 0.

    Returns
    -------
    TCGAFeatures with arrays ready for model input.
    """
    n = patient_df.height

    # ── Patient IDs ──
    pids = patient_df["patient_id"].to_list()

    # ── Demographics ──
    if "AGE" in patient_df.columns:
        age = patient_df["AGE"].cast(pl.Float32, strict=False).fill_null(55).to_numpy()
    else:
        age = np.full(n, 55, dtype=np.float32)

    if "RACE" in patient_df.columns:
        race_raw = patient_df["RACE"].fill_null("OTHER").to_list()
        race_map = {
            "WHITE": 0, "BLACK OR AFRICAN AMERICAN": 1,
            "ASIAN": 2, "OTHER": 4,
        }
        race_codes = np.array([race_map.get(str(r).upper(), 4) for r in race_raw], dtype=np.int32)
    else:
        race_codes = np.full(n, 4, dtype=np.int32)

    # ── Genomic features ──
    # Mutation count / TMB
    if "MUTATION_COUNT" in patient_df.columns:
        mut_count = patient_df["MUTATION_COUNT"].cast(pl.Float32, strict=False).fill_null(0).to_numpy()
    else:
        mut_count = np.zeros(n, dtype=np.float32)

    # Fraction genome altered (CNA)
    if "FRACTION_GENOME_ALTERED" in patient_df.columns:
        fga = patient_df["FRACTION_GENOME_ALTERED"].cast(pl.Float32, strict=False).fill_null(0).to_numpy()
    else:
        fga = np.zeros(n, dtype=np.float32)

    # Stage ordinal
    if "AJCC_PATHOLOGIC_TUMOR_STAGE" in patient_df.columns:
        stage_raw = patient_df["AJCC_PATHOLOGIC_TUMOR_STAGE"].fill_null("").to_list()
        stage_map = {"STAGE I": 1, "STAGE IA": 1, "STAGE IB": 1,
                     "STAGE II": 2, "STAGE IIA": 2, "STAGE IIB": 2,
                     "STAGE III": 3, "STAGE IIIA": 3, "STAGE IIIB": 3, "STAGE IIIC": 3,
                     "STAGE IV": 4}
        stage = np.array([stage_map.get(str(s).upper().strip(), 0) for s in stage_raw], dtype=np.float32)
    else:
        stage = np.zeros(n, dtype=np.float32)

    # ── Per-gene mutation flags ──
    gene_flags = np.zeros((n, len(TNBC_GENES)), dtype=np.float32)
    if mutations_df is not None and mutations_df.height > 0:
        pid_to_idx = {pid: i for i, pid in enumerate(pids)}
        gene_col = "gene" if "gene" in mutations_df.columns else "hugoGeneSymbol"
        pid_col = "patientId" if "patientId" in mutations_df.columns else "patient_id"

        if gene_col in mutations_df.columns and pid_col in mutations_df.columns:
            for row in mutations_df.iter_rows(named=True):
                pid = row.get(pid_col, "")
                gene = row.get(gene_col, "")
                if pid in pid_to_idx and gene in TNBC_GENES:
                    gene_flags[pid_to_idx[pid], TNBC_GENES.index(gene)] = 1.0

    # ── Survival ──
    if "OS_MONTHS" in patient_df.columns:
        time_arr = patient_df["OS_MONTHS"].cast(pl.Float32, strict=False).fill_null(0).to_numpy()
    else:
        time_arr = np.zeros(n, dtype=np.float32)

    if "OS_STATUS" in patient_df.columns:
        os_raw = patient_df["OS_STATUS"].fill_null("").to_list()
        event = np.array([1 if "DECEASED" in str(s).upper() else 0 for s in os_raw], dtype=np.int32)
    else:
        event = np.zeros(n, dtype=np.int32)

    # ── Assemble feature matrix ──
    # Standardize continuous features
    age_z = (age - age.mean()) / (age.std() + 1e-8)
    mut_z = (mut_count - mut_count.mean()) / (mut_count.std() + 1e-8)
    fga_z = (fga - fga.mean()) / (fga.std() + 1e-8)

    feature_names = (
        ["age", "mutation_count", "fraction_genome_altered", "stage_ordinal"]
        + [f"gene_{g}" for g in TNBC_GENES]
    )

    X = np.column_stack([
        age_z,
        mut_z,
        fga_z,
        stage,
        gene_flags,
    ]).astype(np.float32)

    return TCGAFeatures(
        X=X,
        time=time_arr,
        event=event,
        patient_ids=pids,
        feature_names=feature_names,
        race_labels=race_codes,
    )


def extract_from_path(data_path: str | Path) -> TCGAFeatures:
    """Load TCGA CSVs and extract features."""
    p = Path(data_path)
    patient_df = pl.read_csv(p / "patient_clinical.csv", try_parse_dates=True,
                              infer_schema_length=10000, ignore_errors=True)
    mutations_df = None
    mut_path = p / "mutations.csv"
    if mut_path.exists():
        mutations_df = pl.read_csv(mut_path, try_parse_dates=True,
                                    infer_schema_length=10000, ignore_errors=True)
    return extract_tcga_features(patient_df, mutations_df)
