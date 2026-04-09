"""Extract temporal lab sequences and text embeddings from MIMIC OMOP tables.

Data-lineage:
  MIMIC OMOP tables (from mimic_etl.run)
    -> tabular features     (N x d_tab)   demographics + treatment
    -> lab sequences         (N x T x 7)  7 lab values over T time steps
    -> sequence masks        (N x T)      valid time-step indicators
    -> text embeddings       (N x 768)    ClinicalBERT [CLS] vectors
    -> survival time/event   (N,)         from observation_period + death

Temporal features (7 labs):
  WBC, Hemoglobin, Platelets, Creatinine, Albumin, ALP, LDH

Tabular features (9 total):
  age, race (5 one-hot), n_chemo_drugs, has_mastectomy_or_lumpectomy
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import polars as pl


LAB_NAMES: list[str] = [
    "WBC", "Hemoglobin", "Platelets", "Creatinine", "Albumin", "ALP", "LDH",
]

MIMIC_TAB_FEATURE_NAMES: list[str] = [
    "age",
    "race_white", "race_black", "race_asian", "race_hispanic", "race_other",
    "n_chemo_drugs",
    "has_surgery",
]

# Measurement source values used in MIMIC OMOP tables
_LAB_SOURCE_MAP: dict[str, int] = {name: i for i, name in enumerate(LAB_NAMES)}

# Reference ranges for z-score normalization
_LAB_MEANS: dict[str, float] = {
    "WBC": 7.5, "Hemoglobin": 12.0, "Platelets": 250.0,
    "Creatinine": 1.0, "Albumin": 3.5, "ALP": 70.0, "LDH": 200.0,
}
_LAB_STDS: dict[str, float] = {
    "WBC": 3.5, "Hemoglobin": 2.0, "Platelets": 80.0,
    "Creatinine": 0.5, "Albumin": 0.5, "ALP": 30.0, "LDH": 60.0,
}


@dataclass
class MIMICFeatures:
    """Container for MIMIC feature tensors."""
    X_tab: np.ndarray          # (N, d_tab) float32 tabular features
    X_seq: np.ndarray          # (N, T, 7) float32 lab sequences
    seq_mask: np.ndarray       # (N, T) bool valid time-step mask
    X_text: np.ndarray         # (N, 768) float32 text embeddings
    time: np.ndarray           # (N,) survival days float32
    event: np.ndarray          # (N,) 1=dead, 0=censored int32
    person_ids: list[str]      # (N,) patient identifiers
    race_labels: np.ndarray    # (N,) int codes for fairness audit


def extract_mimic_features(
    omop: dict[str, pl.DataFrame],
    max_seq_len: int = 50,
    text_embeddings: np.ndarray | None = None,
    text_person_ids: list[str] | None = None,
) -> MIMICFeatures:
    """Convert MIMIC OMOP tables into feature tensors.

    Parameters
    ----------
    omop : dict[str, pl.DataFrame]
        Output of ``mimic_etl.run()`` containing person, measurement,
        drug_exposure, observation_period, death tables.
    max_seq_len : int
        Maximum number of time steps for lab sequences.
    text_embeddings : np.ndarray | None
        Pre-computed (M, 768) ClinicalBERT embeddings. If None, random
        embeddings are used as placeholder.
    text_person_ids : list[str] | None
        Patient IDs matching rows of text_embeddings.

    Returns
    -------
    MIMICFeatures
        Dataclass with numpy arrays ready for PyTorch.
    """
    person = omop["person"]
    measurement = omop.get("measurement", pl.DataFrame())
    drug = omop.get("drug_exposure", pl.DataFrame())
    obs_period = omop.get("observation_period", pl.DataFrame())
    death = omop.get("death", pl.DataFrame())

    pids = person["person_id"].to_list()
    n = len(pids)
    pid_to_idx = {pid: i for i, pid in enumerate(pids)}

    # ---- Tabular features ----
    # Age: year_of_birth -> approximate age (assume diagnosis ~2020)
    yob = person["year_of_birth"].cast(pl.Float32).fill_null(1960).to_numpy()
    age = (2020.0 - yob).astype(np.float32)

    # Race from race_source_value
    race_map = {"White": 0, "Black": 1, "Asian": 2, "Hispanic": 3}
    race_raw = person["race_source_value"].fill_null("Other").to_list()
    race_codes = np.array(
        [race_map.get(str(r).strip(), 4) for r in race_raw], dtype=np.int32
    )
    race_onehot = np.zeros((n, 5), dtype=np.float32)
    race_onehot[np.arange(n), race_codes] = 1.0

    # Drug count per patient
    n_drugs = np.zeros(n, dtype=np.float32)
    if drug.height > 0:
        drug_counts = drug.group_by("person_id").agg(
            pl.col("drug_source_value").n_unique().alias("n_drugs")
        )
        for row in drug_counts.iter_rows():
            idx = pid_to_idx.get(row[0])
            if idx is not None:
                n_drugs[idx] = row[1]

    # Has surgery (procedure_occurrence if available, else 0)
    has_surgery = np.zeros(n, dtype=np.float32)
    if "procedure_occurrence" in omop and omop["procedure_occurrence"].height > 0:
        proc = omop["procedure_occurrence"]
        surg_pids = proc["person_id"].unique().to_list()
        for pid in surg_pids:
            idx = pid_to_idx.get(pid)
            if idx is not None:
                has_surgery[idx] = 1.0

    X_tab = np.column_stack([age, race_onehot, n_drugs, has_surgery]).astype(np.float32)
    # Standardize age
    mu, sigma = age.mean(), age.std()
    if sigma > 0:
        X_tab[:, 0] = (X_tab[:, 0] - mu) / sigma

    # ---- Temporal lab sequences ----
    X_seq = np.zeros((n, max_seq_len, 7), dtype=np.float32)
    seq_mask = np.zeros((n, max_seq_len), dtype=bool)

    if measurement.height > 0:
        # Filter to known labs and sort by date
        labs = measurement.filter(
            pl.col("measurement_source_value").is_in(LAB_NAMES)
        ).sort(["person_id", "measurement_date"])

        # Group by patient
        for pid, group in labs.group_by("person_id"):
            pid_str = pid[0] if isinstance(pid, tuple) else pid
            idx = pid_to_idx.get(pid_str)
            if idx is None:
                continue

            # Get unique dates -> time steps
            dates = group["measurement_date"].unique().sort().to_list()
            for t_idx, dt in enumerate(dates[:max_seq_len]):
                seq_mask[idx, t_idx] = True
                day_labs = group.filter(pl.col("measurement_date") == dt)
                for row in day_labs.iter_rows(named=True):
                    lab_idx = _LAB_SOURCE_MAP.get(row["measurement_source_value"])
                    if lab_idx is not None and row["value_as_number"] is not None:
                        lab_name = row["measurement_source_value"]
                        # z-score normalize
                        val = (row["value_as_number"] - _LAB_MEANS[lab_name]) / _LAB_STDS[lab_name]
                        X_seq[idx, t_idx, lab_idx] = val

    # ---- Text embeddings ----
    X_text = np.zeros((n, 768), dtype=np.float32)
    if text_embeddings is not None and text_person_ids is not None:
        text_pid_map = {pid: i for i, pid in enumerate(text_person_ids)}
        for pid, idx in pid_to_idx.items():
            text_idx = text_pid_map.get(pid)
            if text_idx is not None:
                X_text[idx] = text_embeddings[text_idx]
    else:
        # Placeholder: small random embeddings for synthetic data
        rng = np.random.default_rng(42)
        X_text = rng.standard_normal((n, 768)).astype(np.float32) * 0.01

    # ---- Survival outcome ----
    time_days = np.zeros(n, dtype=np.float32)
    event = np.zeros(n, dtype=np.int32)

    # Compute survival time from observation_period
    if obs_period.height > 0:
        for row in obs_period.iter_rows(named=True):
            idx = pid_to_idx.get(row["person_id"])
            if idx is not None:
                start = row["observation_period_start_date"]
                end = row["observation_period_end_date"]
                if start is not None and end is not None:
                    time_days[idx] = (end - start).days

    # Mark deaths
    if death.height > 0:
        for row in death.iter_rows(named=True):
            idx = pid_to_idx.get(row["person_id"])
            if idx is not None:
                event[idx] = 1

    return MIMICFeatures(
        X_tab=X_tab,
        X_seq=X_seq,
        seq_mask=seq_mask,
        X_text=X_text,
        time=time_days,
        event=event,
        person_ids=pids,
        race_labels=race_codes,
    )


def compute_text_embeddings(
    notes: pl.DataFrame,
    model_name: str = "emilyalsentzer/Bio_ClinicalBERT",
    batch_size: int = 16,
    max_length: int = 512,
) -> tuple[np.ndarray, list[str]]:
    """Compute ClinicalBERT [CLS] embeddings from discharge notes.

    Parameters
    ----------
    notes : pl.DataFrame
        Must have columns: person_id, text.
    model_name : str
        HuggingFace model name for ClinicalBERT.
    batch_size : int
        Batch size for inference.
    max_length : int
        Max token length for truncation.

    Returns
    -------
    embeddings : np.ndarray  (M, 768)
    person_ids : list[str]
    """
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ImportError as exc:
        raise ImportError(
            "Install transformers and torch for text embeddings:\n"
            "  pip install transformers torch"
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # Deduplicate: one embedding per patient (take first note)
    deduped = notes.group_by("person_id").agg(pl.col("text").first())
    pids = deduped["person_id"].to_list()
    texts = deduped["text"].fill_null("").to_list()

    embeddings_list: list[np.ndarray] = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        ).to(device)

        with torch.no_grad():
            outputs = model(**encoded)
            cls_embeddings = outputs.last_hidden_state[:, 0, :]
            embeddings_list.append(cls_embeddings.cpu().numpy())

    all_embeddings = np.concatenate(embeddings_list, axis=0).astype(np.float32)
    return all_embeddings, pids
