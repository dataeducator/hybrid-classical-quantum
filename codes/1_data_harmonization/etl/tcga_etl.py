"""ETL: TCGA-BRCA (cBioPortal) -> OMOP CDM v5.4.

Supports two modes:
  1. **API mode** (default): Fetches clinical + genomic data directly from
     the cBioPortal public REST API (https://www.cbioportal.org/api/).
  2. **CSV mode**: Reads pre-downloaded CSVs from a local directory.

Data-lineage:
  cBioPortal brca_tcga study (or local CSVs)
    -> TNBC filter (ER-/PR-/HER2-)
    -> person, observation_period, condition_occurrence,
       measurement (biomarkers + genomic), observation (clinical attrs)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import polars as pl

from ..concept_maps import (
    CONDITION, GENDER, MEASUREMENT, OBSERVATION as OBS_CONCEPTS,
    RACE, SOURCE_TAG, TYPE_CONCEPT, BIOMARKER_VALUE, STAGE,
)
from ..omop_schema import (
    CONDITION_OCCURRENCE_SCHEMA, MEASUREMENT_SCHEMA,
    OBSERVATION_PERIOD_SCHEMA, OBSERVATION_SCHEMA,
    PERSON_SCHEMA, cast_to_schema,
)

# ── cBioPortal API configuration ─────────────────────────────────────
CBIO_BASE = "https://www.cbioportal.org/api"
STUDY_ID = "brca_tcga"

# ── TCGA race mapping ────────────────────────────────────────────────
_TCGA_RACE_MAP: dict[str, str] = {
    "WHITE": "White",
    "BLACK OR AFRICAN AMERICAN": "Black",
    "ASIAN": "Asian",
    "AMERICAN INDIAN OR ALASKA NATIVE": "Other",
    "NATIVE HAWAIIAN OR OTHER PACIFIC ISLANDER": "Other",
}


def _fetch_json(endpoint: str) -> list[dict[str, Any]]:
    """GET from cBioPortal API, return parsed JSON."""
    import urllib.request
    import json

    url = f"{CBIO_BASE}/{endpoint}"
    req = urllib.request.Request(url, headers={"Accept": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode())


def _fetch_clinical_patient() -> pl.DataFrame:
    """Fetch patient-level clinical data from cBioPortal."""
    data = _fetch_json(
        f"studies/{STUDY_ID}/clinical-data"
        f"?clinicalDataType=PATIENT&projection=DETAILED"
    )
    rows: dict[str, dict[str, str]] = {}
    for rec in data:
        pid = rec["patientId"]
        if pid not in rows:
            rows[pid] = {"patient_id": pid}
        rows[pid][rec["clinicalAttributeId"]] = rec["value"]
    return pl.DataFrame(list(rows.values()))


def _fetch_clinical_sample() -> pl.DataFrame:
    """Fetch sample-level clinical data from cBioPortal."""
    data = _fetch_json(
        f"studies/{STUDY_ID}/clinical-data"
        f"?clinicalDataType=SAMPLE&projection=DETAILED"
    )
    rows: dict[str, dict[str, str]] = {}
    for rec in data:
        sid = rec["sampleId"]
        if sid not in rows:
            rows[sid] = {"sample_id": sid, "patient_id": rec["patientId"]}
        rows[sid][rec["clinicalAttributeId"]] = rec["value"]
    return pl.DataFrame(list(rows.values()))


def _fetch_mutations(sample_ids: list[str]) -> pl.DataFrame:
    """Fetch mutation data for given samples via cBioPortal API."""
    import urllib.request
    import json

    url = f"{CBIO_BASE}/molecular-profiles/brca_tcga_mutations/mutations/fetch"
    body = json.dumps({
        "sampleIds": sample_ids,
        "sampleListId": None,
        "entrezGeneIds": [],
    }).encode()
    req = urllib.request.Request(
        url, data=body,
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return pl.DataFrame(json.loads(resp.read().decode()))
    except Exception:
        return pl.DataFrame()


def extract_api() -> dict[str, pl.DataFrame]:
    """Extract TCGA-BRCA data from cBioPortal REST API."""
    patient_clin = _fetch_clinical_patient()
    sample_clin = _fetch_clinical_sample()

    # Filter for TNBC: ER-negative, PR-negative, HER2-negative
    # Receptor status is at the PATIENT level in TCGA-BRCA
    tnbc_patients = patient_clin.filter(
        (pl.col("ER_STATUS_BY_IHC").str.to_lowercase() == "negative")
        & (pl.col("PR_STATUS_BY_IHC").str.to_lowercase() == "negative")
        & (
            (pl.col("HER2_FISH_STATUS").str.to_lowercase() == "negative")
            | (pl.col("IHC_HER2").str.to_lowercase() == "negative")
        )
    )
    tnbc_patient_ids = tnbc_patients["patient_id"].unique().to_list()

    # Filter samples to TNBC patients
    tnbc_samples = sample_clin.filter(
        pl.col("patient_id").is_in(tnbc_patient_ids)
    )

    # Fetch mutations for TNBC samples
    tnbc_sample_ids = tnbc_samples["sample_id"].to_list() if tnbc_samples.height > 0 else []
    mutations = _fetch_mutations(tnbc_sample_ids) if tnbc_sample_ids else pl.DataFrame()

    return {
        "patient_clinical": tnbc_patients,
        "sample_clinical": tnbc_samples,
        "mutations": mutations,
    }


def extract_csv(data_path: str | Path) -> dict[str, pl.DataFrame]:
    """Read pre-downloaded TCGA-BRCA CSVs from a local directory."""
    p = Path(data_path)
    tables: dict[str, pl.DataFrame] = {}
    for name in ["patient_clinical", "sample_clinical", "mutations"]:
        fp = p / f"{name}.csv"
        if fp.exists():
            tables[name] = pl.read_csv(fp, try_parse_dates=True)
    return tables


def extract(data_path: str | Path | None = None) -> dict[str, pl.DataFrame]:
    """Extract TCGA-BRCA data. Uses API if no data_path given, else reads CSVs."""
    if data_path is not None and Path(data_path).exists():
        return extract_csv(data_path)

    # Check env var
    env_path = os.environ.get("TCGA_DATA_PATH")
    if env_path and Path(env_path).exists():
        return extract_csv(env_path)

    return extract_api()


def transform(tables: dict[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
    """Map TCGA-BRCA tables to OMOP CDM."""

    patients = tables["patient_clinical"]
    samples = tables.get("sample_clinical", pl.DataFrame())

    # ── person ────────────────────────────────────────────────────
    def _map_sex(s: str) -> int:
        return GENDER.get(s, GENDER.get(s.capitalize(), 0))

    def _map_race(r: str) -> int:
        mapped = _TCGA_RACE_MAP.get(r.upper(), "Other") if isinstance(r, str) else "Other"
        return RACE.get(mapped, 0)

    # Extract year_of_birth from AGE (approximate: current year - age)
    person_cols = {
        "person_id": patients["patient_id"].map_elements(lambda pid: f"TCGA_{pid}", return_dtype=pl.Utf8),
    }

    person = pl.DataFrame({
        "person_id": patients["patient_id"].map_elements(lambda pid: f"TCGA_{pid}", return_dtype=pl.Utf8),
    })

    # Map available columns
    if "SEX" in patients.columns:
        person = person.with_columns(
            patients["SEX"].map_elements(_map_sex, return_dtype=pl.Int64)
            .alias("gender_concept_id")
        )
    else:
        person = person.with_columns(
            pl.lit(GENDER["Female"]).cast(pl.Int64).alias("gender_concept_id")
        )

    if "AGE" in patients.columns:
        person = person.with_columns(
            (2024 - patients["AGE"].cast(pl.Int32)).alias("year_of_birth"),
            pl.lit(1).cast(pl.Int32).alias("month_of_birth"),
        )
    else:
        person = person.with_columns(
            pl.lit(1965).cast(pl.Int32).alias("year_of_birth"),
            pl.lit(1).cast(pl.Int32).alias("month_of_birth"),
        )

    if "RACE" in patients.columns:
        person = person.with_columns(
            patients["RACE"].map_elements(_map_race, return_dtype=pl.Int64)
            .alias("race_concept_id")
        )
    else:
        person = person.with_columns(pl.lit(0).cast(pl.Int64).alias("race_concept_id"))

    if "ETHNICITY" in patients.columns:
        person = person.with_columns(
            patients["ETHNICITY"].map_elements(
                lambda e: 38003563 if isinstance(e, str) and "hispanic" in e.lower() else 38003564,
                return_dtype=pl.Int64,
            ).alias("ethnicity_concept_id")
        )
    else:
        person = person.with_columns(pl.lit(38003564).cast(pl.Int64).alias("ethnicity_concept_id"))

    person = person.with_columns(
        patients["patient_id"].alias("person_source_value"),
        pl.lit("").alias("gender_source_value"),
        pl.lit("").alias("race_source_value"),
        pl.lit(SOURCE_TAG["tcga"]).alias("data_source"),
    )
    person = cast_to_schema(person, PERSON_SCHEMA)

    # ── observation_period ────────────────────────────────────────
    obs_period_rows = []
    for pid in patients["patient_id"]:
        obs_period_rows.append({
            "person_id": f"TCGA_{pid}",
            "observation_period_start_date": None,
            "observation_period_end_date": None,
            "period_type_concept_id": TYPE_CONCEPT["registry"],
        })
    obs_period = cast_to_schema(
        pl.DataFrame(obs_period_rows), OBSERVATION_PERIOD_SCHEMA
    )

    # ── condition_occurrence (TNBC diagnosis) ─────────────────────
    cond_rows = []
    for pid in patients["patient_id"]:
        cond_rows.append({
            "person_id": f"TCGA_{pid}",
            "condition_concept_id": CONDITION["tnbc"],
            "condition_start_date": None,
            "condition_end_date": None,
            "condition_type_concept_id": TYPE_CONCEPT["registry"],
            "condition_source_value": "Triple-negative breast cancer (TCGA-BRCA)",
        })
    condition = cast_to_schema(
        pl.DataFrame(cond_rows), CONDITION_OCCURRENCE_SCHEMA
    )

    # ── measurement (genomic + clinical biomarkers) ───────────────
    meas_rows: list[dict] = []
    for row in patients.iter_rows(named=True):
        pid = f"TCGA_{row['patient_id']}"

        # Tumor mutational burden
        if "MUTATION_COUNT" in row and row["MUTATION_COUNT"] is not None:
            try:
                meas_rows.append({
                    "person_id": pid,
                    "measurement_concept_id": MEASUREMENT["tmb"],
                    "measurement_date": None,
                    "value_as_number": float(row["MUTATION_COUNT"]),
                    "value_as_concept_id": 0,
                    "unit_concept_id": 0,
                    "measurement_source_value": "mutation_count",
                })
            except (ValueError, TypeError):
                pass

        # Fraction genome altered
        if "FRACTION_GENOME_ALTERED" in row and row["FRACTION_GENOME_ALTERED"] is not None:
            try:
                meas_rows.append({
                    "person_id": pid,
                    "measurement_concept_id": MEASUREMENT.get("fraction_genome_altered", 36304254),
                    "measurement_date": None,
                    "value_as_number": float(row["FRACTION_GENOME_ALTERED"]),
                    "value_as_concept_id": 0,
                    "unit_concept_id": 0,
                    "measurement_source_value": "fraction_genome_altered",
                })
            except (ValueError, TypeError):
                pass

    # Add sample-level biomarker measurements
    if samples.height > 0:
        for row in samples.iter_rows(named=True):
            pid = f"TCGA_{row['patient_id']}"
            # ER status
            if "ER_STATUS_BY_IHC" in row and row["ER_STATUS_BY_IHC"]:
                val_concept = BIOMARKER_VALUE.get(row["ER_STATUS_BY_IHC"], 0)
                meas_rows.append({
                    "person_id": pid,
                    "measurement_concept_id": MEASUREMENT["er_status"],
                    "measurement_date": None,
                    "value_as_number": 0.0 if row["ER_STATUS_BY_IHC"] == "Negative" else 1.0,
                    "value_as_concept_id": val_concept,
                    "unit_concept_id": 0,
                    "measurement_source_value": "er_status_by_ihc",
                })
            # PR status
            if "PR_STATUS_BY_IHC" in row and row["PR_STATUS_BY_IHC"]:
                val_concept = BIOMARKER_VALUE.get(row["PR_STATUS_BY_IHC"], 0)
                meas_rows.append({
                    "person_id": pid,
                    "measurement_concept_id": MEASUREMENT["pr_status"],
                    "measurement_date": None,
                    "value_as_number": 0.0 if row["PR_STATUS_BY_IHC"] == "Negative" else 1.0,
                    "value_as_concept_id": val_concept,
                    "unit_concept_id": 0,
                    "measurement_source_value": "pr_status_by_ihc",
                })
            # HER2 status
            her2_val = row.get("HER2_FISH_STATUS") or row.get("IHC_HER2")
            if her2_val:
                val_concept = BIOMARKER_VALUE.get(her2_val, 0)
                meas_rows.append({
                    "person_id": pid,
                    "measurement_concept_id": MEASUREMENT["her2_status"],
                    "measurement_date": None,
                    "value_as_number": 0.0 if "Negative" in str(her2_val) else 1.0,
                    "value_as_concept_id": val_concept,
                    "unit_concept_id": 0,
                    "measurement_source_value": "her2_status",
                })

    measurement = (
        cast_to_schema(pl.DataFrame(meas_rows), MEASUREMENT_SCHEMA)
        if meas_rows else None
    )

    # ── observation (staging, clinical attributes) ────────────────
    obs_rows: list[dict] = []
    for row in patients.iter_rows(named=True):
        pid = f"TCGA_{row['patient_id']}"

        # AJCC stage
        stage_val = row.get("AJCC_PATHOLOGIC_TUMOR_STAGE")
        if stage_val and isinstance(stage_val, str):
            # Clean up: "STAGE IIA" -> "IIA"
            clean_stage = stage_val.replace("STAGE ", "").strip()
            obs_rows.append({
                "person_id": pid,
                "observation_concept_id": STAGE.get(clean_stage, 0),
                "observation_date": None,
                "value_as_number": None,
                "value_as_string": stage_val,
                "observation_source_value": "ajcc_pathologic_stage",
            })

        # Cancer type detailed
        ct = row.get("CANCER_TYPE_DETAILED")
        if ct:
            obs_rows.append({
                "person_id": pid,
                "observation_concept_id": 0,
                "observation_date": None,
                "value_as_number": None,
                "value_as_string": str(ct),
                "observation_source_value": "cancer_type_detailed",
            })

        # OS status and months
        os_status = row.get("OS_STATUS")
        os_months = row.get("OS_MONTHS")
        if os_status:
            obs_rows.append({
                "person_id": pid,
                "observation_concept_id": 0,
                "observation_date": None,
                "value_as_number": float(os_months) if os_months else None,
                "value_as_string": str(os_status),
                "observation_source_value": "os_status",
            })
        if os_months:
            obs_rows.append({
                "person_id": pid,
                "observation_concept_id": 0,
                "observation_date": None,
                "value_as_number": float(os_months),
                "value_as_string": None,
                "observation_source_value": "os_months",
            })

        # DFS
        dfs_months = row.get("DFS_MONTHS")
        if dfs_months:
            obs_rows.append({
                "person_id": pid,
                "observation_concept_id": 0,
                "observation_date": None,
                "value_as_number": float(dfs_months),
                "value_as_string": None,
                "observation_source_value": "dfs_months",
            })

    observation = (
        cast_to_schema(pl.DataFrame(obs_rows), OBSERVATION_SCHEMA)
        if obs_rows else None
    )

    result: dict[str, pl.DataFrame] = {
        "person": person,
        "observation_period": obs_period,
        "condition_occurrence": condition,
    }
    if measurement is not None:
        result["measurement"] = measurement
    if observation is not None:
        result["observation"] = observation

    return result


def run(data_path: str | Path | None = None) -> dict[str, pl.DataFrame]:
    """Full extract-transform pipeline for TCGA-BRCA data.

    Parameters
    ----------
    data_path : str | Path | None
        Path to local CSVs, or None to fetch from cBioPortal API.
    """
    tables = extract(data_path)
    return transform(tables)
