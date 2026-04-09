"""ETL: MIMIC-IV hospital tables -> OMOP CDM v5.4.

Supports two extraction modes (set via MIMIC_SOURCE env var):
  - "csv"      (default)  Local CSV files in MIMIC_DATA_PATH
  - "bigquery"            Google BigQuery physionet-data project

Data-lineage:
  CSV mode:
    patients.csv, admissions.csv, diagnoses_icd.csv,
    labevents.csv, prescriptions.csv, discharge_notes.csv
  BigQuery mode:
    physionet-data.mimiciv_hosp.patients
    physionet-data.mimiciv_hosp.admissions
    physionet-data.mimiciv_hosp.diagnoses_icd
    physionet-data.mimiciv_hosp.labevents  (filtered to breast-CA labs)
    physionet-data.mimiciv_hosp.prescriptions
    physionet-data.mimiciv_note.discharge
  -> person, observation_period, condition_occurrence,
     measurement, drug_exposure, death
"""

from __future__ import annotations

import os
from pathlib import Path

import polars as pl

from ..concept_maps import (
    CONDITION, DRUG, GENDER, MEASUREMENT, RACE,
    SOURCE_TAG, TYPE_CONCEPT,
)
from ..omop_schema import (
    CONDITION_OCCURRENCE_SCHEMA, DEATH_SCHEMA, DRUG_EXPOSURE_SCHEMA,
    MEASUREMENT_SCHEMA, OBSERVATION_PERIOD_SCHEMA, PERSON_SCHEMA,
    cast_to_schema,
)

_LAB_CONCEPT_MAP = {
    "WBC": MEASUREMENT["wbc"],
    "Hemoglobin": MEASUREMENT["hemoglobin"],
    "Platelets": MEASUREMENT["platelets"],
    "Creatinine": MEASUREMENT["creatinine"],
    "Albumin": MEASUREMENT["albumin"],
    "ALP": MEASUREMENT["alp"],
    "LDH": MEASUREMENT["ldh"],
}

_MIMIC_RACE_MAP = {
    "WHITE": "White",
    "BLACK/AFRICAN AMERICAN": "Black",
    "ASIAN": "Asian",
    "HISPANIC/LATINO": "Hispanic",
    "OTHER": "Other",
}

_DRUG_CONCEPT_MAP = {
    "doxorubicin": DRUG["doxorubicin"],
    "cyclophosphamide": DRUG["cyclophosphamide"],
    "paclitaxel": DRUG["paclitaxel"],
    "carboplatin": DRUG["carboplatin"],
    "pembrolizumab": DRUG["pembrolizumab"],
    "capecitabine": DRUG["capecitabine"],
}


def extract(data_path: str | Path) -> dict[str, pl.DataFrame]:
    """Read all MIMIC CSV tables into Polars DataFrames (local CSV mode)."""
    p = Path(data_path)
    tables: dict[str, pl.DataFrame] = {}
    for name in ["patients", "admissions", "diagnoses_icd",
                 "labevents", "prescriptions", "discharge_notes"]:
        tables[name] = pl.read_csv(
            p / f"{name}.csv", try_parse_dates=True,
            infer_schema_length=10000, ignore_errors=True,
        )
    return tables


# ── BigQuery extraction ──────────────────────────────────────────────

# Lab itemids for the seven labs we track (from MIMIC-IV d_labitems)
_BQ_LAB_ITEMIDS = {
    51301: "WBC",        # White Blood Cells
    51222: "Hemoglobin",
    51265: "Platelets",
    50912: "Creatinine",
    50862: "Albumin",
    50863: "ALP",        # Alkaline Phosphatase
    50954: "LDH",        # Lactate Dehydrogenase
}

# Chemo drug names (lower-cased substrings to match prescriptions.drug)
_BQ_CHEMO_DRUGS = [
    "doxorubicin", "cyclophosphamide", "paclitaxel",
    "carboplatin", "pembrolizumab", "capecitabine",
]


def extract_bigquery(
    bq_project: str = "physionet-data",
    dataset_hosp: str = "mimiciv_hosp",
    dataset_note: str = "mimiciv_note",
    gcp_project: str | None = None,
) -> dict[str, pl.DataFrame]:
    """Pull MIMIC-IV breast-cancer cohort directly from BigQuery.

    Requires ``google-cloud-bigquery`` and ``db-dtypes`` packages, plus
    a GCP project with BigQuery access to physionet-data.

    The queries filter server-side to only breast-cancer patients
    (ICD-10 C50.*) to minimise data transfer.
    """
    try:
        from google.cloud import bigquery
    except ImportError as exc:
        raise ImportError(
            "Install google-cloud-bigquery and db-dtypes:\n"
            "  pip install google-cloud-bigquery db-dtypes"
        ) from exc

    gcp_project = gcp_project or os.getenv("GCP_PROJECT")
    client = bigquery.Client(project=gcp_project)

    def _query(sql: str) -> pl.DataFrame:
        return pl.from_pandas(client.query(sql).to_dataframe())

    hosp = f"`{bq_project}.{dataset_hosp}`"
    note = f"`{bq_project}.{dataset_note}`"

    # 1. Find all subject_ids with a breast cancer ICD-10 code
    cohort_sql = f"""
    SELECT DISTINCT subject_id
    FROM {hosp}.diagnoses_icd
    WHERE icd_code LIKE 'C50%' AND icd_version = 10
    """

    # 2. patients
    patients_sql = f"""
    SELECT p.subject_id, p.gender, p.anchor_year, p.anchor_age,
           p.anchor_year_group, p.dod
    FROM {hosp}.patients p
    INNER JOIN ({cohort_sql}) c USING (subject_id)
    """

    # 3. admissions
    admissions_sql = f"""
    SELECT a.subject_id, a.hadm_id,
           a.admittime, a.dischtime,
           a.insurance, a.race
    FROM {hosp}.admissions a
    INNER JOIN ({cohort_sql}) c USING (subject_id)
    """

    # 4. diagnoses_icd (C50.* only)
    diagnoses_sql = f"""
    SELECT subject_id, hadm_id, icd_code, icd_version, seq_num
    FROM {hosp}.diagnoses_icd
    WHERE icd_code LIKE 'C50%' AND icd_version = 10
    """

    # 5. labevents (filtered to relevant itemids)
    itemid_list = ",".join(str(i) for i in _BQ_LAB_ITEMIDS)
    labs_sql = f"""
    SELECT le.subject_id, le.hadm_id, le.charttime,
           le.itemid, le.valuenum, le.valueuom
    FROM {hosp}.labevents le
    INNER JOIN ({cohort_sql}) c USING (subject_id)
    WHERE le.itemid IN ({itemid_list})
      AND le.valuenum IS NOT NULL
    """

    # 6. prescriptions (chemo drugs only)
    drug_clauses = " OR ".join(
        f"LOWER(drug) LIKE '%{d}%'" for d in _BQ_CHEMO_DRUGS
    )
    rx_sql = f"""
    SELECT p.subject_id, p.hadm_id, p.drug,
           p.starttime, p.stoptime, p.route
    FROM {hosp}.prescriptions p
    INNER JOIN ({cohort_sql}) c USING (subject_id)
    WHERE {drug_clauses}
    """

    # 7. discharge notes
    notes_sql = f"""
    SELECT d.subject_id, d.hadm_id,
           'Discharge summary' AS category, d.text
    FROM {note}.discharge d
    INNER JOIN ({cohort_sql}) c USING (subject_id)
    """

    print("      [BQ] Querying patients ...")
    patients = _query(patients_sql)

    print("      [BQ] Querying admissions ...")
    admissions = _query(admissions_sql)
    # Normalise race column to match synthetic format
    admissions = admissions.with_columns(
        pl.col("race").str.to_uppercase().alias("race")
    )
    # Add has_icu_stay / icu_los_days (simplified: not queried here)
    admissions = admissions.with_columns(
        pl.lit(False).alias("has_icu_stay"),
        pl.lit(0).alias("icu_los_days"),
    )

    print("      [BQ] Querying diagnoses ...")
    diagnoses = _query(diagnoses_sql)

    print("      [BQ] Querying lab results ...")
    labs_raw = _query(labs_sql)
    # Map itemid -> label name to match synthetic format
    labs = labs_raw.with_columns(
        pl.col("itemid").map_elements(
            lambda i: _BQ_LAB_ITEMIDS.get(i, "Unknown"),
            return_dtype=pl.Utf8,
        ).alias("label"),
        pl.col("valuenum").alias("value"),
    ).select("subject_id", "hadm_id", "charttime", "label", "value",
             pl.col("valueuom").alias("valueuom"))

    print("      [BQ] Querying prescriptions ...")
    prescriptions = _query(rx_sql)
    # Normalise drug name to lower case to match concept map keys
    prescriptions = prescriptions.with_columns(
        pl.col("drug").str.to_lowercase().alias("drug")
    )

    print("      [BQ] Querying discharge notes ...")
    notes = _query(notes_sql)

    # Prefix subject_id with MIMIC_ to match synthetic namespace
    for name in ["patients", "admissions", "diagnoses", "labs",
                 "prescriptions", "notes"]:
        tbl = locals()[name]
        if "subject_id" in tbl.columns:
            locals()[name] = tbl.with_columns(
                ("MIMIC_" + pl.col("subject_id").cast(pl.Utf8)).alias("subject_id")
            )

    return {
        "patients": patients,
        "admissions": admissions,
        "diagnoses_icd": diagnoses,
        "labevents": labs,
        "prescriptions": prescriptions,
        "discharge_notes": notes,
    }


def transform(tables: dict[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
    """Map raw MIMIC-IV tables to OMOP CDM."""

    patients = tables["patients"]
    admissions = tables["admissions"]
    labs = tables["labevents"]
    prescriptions = tables["prescriptions"]

    # Resolve race: take the first admission's race per patient
    pat_race = admissions.group_by("subject_id").agg(
        pl.col("race").first()
    )

    # ── person ────────────────────────────────────────────────────
    person = patients.join(pat_race, on="subject_id", how="left").select(
        pl.col("subject_id").alias("person_id"),
        pl.col("gender").map_elements(
            lambda g: GENDER.get(g, 0), return_dtype=pl.Int64
        ).alias("gender_concept_id"),
        (pl.col("anchor_year") - pl.col("anchor_age")).alias("year_of_birth").cast(pl.Int32),
        pl.lit(1).cast(pl.Int32).alias("month_of_birth"),
        pl.col("race").map_elements(
            lambda r: RACE.get(_MIMIC_RACE_MAP.get(r, "Unknown"), 0),
            return_dtype=pl.Int64,
        ).alias("race_concept_id"),
        pl.lit(0).cast(pl.Int64).alias("ethnicity_concept_id"),
        pl.col("subject_id").alias("person_source_value"),
        pl.col("gender").alias("gender_source_value"),
        pl.col("race").alias("race_source_value"),
        pl.lit(SOURCE_TAG["mimic"]).alias("data_source"),
    )
    person = cast_to_schema(person, PERSON_SCHEMA)

    # ── observation_period ────────────────────────────────────────
    obs = admissions.group_by("subject_id").agg(
        pl.col("admittime").min().cast(pl.Date).alias("observation_period_start_date"),
        pl.col("dischtime").max().cast(pl.Date).alias("observation_period_end_date"),
    ).select(
        pl.col("subject_id").alias("person_id"),
        "observation_period_start_date",
        "observation_period_end_date",
        pl.lit(TYPE_CONCEPT["ehr"]).cast(pl.Int64).alias("period_type_concept_id"),
    )
    obs_period = cast_to_schema(obs, OBSERVATION_PERIOD_SCHEMA)

    # ── condition_occurrence ──────────────────────────────────────
    diagnoses = tables["diagnoses_icd"]
    cond = diagnoses.filter(
        pl.col("icd_code").str.starts_with("C50")
    ).join(
        admissions.select("subject_id", "hadm_id", "admittime"),
        on=["subject_id", "hadm_id"],
    ).select(
        pl.col("subject_id").alias("person_id"),
        pl.lit(CONDITION["breast_cancer_icd10_C50"]).cast(pl.Int64).alias("condition_concept_id"),
        pl.col("admittime").cast(pl.Date).alias("condition_start_date"),
        pl.lit(None).cast(pl.Date).alias("condition_end_date"),
        pl.lit(TYPE_CONCEPT["ehr"]).cast(pl.Int64).alias("condition_type_concept_id"),
        pl.col("icd_code").alias("condition_source_value"),
    )
    condition = cast_to_schema(cond, CONDITION_OCCURRENCE_SCHEMA)

    # ── measurement (labs) ────────────────────────────────────────
    meas = labs.select(
        pl.col("subject_id").alias("person_id"),
        pl.col("label").map_elements(
            lambda l: _LAB_CONCEPT_MAP.get(l, 0), return_dtype=pl.Int64
        ).alias("measurement_concept_id"),
        pl.col("charttime").cast(pl.Date).alias("measurement_date"),
        pl.col("value").cast(pl.Float64).alias("value_as_number"),
        pl.lit(0).cast(pl.Int64).alias("value_as_concept_id"),
        pl.lit(0).cast(pl.Int64).alias("unit_concept_id"),
        pl.col("label").alias("measurement_source_value"),
    )
    measurement = cast_to_schema(meas, MEASUREMENT_SCHEMA)

    # ── drug_exposure ─────────────────────────────────────────────
    drug = prescriptions.select(
        pl.col("subject_id").alias("person_id"),
        pl.col("drug").map_elements(
            lambda d: _DRUG_CONCEPT_MAP.get(d, 0), return_dtype=pl.Int64
        ).alias("drug_concept_id"),
        pl.col("starttime").cast(pl.Date).alias("drug_exposure_start_date"),
        pl.col("stoptime").cast(pl.Date).alias("drug_exposure_end_date"),
        pl.lit(TYPE_CONCEPT["ehr"]).cast(pl.Int64).alias("drug_type_concept_id"),
        pl.col("drug").alias("drug_source_value"),
    )
    drug_exposure = cast_to_schema(drug, DRUG_EXPOSURE_SCHEMA)

    # ── death ────────────────────────────────────────────────────
    dead = patients.filter(pl.col("dod").is_not_null()).select(
        pl.col("subject_id").alias("person_id"),
        pl.col("dod").cast(pl.Date).alias("death_date"),
        pl.lit(TYPE_CONCEPT["ehr"]).cast(pl.Int64).alias("death_type_concept_id"),
        pl.lit(CONDITION["malignant_neoplasm_breast"]).cast(pl.Int64).alias("cause_concept_id"),
        pl.lit("breast_cancer").alias("cause_source_value"),
    )
    death = cast_to_schema(dead, DEATH_SCHEMA)

    return {
        "person": person,
        "observation_period": obs_period,
        "condition_occurrence": condition,
        "measurement": measurement,
        "drug_exposure": drug_exposure,
        "death": death,
    }


def run(data_path: str | Path) -> dict[str, pl.DataFrame]:
    """Full extract-transform pipeline for MIMIC-IV data.

    Checks ``MIMIC_SOURCE`` env var:
      - ``"bigquery"`` -> pull from physionet-data on GCP BigQuery
      - anything else  -> read local CSVs from *data_path*
    """
    source = os.getenv("MIMIC_SOURCE", "csv").lower()

    if source == "bigquery":
        tables = extract_bigquery(
            bq_project=os.getenv("MIMIC_BQ_PROJECT", "physionet-data"),
            dataset_hosp=os.getenv("MIMIC_BQ_DATASET_HOSP", "mimiciv_hosp"),
            dataset_note=os.getenv("MIMIC_BQ_DATASET_NOTE", "mimiciv_note"),
            gcp_project=os.getenv("GCP_PROJECT"),
        )
    else:
        tables = extract(data_path)

    return transform(tables)
