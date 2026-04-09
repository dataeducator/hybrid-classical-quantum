"""ETL: All of Us research tables → OMOP CDM v5.4.

Data-lineage:
  person.csv, condition_occurrence.csv, survey.csv,
  genomic.csv, wearable.csv  (ALLOFUS_DATA_PATH)
    → person, observation_period, condition_occurrence,
      measurement (genomic + wearable), observation (survey)
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from ..concept_maps import (
    CONDITION, GENDER, MEASUREMENT, OBSERVATION as OBS_CONCEPTS,
    RACE, SOURCE_TAG, TYPE_CONCEPT,
)
from ..omop_schema import (
    CONDITION_OCCURRENCE_SCHEMA, MEASUREMENT_SCHEMA,
    OBSERVATION_PERIOD_SCHEMA, OBSERVATION_SCHEMA,
    PERSON_SCHEMA, cast_to_schema,
)


def extract(data_path: str | Path) -> dict[str, pl.DataFrame]:
    """Read All of Us CSVs into Polars DataFrames."""
    p = Path(data_path)
    tables: dict[str, pl.DataFrame] = {}
    for name in ["person", "condition_occurrence", "survey",
                 "genomic", "wearable"]:
        fp = p / f"{name}.csv"
        if fp.exists():
            tables[name] = pl.read_csv(fp, try_parse_dates=True)
    return tables


def transform(tables: dict[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
    """Map All of Us tables to OMOP CDM."""

    persons = tables["person"]
    conditions = tables["condition_occurrence"]
    surveys = tables["survey"]

    # ── person ────────────────────────────────────────────────────
    person = persons.select(
        pl.col("person_id"),
        pl.col("sex").map_elements(
            lambda s: GENDER.get(s, 0), return_dtype=pl.Int64
        ).alias("gender_concept_id"),
        pl.col("year_of_birth").cast(pl.Int32),
        pl.lit(1).cast(pl.Int32).alias("month_of_birth"),
        pl.col("race").map_elements(
            lambda r: RACE.get(r, 0), return_dtype=pl.Int64
        ).alias("race_concept_id"),
        pl.col("ethnicity").map_elements(
            lambda e: 38003563 if e == "Hispanic" else 38003564,
            return_dtype=pl.Int64,
        ).alias("ethnicity_concept_id"),
        pl.col("person_id").alias("person_source_value"),
        pl.col("sex").alias("gender_source_value"),
        pl.col("race").alias("race_source_value"),
        pl.lit(SOURCE_TAG["allofus"]).alias("data_source"),
    )
    person = cast_to_schema(person, PERSON_SCHEMA)

    # ── observation_period ────────────────────────────────────────
    obs = conditions.select(
        pl.col("person_id"),
        pl.col("condition_start_date").cast(pl.Date).alias("observation_period_start_date"),
        pl.col("condition_start_date").cast(pl.Date).alias("observation_period_end_date"),
        pl.lit(TYPE_CONCEPT["ehr"]).cast(pl.Int64).alias("period_type_concept_id"),
    )
    obs_period = cast_to_schema(obs, OBSERVATION_PERIOD_SCHEMA)

    # ── condition_occurrence ──────────────────────────────────────
    cond = conditions.select(
        pl.col("person_id"),
        pl.col("condition_concept_id").cast(pl.Int64),
        pl.col("condition_start_date").cast(pl.Date),
        pl.lit(None).cast(pl.Date).alias("condition_end_date"),
        pl.lit(TYPE_CONCEPT["ehr"]).cast(pl.Int64).alias("condition_type_concept_id"),
        pl.col("condition_source_value"),
    )
    condition = cast_to_schema(cond, CONDITION_OCCURRENCE_SCHEMA)

    # ── observation (survey data) ─────────────────────────────────
    obs_rows: list[pl.DataFrame] = []
    survey_map = {
        "income_bracket": OBS_CONCEPTS["income_level"],
        "education": OBS_CONCEPTS["education_level"],
        "insurance": OBS_CONCEPTS["insurance_type"],
        "smoking_status": OBS_CONCEPTS["smoking_status"],
    }
    for col_name, concept_id in survey_map.items():
        o = surveys.select(
            pl.col("person_id"),
            pl.lit(concept_id).cast(pl.Int64).alias("observation_concept_id"),
            pl.col("survey_date").cast(pl.Date).alias("observation_date"),
            pl.lit(None).cast(pl.Float64).alias("value_as_number"),
            pl.col(col_name).alias("value_as_string"),
            pl.lit(col_name).alias("observation_source_value"),
        )
        obs_rows.append(o)
    observation = cast_to_schema(pl.concat(obs_rows), OBSERVATION_SCHEMA)

    # ── measurement (genomic) ─────────────────────────────────────
    meas_rows: list[pl.DataFrame] = []
    if "genomic" in tables:
        genomic = tables["genomic"]
        # BRCA1
        brca1 = genomic.select(
            pl.col("person_id"),
            pl.lit(OBS_CONCEPTS["brca1_variant"]).cast(pl.Int64).alias("measurement_concept_id"),
            pl.lit(None).cast(pl.Date).alias("measurement_date"),
            pl.col("brca1_pathogenic").cast(pl.Float64).alias("value_as_number"),
            pl.lit(0).cast(pl.Int64).alias("value_as_concept_id"),
            pl.lit(0).cast(pl.Int64).alias("unit_concept_id"),
            pl.lit("brca1_pathogenic").alias("measurement_source_value"),
        )
        meas_rows.append(brca1)
        # BRCA2
        brca2 = genomic.select(
            pl.col("person_id"),
            pl.lit(OBS_CONCEPTS["brca2_variant"]).cast(pl.Int64).alias("measurement_concept_id"),
            pl.lit(None).cast(pl.Date).alias("measurement_date"),
            pl.col("brca2_pathogenic").cast(pl.Float64).alias("value_as_number"),
            pl.lit(0).cast(pl.Int64).alias("value_as_concept_id"),
            pl.lit(0).cast(pl.Int64).alias("unit_concept_id"),
            pl.lit("brca2_pathogenic").alias("measurement_source_value"),
        )
        meas_rows.append(brca2)
        # TMB
        tmb = genomic.select(
            pl.col("person_id"),
            pl.lit(MEASUREMENT["tmb"]).cast(pl.Int64).alias("measurement_concept_id"),
            pl.lit(None).cast(pl.Date).alias("measurement_date"),
            pl.col("tumor_mutational_burden").cast(pl.Float64).alias("value_as_number"),
            pl.lit(0).cast(pl.Int64).alias("value_as_concept_id"),
            pl.lit(0).cast(pl.Int64).alias("unit_concept_id"),
            pl.lit("tumor_mutational_burden").alias("measurement_source_value"),
        )
        meas_rows.append(tmb)

    # ── measurement (wearable) ────────────────────────────────────
    if "wearable" in tables:
        wearable = tables["wearable"]
        for col_name, concept_key in [("heart_rate_mean", "heart_rate"),
                                       ("steps_daily", "steps_daily"),
                                       ("sleep_hours", "sleep_hours")]:
            w = wearable.select(
                pl.col("person_id"),
                pl.lit(MEASUREMENT[concept_key]).cast(pl.Int64).alias("measurement_concept_id"),
                pl.col("measurement_date").cast(pl.Date),
                pl.col(col_name).cast(pl.Float64).alias("value_as_number"),
                pl.lit(0).cast(pl.Int64).alias("value_as_concept_id"),
                pl.lit(0).cast(pl.Int64).alias("unit_concept_id"),
                pl.lit(col_name).alias("measurement_source_value"),
            )
            meas_rows.append(w)

    measurement = cast_to_schema(pl.concat(meas_rows), MEASUREMENT_SCHEMA) if meas_rows else None

    result: dict[str, pl.DataFrame] = {
        "person": person,
        "observation_period": obs_period,
        "condition_occurrence": condition,
        "observation": observation,
    }
    if measurement is not None:
        result["measurement"] = measurement

    return result


def run(data_path: str | Path) -> dict[str, pl.DataFrame]:
    """Full extract-transform pipeline for All of Us data."""
    tables = extract(data_path)
    return transform(tables)
