"""ETL: NHANES survey & examination files → OMOP CDM v5.4.

Data-lineage:
  demographics.csv, environmental.csv, dietary.csv,
  examination.csv, cancer_history.csv  (NHANES_DATA_PATH)
    → person, condition_occurrence (self-reported breast CA),
      measurement (environmental biomarkers, vitals),
      observation (diet, SES, smoking)
"""

from __future__ import annotations

from datetime import date
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

_NHANES_RACE_MAP = {
    "Non-Hispanic White": "White",
    "Non-Hispanic Black": "Black",
    "Mexican American": "Hispanic",
    "Other Hispanic": "Hispanic",
    "Non-Hispanic Asian": "Asian",
    "Other/Multi": "Other",
}


def extract(data_path: str | Path) -> dict[str, pl.DataFrame]:
    """Read NHANES CSVs into Polars DataFrames."""
    p = Path(data_path)
    tables: dict[str, pl.DataFrame] = {}
    for name in ["demographics", "environmental", "dietary",
                 "examination", "cancer_history"]:
        fp = p / f"{name}.csv"
        if fp.exists():
            tables[name] = pl.read_csv(fp, try_parse_dates=True)
    return tables


def _survey_date(cycle: str) -> date:
    """Approximate date from NHANES cycle string e.g. '2017-2018'."""
    try:
        year = int(cycle.split("-")[0])
    except (ValueError, IndexError):
        year = 2018
    return date(year, 6, 15)


def transform(tables: dict[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
    """Map NHANES tables to OMOP CDM."""

    demo = tables["demographics"]
    env = tables["environmental"]
    diet = tables["dietary"]
    exam = tables["examination"]

    # ── person ────────────────────────────────────────────────────
    person = demo.select(
        pl.col("seqn").alias("person_id"),
        pl.col("sex").map_elements(
            lambda s: GENDER.get(s, 0), return_dtype=pl.Int64
        ).alias("gender_concept_id"),
        pl.col("survey_cycle").map_elements(
            lambda c: int(c.split("-")[0]) - demo["age"].to_list()[0]
            if isinstance(c, str) else 2000,
            return_dtype=pl.Int64,
        ).alias("year_of_birth"),
        pl.lit(1).cast(pl.Int32).alias("month_of_birth"),
        pl.col("race_ethnicity").map_elements(
            lambda r: RACE.get(_NHANES_RACE_MAP.get(r, "Unknown"), 0),
            return_dtype=pl.Int64,
        ).alias("race_concept_id"),
        pl.col("race_ethnicity").map_elements(
            lambda r: 38003563 if "Hispanic" in r or "Mexican" in r else 38003564,
            return_dtype=pl.Int64,
        ).alias("ethnicity_concept_id"),
        pl.col("seqn").alias("person_source_value"),
        pl.col("sex").alias("gender_source_value"),
        pl.col("race_ethnicity").alias("race_source_value"),
        pl.lit(SOURCE_TAG["nhanes"]).alias("data_source"),
    )
    # Fix year_of_birth properly using row-wise calculation
    person = demo.select(
        pl.col("seqn").alias("person_id"),
        pl.col("sex").map_elements(
            lambda s: GENDER.get(s, 0), return_dtype=pl.Int64
        ).alias("gender_concept_id"),
        (pl.col("survey_cycle").map_elements(
            lambda c: int(c.split("-")[0]) if isinstance(c, str) else 2018,
            return_dtype=pl.Int64,
        ) - pl.col("age")).cast(pl.Int32).alias("year_of_birth"),
        pl.lit(1).cast(pl.Int32).alias("month_of_birth"),
        pl.col("race_ethnicity").map_elements(
            lambda r: RACE.get(_NHANES_RACE_MAP.get(r, "Unknown"), 0),
            return_dtype=pl.Int64,
        ).alias("race_concept_id"),
        pl.col("race_ethnicity").map_elements(
            lambda r: 38003563 if "Hispanic" in r or "Mexican" in r else 38003564,
            return_dtype=pl.Int64,
        ).alias("ethnicity_concept_id"),
        pl.col("seqn").alias("person_source_value"),
        pl.col("sex").alias("gender_source_value"),
        pl.col("race_ethnicity").alias("race_source_value"),
        pl.lit(SOURCE_TAG["nhanes"]).alias("data_source"),
    )
    person = cast_to_schema(person, PERSON_SCHEMA)

    # ── observation_period ────────────────────────────────────────
    obs = demo.select(
        pl.col("seqn").alias("person_id"),
        pl.col("survey_cycle").map_elements(
            _survey_date, return_dtype=pl.Date,
        ).alias("observation_period_start_date"),
        pl.col("survey_cycle").map_elements(
            _survey_date, return_dtype=pl.Date,
        ).alias("observation_period_end_date"),
        pl.lit(TYPE_CONCEPT["survey"]).cast(pl.Int64).alias("period_type_concept_id"),
    )
    obs_period = cast_to_schema(obs, OBSERVATION_PERIOD_SCHEMA)

    # ── condition_occurrence (self-reported breast cancer) ─────────
    condition = None
    if "cancer_history" in tables and tables["cancer_history"].height > 0:
        cancer = tables["cancer_history"]
        cond = cancer.filter(pl.col("cancer_type") == "Breast").select(
            pl.col("seqn").alias("person_id"),
            pl.lit(CONDITION["malignant_neoplasm_breast"]).cast(pl.Int64).alias("condition_concept_id"),
            pl.col("age_at_diagnosis").map_elements(
                lambda a: date(2018 - 50 + a, 6, 1),  # approximate
                return_dtype=pl.Date,
            ).alias("condition_start_date"),
            pl.lit(None).cast(pl.Date).alias("condition_end_date"),
            pl.lit(TYPE_CONCEPT["survey"]).cast(pl.Int64).alias("condition_type_concept_id"),
            pl.lit("self_report_breast_cancer").alias("condition_source_value"),
        )
        condition = cast_to_schema(cond, CONDITION_OCCURRENCE_SCHEMA)

    # ── measurement (environmental biomarkers + vitals) ───────────
    meas_rows: list[pl.DataFrame] = []
    survey_date_col = demo.select(
        pl.col("seqn"),
        pl.col("survey_cycle").map_elements(
            _survey_date, return_dtype=pl.Date
        ).alias("meas_date"),
    )

    env_with_date = env.join(survey_date_col, left_on="seqn", right_on="seqn")
    for col_name, concept_key in [
        ("blood_lead_ugdl", "blood_lead"),
        ("blood_cadmium_ugdl", "blood_cadmium"),
        ("blood_mercury_ugdl", "blood_mercury"),
        ("urinary_bpa_ngml", "urinary_bpa"),
    ]:
        m = env_with_date.select(
            pl.col("seqn").alias("person_id"),
            pl.lit(MEASUREMENT[concept_key]).cast(pl.Int64).alias("measurement_concept_id"),
            pl.col("meas_date").alias("measurement_date"),
            pl.col(col_name).cast(pl.Float64).alias("value_as_number"),
            pl.lit(0).cast(pl.Int64).alias("value_as_concept_id"),
            pl.lit(0).cast(pl.Int64).alias("unit_concept_id"),
            pl.lit(col_name).alias("measurement_source_value"),
        )
        meas_rows.append(m)

    # BMI, BP
    exam_with_date = exam.join(survey_date_col, left_on="seqn", right_on="seqn")
    for col_name, concept_key in [
        ("bmi", "bmi"),
        ("systolic_bp", "systolic_bp"),
        ("diastolic_bp", "diastolic_bp"),
    ]:
        m = exam_with_date.select(
            pl.col("seqn").alias("person_id"),
            pl.lit(MEASUREMENT[concept_key]).cast(pl.Int64).alias("measurement_concept_id"),
            pl.col("meas_date").alias("measurement_date"),
            pl.col(col_name).cast(pl.Float64).alias("value_as_number"),
            pl.lit(0).cast(pl.Int64).alias("value_as_concept_id"),
            pl.lit(0).cast(pl.Int64).alias("unit_concept_id"),
            pl.lit(col_name).alias("measurement_source_value"),
        )
        meas_rows.append(m)

    measurement = cast_to_schema(pl.concat(meas_rows), MEASUREMENT_SCHEMA)

    # ── observation (diet, SES, smoking) ──────────────────────────
    obs_rows: list[pl.DataFrame] = []

    # Dietary
    diet_with_date = diet.join(survey_date_col, left_on="seqn", right_on="seqn")
    for col_name, concept_key in [
        ("total_calories_kcal", "total_calories"),
        ("dietary_fiber_g", "dietary_fiber"),
        ("fruit_veg_servings", "fruit_veg_servings"),
    ]:
        o = diet_with_date.select(
            pl.col("seqn").alias("person_id"),
            pl.lit(OBS_CONCEPTS[concept_key]).cast(pl.Int64).alias("observation_concept_id"),
            pl.col("meas_date").alias("observation_date"),
            pl.col(col_name).cast(pl.Float64).alias("value_as_number"),
            pl.lit(None).cast(pl.Utf8).alias("value_as_string"),
            pl.lit(col_name).alias("observation_source_value"),
        )
        obs_rows.append(o)

    # Poverty income ratio
    pir_with_date = demo.select(
        "seqn", "poverty_income_ratio",
    ).join(survey_date_col, left_on="seqn", right_on="seqn")
    pir = pir_with_date.select(
        pl.col("seqn").alias("person_id"),
        pl.lit(OBS_CONCEPTS["poverty_income_ratio"]).cast(pl.Int64).alias("observation_concept_id"),
        pl.col("meas_date").alias("observation_date"),
        pl.col("poverty_income_ratio").cast(pl.Float64).alias("value_as_number"),
        pl.lit(None).cast(pl.Utf8).alias("value_as_string"),
        pl.lit("poverty_income_ratio").alias("observation_source_value"),
    )
    obs_rows.append(pir)

    # Smoking
    smoking_with_date = exam.select("seqn", "smoking_status").join(
        survey_date_col, left_on="seqn", right_on="seqn"
    )
    smoke = smoking_with_date.select(
        pl.col("seqn").alias("person_id"),
        pl.lit(OBS_CONCEPTS["smoking_status"]).cast(pl.Int64).alias("observation_concept_id"),
        pl.col("meas_date").alias("observation_date"),
        pl.lit(None).cast(pl.Float64).alias("value_as_number"),
        pl.col("smoking_status").alias("value_as_string"),
        pl.lit("smoking_status").alias("observation_source_value"),
    )
    obs_rows.append(smoke)

    observation = cast_to_schema(pl.concat(obs_rows), OBSERVATION_SCHEMA)

    result: dict[str, pl.DataFrame] = {
        "person": person,
        "observation_period": obs_period,
        "measurement": measurement,
        "observation": observation,
    }
    if condition is not None:
        result["condition_occurrence"] = condition

    return result


def run(data_path: str | Path) -> dict[str, pl.DataFrame]:
    """Full extract-transform pipeline for NHANES data."""
    tables = extract(data_path)
    return transform(tables)
