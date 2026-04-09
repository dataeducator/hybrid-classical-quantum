"""OMOP CDM v5.4 schema definitions as Polars-compatible typed dicts.

Reference: https://ohdsi.github.io/CommonDataModel/cdm54.html
Each function returns an empty Polars DataFrame with the correct schema
so downstream code can validate and append rows consistently.

Data-lineage: This module defines the *target* schema.  Every ETL
pipeline must produce DataFrames whose columns are a subset of these.
"""

from __future__ import annotations

import polars as pl


# ── person ────────────────────────────────────────────────────────────
PERSON_SCHEMA: dict[str, pl.DataType] = {
    "person_id": pl.Utf8,
    "gender_concept_id": pl.Int64,
    "year_of_birth": pl.Int32,
    "month_of_birth": pl.Int32,
    "race_concept_id": pl.Int64,
    "ethnicity_concept_id": pl.Int64,
    "person_source_value": pl.Utf8,
    "gender_source_value": pl.Utf8,
    "race_source_value": pl.Utf8,
    "data_source": pl.Utf8,
}


# ── observation_period ────────────────────────────────────────────────
OBSERVATION_PERIOD_SCHEMA: dict[str, pl.DataType] = {
    "person_id": pl.Utf8,
    "observation_period_start_date": pl.Date,
    "observation_period_end_date": pl.Date,
    "period_type_concept_id": pl.Int64,
}


# ── condition_occurrence ──────────────────────────────────────────────
CONDITION_OCCURRENCE_SCHEMA: dict[str, pl.DataType] = {
    "person_id": pl.Utf8,
    "condition_concept_id": pl.Int64,
    "condition_start_date": pl.Date,
    "condition_end_date": pl.Date,
    "condition_type_concept_id": pl.Int64,
    "condition_source_value": pl.Utf8,
}


# ── measurement ───────────────────────────────────────────────────────
MEASUREMENT_SCHEMA: dict[str, pl.DataType] = {
    "person_id": pl.Utf8,
    "measurement_concept_id": pl.Int64,
    "measurement_date": pl.Date,
    "value_as_number": pl.Float64,
    "value_as_concept_id": pl.Int64,
    "unit_concept_id": pl.Int64,
    "measurement_source_value": pl.Utf8,
}


# ── drug_exposure ─────────────────────────────────────────────────────
DRUG_EXPOSURE_SCHEMA: dict[str, pl.DataType] = {
    "person_id": pl.Utf8,
    "drug_concept_id": pl.Int64,
    "drug_exposure_start_date": pl.Date,
    "drug_exposure_end_date": pl.Date,
    "drug_type_concept_id": pl.Int64,
    "drug_source_value": pl.Utf8,
}


# ── procedure_occurrence ─────────────────────────────────────────────
PROCEDURE_OCCURRENCE_SCHEMA: dict[str, pl.DataType] = {
    "person_id": pl.Utf8,
    "procedure_concept_id": pl.Int64,
    "procedure_date": pl.Date,
    "procedure_type_concept_id": pl.Int64,
    "procedure_source_value": pl.Utf8,
}


# ── observation (surveys, exposures, misc) ────────────────────────────
OBSERVATION_SCHEMA: dict[str, pl.DataType] = {
    "person_id": pl.Utf8,
    "observation_concept_id": pl.Int64,
    "observation_date": pl.Date,
    "value_as_number": pl.Float64,
    "value_as_string": pl.Utf8,
    "observation_source_value": pl.Utf8,
}


# ── death ─────────────────────────────────────────────────────────────
DEATH_SCHEMA: dict[str, pl.DataType] = {
    "person_id": pl.Utf8,
    "death_date": pl.Date,
    "death_type_concept_id": pl.Int64,
    "cause_concept_id": pl.Int64,
    "cause_source_value": pl.Utf8,
}


# ── helpers ───────────────────────────────────────────────────────────

def empty_frame(schema: dict[str, pl.DataType]) -> pl.DataFrame:
    """Return an empty DataFrame conforming to *schema*."""
    return pl.DataFrame(schema={k: v for k, v in schema.items()})


def cast_to_schema(df: pl.DataFrame, schema: dict[str, pl.DataType]) -> pl.DataFrame:
    """Select and cast *df* columns to match *schema*.

    Columns present in *schema* but missing from *df* are filled with null.
    Extra columns in *df* are dropped.
    """
    exprs: list[pl.Expr] = []
    for col_name, col_type in schema.items():
        if col_name in df.columns:
            exprs.append(pl.col(col_name).cast(col_type))
        else:
            exprs.append(pl.lit(None).cast(col_type).alias(col_name))
    return df.select(exprs)
