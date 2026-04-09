"""Unify OMOP CDM outputs from all ETL pipelines into a single
TNBC cohort Parquet file with a shared patient_id namespace.

Data-lineage:
  SEER OMOP tables      ─┐
  MIMIC OMOP tables      ─┤  cohort_builder.build()
  TCGA-BRCA OMOP tables  ─┘  → data/output/tnbc_cohort.parquet

The output is a *wide* denormalized table: one row per person, with
columns aggregated from person, condition, measurement, drug, procedure,
observation, and death tables across all sources.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl


def _pivot_measurements(meas: pl.DataFrame) -> pl.DataFrame:
    """Pivot measurement rows into one column per measurement_source_value.

    Takes the *last* (most recent) value per person × source_value.
    """
    if meas.height == 0:
        return pl.DataFrame({"person_id": []}).cast({"person_id": pl.Utf8})

    pivoted = (
        meas.group_by(["person_id", "measurement_source_value"])
        .agg(pl.col("value_as_number").last())
        .pivot(
            on="measurement_source_value",
            index="person_id",
            values="value_as_number",
        )
    )
    return pivoted


def _pivot_observations(obs: pl.DataFrame) -> pl.DataFrame:
    """Pivot observation rows into one column per observation_source_value."""
    if obs.height == 0:
        return pl.DataFrame({"person_id": []}).cast({"person_id": pl.Utf8})

    # Numeric observations
    numeric = obs.filter(pl.col("value_as_number").is_not_null())
    string = obs.filter(
        pl.col("value_as_number").is_null() & pl.col("value_as_string").is_not_null()
    )

    parts: list[pl.DataFrame] = []
    if numeric.height > 0:
        pn = (
            numeric.group_by(["person_id", "observation_source_value"])
            .agg(pl.col("value_as_number").last())
            .pivot(
                on="observation_source_value",
                index="person_id",
                values="value_as_number",
            )
        )
        parts.append(pn)
    if string.height > 0:
        ps = (
            string.group_by(["person_id", "observation_source_value"])
            .agg(pl.col("value_as_string").last())
            .pivot(
                on="observation_source_value",
                index="person_id",
                values="value_as_string",
            )
        )
        parts.append(ps)

    if not parts:
        return pl.DataFrame({"person_id": []}).cast({"person_id": pl.Utf8})

    result = parts[0]
    for p in parts[1:]:
        result = result.join(p, on="person_id", how="full", coalesce=True)
    return result


def _count_drugs(drug: pl.DataFrame) -> pl.DataFrame:
    """Count distinct drugs per person."""
    if drug.height == 0:
        return pl.DataFrame({"person_id": []}).cast({"person_id": pl.Utf8})

    return drug.group_by("person_id").agg(
        pl.col("drug_source_value").n_unique().alias("n_distinct_drugs"),
        pl.col("drug_source_value").first().alias("first_drug"),
    )


def _count_procedures(proc: pl.DataFrame) -> pl.DataFrame:
    """Summarise procedures per person."""
    if proc.height == 0:
        return pl.DataFrame({"person_id": []}).cast({"person_id": pl.Utf8})

    return proc.group_by("person_id").agg(
        pl.col("procedure_source_value").n_unique().alias("n_procedures"),
    )


def build(
    source_tables: dict[str, dict[str, pl.DataFrame]],
    output_path: str | Path = "data/output/tnbc_cohort.parquet",
) -> pl.DataFrame:
    """Merge all source OMOP tables into one wide cohort DataFrame.

    Parameters
    ----------
    source_tables : dict
        Keyed by source name (``"seer"``, ``"mimic"``, etc.), each value
        is the dict of OMOP table DataFrames returned by that ETL's
        ``run()`` function.
    output_path : str | Path
        Where to write the final Parquet file.

    Returns
    -------
    pl.DataFrame
        The unified cohort (also written to *output_path*).
    """

    # Collect all OMOP tables across sources
    all_person: list[pl.DataFrame] = []
    all_condition: list[pl.DataFrame] = []
    all_measurement: list[pl.DataFrame] = []
    all_drug: list[pl.DataFrame] = []
    all_procedure: list[pl.DataFrame] = []
    all_observation: list[pl.DataFrame] = []
    all_death: list[pl.DataFrame] = []

    for _source, tbl in source_tables.items():
        if "person" in tbl:
            all_person.append(tbl["person"])
        if "condition_occurrence" in tbl:
            all_condition.append(tbl["condition_occurrence"])
        if "measurement" in tbl:
            all_measurement.append(tbl["measurement"])
        if "drug_exposure" in tbl:
            all_drug.append(tbl["drug_exposure"])
        if "procedure_occurrence" in tbl:
            all_procedure.append(tbl["procedure_occurrence"])
        if "observation" in tbl:
            all_observation.append(tbl["observation"])
        if "death" in tbl:
            all_death.append(tbl["death"])

    # ── persons (spine) ───────────────────────────────────────────
    person = pl.concat(all_person) if all_person else pl.DataFrame()
    if person.height == 0:
        raise ValueError("No person records found across any source.")

    cohort = person

    # ── earliest condition date ───────────────────────────────────
    if all_condition:
        conditions = pl.concat(all_condition)
        first_dx = conditions.group_by("person_id").agg(
            pl.col("condition_start_date").min().alias("diagnosis_date"),
            pl.col("condition_source_value").first().alias("diagnosis_source"),
        )
        cohort = cohort.join(first_dx, on="person_id", how="left")

    # ── measurements (pivoted wide) ───────────────────────────────
    if all_measurement:
        measurements = pl.concat(all_measurement)
        meas_wide = _pivot_measurements(measurements)
        cohort = cohort.join(meas_wide, on="person_id", how="left")

    # ── drugs ─────────────────────────────────────────────────────
    if all_drug:
        drugs = pl.concat(all_drug)
        drug_agg = _count_drugs(drugs)
        cohort = cohort.join(drug_agg, on="person_id", how="left")

    # ── procedures ────────────────────────────────────────────────
    if all_procedure:
        procedures = pl.concat(all_procedure)
        proc_agg = _count_procedures(procedures)
        cohort = cohort.join(proc_agg, on="person_id", how="left")

    # ── observations (pivoted wide) ──────────────────────────────
    if all_observation:
        observations = pl.concat(all_observation)
        obs_wide = _pivot_observations(observations)
        cohort = cohort.join(obs_wide, on="person_id", how="left")

    # ── death / vital status ─────────────────────────────────────
    if all_death:
        deaths = pl.concat(all_death)
        death_info = deaths.group_by("person_id").agg(
            pl.col("death_date").min().alias("death_date"),
        )
        cohort = cohort.join(death_info, on="person_id", how="left")
        cohort = cohort.with_columns(
            pl.when(pl.col("death_date").is_not_null())
            .then(pl.lit(1))
            .otherwise(pl.lit(0))
            .alias("deceased")
        )

    # ── write ─────────────────────────────────────────────────────
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cohort.write_parquet(out)

    return cohort
