"""ETL: SEER cancer-registry flat file -> OMOP CDM v5.4.

Supports two input formats (auto-detected):
  - Synthetic CSV (from seer_synth.py) -- columns already normalised
  - Real SEERStat export CSV           -- columns renamed via adapter

Data-lineage:
  seer_tnbc.csv  OR  seer_tnbc_real.csv  (SEER_DATA_PATH)
    -> person, observation_period, condition_occurrence,
       measurement (biomarkers, tumor size),
       procedure_occurrence, drug_exposure, death
"""

from __future__ import annotations

import os
from datetime import date
from pathlib import Path

import polars as pl

from ..concept_maps import (
    BIOMARKER_VALUE, CONDITION, DRUG, GENDER, MEASUREMENT,
    PROCEDURE, RACE, SOURCE_TAG, STAGE, TYPE_CONCEPT,
)
from ..omop_schema import (
    CONDITION_OCCURRENCE_SCHEMA, DEATH_SCHEMA, DRUG_EXPOSURE_SCHEMA,
    MEASUREMENT_SCHEMA, OBSERVATION_PERIOD_SCHEMA, PERSON_SCHEMA,
    PROCEDURE_OCCURRENCE_SCHEMA, cast_to_schema,
)


# ── SEERStat real-export column mapping ──────────────────────────────
# Keys   = common SEERStat export column names (case-insensitive match)
# Values = internal pipeline column names
_SEERSTAT_COLUMN_MAP: dict[str, str] = {
    # Patient ID
    "patient id":                                     "patient_id",
    # Age -- many possible recode names
    "age at diagnosis":                               "age_at_diagnosis",
    "age recode with <1 year olds":                   "age_at_diagnosis",
    "age recode with single ages and 85+":            "age_at_diagnosis",
    "age recode (<60,60-69,70+)":                     "age_at_diagnosis",
    # Demographics
    "sex":                                            "sex",
    "race recode (w, b, ai, api)":                    "race",
    "race recode (white, black, other)":              "race",
    "race and origin recode (nhia)":                  "race",
    # Diagnosis
    "year of diagnosis":                              "year_of_diagnosis",
    "primary site - labeled":                         "primary_site",
    "primary site":                                   "primary_site",
    "icd-o-3 hist/behav":                             "histology_raw",
    "histologic type icd-o-3":                        "histology_raw",
    # Grade (varies by era)
    "grade (thru 2017)":                              "grade",
    "grade recode (thru 2017)":                       "grade",
    "grade pathological (2018+)":                     "grade_2018",
    # Stage (varies by era)
    "derived ajcc stage group, 7th ed (2010-2015)":   "ajcc_stage",
    "derived ajcc stage group, 8th ed (2018+)":       "ajcc_stage_8th",
    "derived eod 2018 stage group recode (2018+)":    "ajcc_stage_eod",
    "ajcc stage 3rd edition (1988-2003)":             "ajcc_stage",
    # Biomarkers (year ranges vary by export)
    "er status recode breast cancer (2010+)":         "er_status",
    "er status recode breast cancer (1990+)":         "er_status",
    "pr status recode breast cancer (2010+)":         "pr_status",
    "pr status recode breast cancer (1990+)":         "pr_status",
    "derived her2 recode (2010+)":                    "her2_status",
    # Tumor size (varies by era)
    "cs tumor size (2004-2015)":                      "tumor_size_mm_pre2016",
    "tumor size summary (2016+)":                     "tumor_size_mm_post2016",
    # Other clinical
    "regional nodes positive (1988+)":                "lymph_nodes_positive",
    "laterality":                                     "laterality",
    "rx summ--surg prim site (1998+)":                "surgery_raw",
    "radiation recode":                               "radiation",
    "chemotherapy recode (yes, no/unk)":              "chemotherapy",
    # Outcome
    "survival months":                                "survival_months",
    "vital status recode (study cutoff used)":        "vital_status",
    "end calc vital status (adjusted)":               "vital_status",
    # Bonus SES variable
    "median household income inflation adj to 2023":  "median_income",
}

# SEERStat race values -> pipeline internal values
_SEERSTAT_RACE_MAP: dict[str, str] = {
    "White": "White",
    "Black": "Black",
    "American Indian/Alaska Native": "Other",
    "Asian or Pacific Islander": "Asian",
    "Other (American Indian/AK Native, Asian/Pacific Islander)": "Other",
    "Unknown": "Unknown",
}

# SEERStat surgery: numeric code -> simplified category
# Source: SEER "RX Summ--Surg Prim Site" coding manual for breast
_SEERSTAT_SURGERY_CODE_MAP: dict[int, str] = {
    0:  "None",           # No surgery
    19: "None",           # Unknown if surgery performed; death cert only
    20: "Lumpectomy",     # Local tumor destruction/excision
    21: "Lumpectomy",     # Partial mastectomy, NOS
    22: "Lumpectomy",     # Less than total mastectomy/excisional biopsy
    23: "Lumpectomy",     # Segmental mastectomy (includes wedge)
    24: "Lumpectomy",     # Lumpectomy or excisional biopsy
    30: "Mastectomy",     # Subcutaneous mastectomy
    40: "Mastectomy",     # Total (simple) mastectomy
    41: "Mastectomy",     # Total (simple) mastectomy
    42: "Mastectomy",     # Modified radical mastectomy
    43: "Mastectomy",     # Modified radical mastectomy
    44: "Mastectomy",     # Modified radical mastectomy
    45: "Mastectomy",     # Radical mastectomy
    46: "Mastectomy",     # Radical mastectomy
    50: "Mastectomy",     # Extended radical mastectomy
    60: "Mastectomy",     # Mastectomy, NOS
    70: "Mastectomy",     # Mastectomy with reconstruction
    80: "Mastectomy",     # Mastectomy, NOS (historical)
    90: "None",           # Surgery, NOS / unknown type
    99: "None",           # Unknown
}

# Also handle text labels in case of different export format
_SEERSTAT_SURGERY_TEXT_MAP: dict[str, str] = {
    "Mastectomy": "Mastectomy",
    "Partial mastectomy": "Lumpectomy",
    "Lumpectomy": "Lumpectomy",
    "Subcutaneous mastectomy": "Mastectomy",
    "Total (simple) mastectomy": "Mastectomy",
    "Modified radical mastectomy": "Mastectomy",
    "Radical mastectomy": "Mastectomy",
    "No surgery": "None",
    "None": "None",
}


def _dx_date(year: int) -> date:
    """Approximate diagnosis date from year alone (SEER gives year only)."""
    return date(year, 7, 1)


def _normalise_seerstat(df: pl.DataFrame) -> pl.DataFrame:
    """Rename and clean a real SEERStat CSV export to match the synthetic schema.

    Auto-detects SEERStat format by checking whether the internal column
    names already exist (synthetic) vs. the SEERStat column names.
    """
    # If it already has our internal column names, return as-is (synthetic)
    if "patient_id" in df.columns and "age_at_diagnosis" in df.columns:
        return df

    # Build a rename map from whatever columns are present
    lower_cols = {c.lower().strip(): c for c in df.columns}
    rename: dict[str, str] = {}
    for seer_name, internal_name in _SEERSTAT_COLUMN_MAP.items():
        if seer_name in lower_cols:
            rename[lower_cols[seer_name]] = internal_name

    df = df.rename(rename)

    # Generate patient_id if not present
    if "patient_id" not in df.columns:
        df = df.with_row_index("_idx").with_columns(
            ("SEER_" + pl.col("_idx").cast(pl.Utf8).str.zfill(6)).alias("patient_id")
        ).drop("_idx")

    # Parse age from age-recode strings like "50-54" -> midpoint
    if "age_at_diagnosis" in df.columns:
        df = df.with_columns(
            pl.col("age_at_diagnosis").map_elements(
                _parse_age, return_dtype=pl.Int64
            ).alias("age_at_diagnosis")
        )

    # Normalise race
    if "race" in df.columns:
        df = df.with_columns(
            pl.col("race").map_elements(
                lambda r: _SEERSTAT_RACE_MAP.get(str(r).strip(), "Other"),
                return_dtype=pl.Utf8,
            ).alias("race")
        )

    # Normalise surgery -- handles both numeric codes and text labels
    if "surgery_raw" in df.columns:
        df = df.with_columns(
            pl.col("surgery_raw").map_elements(
                _match_surgery, return_dtype=pl.Utf8,
            ).alias("surgery")
        ).drop("surgery_raw")
    elif "surgery" not in df.columns:
        df = df.with_columns(pl.lit("None").alias("surgery"))

    # Normalise vital status ("Alive" / "Dead")
    if "vital_status" in df.columns:
        df = df.with_columns(
            pl.col("vital_status").map_elements(
                lambda v: "Dead" if "dead" in str(v).lower() else "Alive",
                return_dtype=pl.Utf8,
            ).alias("vital_status")
        )

    # Normalise receptor status values
    for col in ["er_status", "pr_status", "her2_status"]:
        if col in df.columns:
            df = df.with_columns(
                pl.col(col).map_elements(
                    lambda v: _normalise_receptor(str(v)),
                    return_dtype=pl.Utf8,
                ).alias(col)
            )

    # Normalise radiation/chemo to Yes/No
    for col in ["radiation", "chemotherapy"]:
        if col in df.columns:
            df = df.with_columns(
                pl.col(col).map_elements(
                    lambda v: "Yes" if "yes" in str(v).lower()
                    or "beam" in str(v).lower() else "No",
                    return_dtype=pl.Utf8,
                ).alias(col)
            )

    # Merge stage columns: prefer 7th ed, fall back to EOD 2018 / 8th ed
    stage_cols = [c for c in ["ajcc_stage", "ajcc_stage_eod", "ajcc_stage_8th"]
                  if c in df.columns]
    if stage_cols:
        if len(stage_cols) == 1:
            if stage_cols[0] != "ajcc_stage":
                df = df.rename({stage_cols[0]: "ajcc_stage"})
        else:
            # Coalesce: take first non-blank value across era columns
            df = df.with_columns(
                pl.coalesce([
                    pl.when(pl.col(c).cast(pl.Utf8).str.to_lowercase().is_in(
                        ["blank(s)", "", "na", "not applicable"]
                    )).then(None).otherwise(pl.col(c))
                    for c in stage_cols
                ]).alias("ajcc_stage")
            ).drop([c for c in stage_cols if c != "ajcc_stage"])
        df = df.with_columns(
            pl.col("ajcc_stage").map_elements(
                lambda v: _normalise_stage(str(v)),
                return_dtype=pl.Utf8,
            ).alias("ajcc_stage")
        )

    # Merge grade columns (pre-2017 text vs 2018+ numeric)
    if "grade_2018" in df.columns:
        if "grade" in df.columns:
            df = df.with_columns(
                pl.coalesce([
                    pl.when(pl.col("grade").cast(pl.Utf8).str.to_lowercase().is_in(
                        ["blank(s)", "", "unknown"]
                    )).then(None).otherwise(pl.col("grade")),
                    pl.when(pl.col("grade_2018").cast(pl.Utf8).str.to_lowercase().is_in(
                        ["blank(s)", "", "unknown"]
                    )).then(None).otherwise(pl.col("grade_2018")),
                ]).alias("grade")
            ).drop("grade_2018")
        else:
            df = df.rename({"grade_2018": "grade"})
    # Parse grade from text like "Poorly differentiated; Grade III"
    if "grade" in df.columns:
        df = df.with_columns(
            pl.col("grade").map_elements(
                _parse_grade, return_dtype=pl.Int64
            ).alias("grade")
        )

    # Merge tumor size columns (pre-2016 and post-2016)
    size_cols = [c for c in ["tumor_size_mm_pre2016", "tumor_size_mm_post2016"]
                 if c in df.columns]
    if size_cols:
        df = df.with_columns(
            pl.coalesce([
                pl.when(pl.col(c).cast(pl.Utf8).str.to_lowercase().is_in(
                    ["blank(s)", "", "999", "000"]
                )).then(None).otherwise(pl.col(c))
                for c in size_cols
            ]).alias("tumor_size_mm")
        ).drop(size_cols)

    # Parse median income from dollar-range strings to numeric
    if "median_income" in df.columns:
        df = df.with_columns(
            pl.col("median_income").map_elements(
                _parse_income, return_dtype=pl.Float64
            ).alias("median_income")
        )

    # Ensure numeric columns are numeric
    for col in ["year_of_diagnosis", "survival_months",
                "tumor_size_mm", "lymph_nodes_positive"]:
        if col in df.columns:
            df = df.with_columns(
                pl.col(col).cast(pl.Utf8).map_elements(
                    _safe_int, return_dtype=pl.Int64
                ).alias(col)
            )

    # Prefix patient_id with SEER_ for namespace
    if "patient_id" in df.columns:
        df = df.with_columns(
            ("SEER_" + pl.col("patient_id").cast(pl.Utf8)).alias("patient_id")
        )

    # Fill any missing expected columns with defaults
    defaults: dict[str, pl.Expr] = {
        "sex": pl.lit("Female"),
        "primary_site": pl.lit("C50.9"),
        "histology": pl.lit(8500),
        "behavior": pl.lit(3),
        "grade": pl.lit(3),
        "er_status": pl.lit("Negative"),
        "pr_status": pl.lit("Negative"),
        "her2_status": pl.lit("Negative"),
        "laterality": pl.lit("Unknown"),
        "surgery": pl.lit("None"),
        "radiation": pl.lit("No"),
        "chemotherapy": pl.lit("No"),
        "lymph_nodes_positive": pl.lit(0),
        "tumor_size_mm": pl.lit(None).cast(pl.Int64),
    }
    for col_name, default_expr in defaults.items():
        if col_name not in df.columns:
            df = df.with_columns(default_expr.alias(col_name))

    return df


def _parse_age(val: object) -> int:
    """Parse SEERStat age value.

    Handles: plain int, "50-54 years", "85+ years", "70-74", etc.
    """
    s = str(val).strip().lower().replace(" years", "")
    if s.isdigit():
        return int(s)
    if "85+" in s:
        return 88
    if "-" in s:
        parts = s.split("-")
        try:
            digits_a = "".join(c for c in parts[0] if c.isdigit())
            digits_b = "".join(c for c in parts[1] if c.isdigit())
            if digits_a and digits_b:
                return (int(digits_a) + int(digits_b)) // 2
        except (ValueError, IndexError):
            pass
    digits = "".join(c for c in s if c.isdigit())
    return int(digits) if digits else 60


def _parse_grade(val: object) -> int:
    """Parse SEER grade from text or numeric.

    Handles: "Poorly differentiated; Grade III", "3", "Blank(s)", etc.
    """
    s = str(val).strip().lower()
    if s in ("blank(s)", "", "unknown", "not determined", "none"):
        return 9  # unknown
    # Direct numeric
    digits = "".join(c for c in s if c.isdigit())
    if digits and int(digits) in (1, 2, 3, 4):
        return int(digits)
    if "well differentiated" in s:
        return 1
    if "moderately differentiated" in s:
        return 2
    if "poorly differentiated" in s or "undifferentiated" in s:
        return 3
    # Roman numerals
    for roman, num in [("iv", 4), ("iii", 3), ("ii", 2), ("i", 1)]:
        if f"grade {roman}" in s:
            return num
    return 9


def _match_surgery(val: object) -> str:
    """Map SEERStat surgery code (numeric or text) to Mastectomy/Lumpectomy/None."""
    s = str(val).strip()
    # Try numeric code first
    try:
        code = int(s)
        return _SEERSTAT_SURGERY_CODE_MAP.get(code, "None")
    except ValueError:
        pass
    # Fall back to text matching
    v = s.lower()
    if "no surgery" in v or "none" in v or v == "":
        return "None"
    for key, result in _SEERSTAT_SURGERY_TEXT_MAP.items():
        if key.lower() in v:
            return result
    if "mastectom" in v:
        return "Mastectomy"
    if "lumpectom" in v or "partial" in v or "excision" in v:
        return "Lumpectomy"
    return "None"


def _normalise_receptor(val: str) -> str:
    """Normalise ER/PR/HER2 status to Negative/Positive."""
    v = val.strip().lower()
    if "negative" in v or v == "-":
        return "Negative"
    if "positive" in v or v == "+":
        return "Positive"
    if "borderline" in v:
        return "Positive"
    return "Unknown"


def _normalise_stage(val: str) -> str:
    """Clean AJCC stage value."""
    v = val.strip().upper()
    if v in ("", "BLANK(S)", "UNK STAGE", "NA", "NOT APPLICABLE"):
        return "Unknown"
    # Strip leading "STAGE " prefix if present
    v = v.replace("STAGE ", "")
    return v


def _parse_income(val: object) -> float | None:
    """Parse SEERStat household income range to numeric midpoint.

    Handles: "$90,000 - $94,999", "$120,000+", "< $40,000",
             "Unknown/missing/no match/Not 1990-2023"
    """
    s = str(val).strip()
    if "unknown" in s.lower() or "missing" in s.lower() or s == "":
        return None
    # Remove $ and commas
    cleaned = s.replace("$", "").replace(",", "")
    if "+" in cleaned:
        # e.g. "120000+" -> use the value as lower bound + 5000
        digits = "".join(c for c in cleaned.replace("+", "") if c.isdigit())
        return float(int(digits) + 5000) if digits else None
    if "<" in cleaned:
        # e.g. "< 40000" -> use 35000
        digits = "".join(c for c in cleaned.replace("<", "") if c.isdigit())
        return float(int(digits) - 5000) if digits else None
    if "-" in cleaned:
        # e.g. "90000 - 94999" -> midpoint
        parts = cleaned.split("-")
        try:
            lo = int("".join(c for c in parts[0] if c.isdigit()))
            hi = int("".join(c for c in parts[1] if c.isdigit()))
            return float((lo + hi) // 2)
        except (ValueError, IndexError):
            pass
    digits = "".join(c for c in cleaned if c.isdigit())
    return float(int(digits)) if digits else None


def _safe_int(val: object) -> int | None:
    """Convert to int, returning None for non-numeric values."""
    s = str(val).strip()
    digits = "".join(c for c in s if c.isdigit())
    if digits:
        return int(digits)
    return None


def extract(data_path: str | Path) -> pl.DataFrame:
    """Read the SEER CSV into a raw Polars DataFrame.

    Looks for files in this priority order:
      1. seer_tnbc_real.csv  (real SEERStat export)
      2. seer_tnbc.csv       (synthetic)
    """
    p = Path(data_path)
    real_path = p / "seer_tnbc_real.csv"
    synth_path = p / "seer_tnbc.csv"

    if real_path.exists():
        df = pl.read_csv(real_path, infer_schema_length=5000)
        return _normalise_seerstat(df)
    elif synth_path.exists():
        return pl.read_csv(synth_path)
    else:
        raise FileNotFoundError(
            f"No SEER data found. Place seer_tnbc_real.csv or "
            f"seer_tnbc.csv in {p}"
        )


def transform(raw: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Map raw SEER columns to OMOP CDM tables."""

    # ── person ────────────────────────────────────────────────────
    person = raw.select(
        pl.col("patient_id").alias("person_id"),
        pl.col("sex").map_elements(
            lambda s: GENDER.get(s, 0), return_dtype=pl.Int64
        ).alias("gender_concept_id"),
        (pl.col("year_of_diagnosis") - pl.col("age_at_diagnosis")).alias("year_of_birth"),
        pl.lit(7).cast(pl.Int32).alias("month_of_birth"),
        pl.col("race").map_elements(
            lambda r: RACE.get(r, 0), return_dtype=pl.Int64
        ).alias("race_concept_id"),
        pl.lit(0).cast(pl.Int64).alias("ethnicity_concept_id"),
        pl.col("patient_id").alias("person_source_value"),
        pl.col("sex").alias("gender_source_value"),
        pl.col("race").alias("race_source_value"),
        pl.lit(SOURCE_TAG["seer"]).alias("data_source"),
    )
    person = cast_to_schema(person, PERSON_SCHEMA)

    # ── observation_period ────────────────────────────────────────
    obs_period = raw.select(
        pl.col("patient_id").alias("person_id"),
        pl.col("year_of_diagnosis").map_elements(
            _dx_date, return_dtype=pl.Date
        ).alias("observation_period_start_date"),
        pl.struct(["year_of_diagnosis", "survival_months"]).map_elements(
            lambda s: _dx_date(s["year_of_diagnosis"]).replace(
                year=s["year_of_diagnosis"] + s["survival_months"] // 12,
                month=min(12, 1 + (6 + s["survival_months"]) % 12),
            ),
            return_dtype=pl.Date,
        ).alias("observation_period_end_date"),
        pl.lit(TYPE_CONCEPT["registry"]).cast(pl.Int64).alias("period_type_concept_id"),
    )
    obs_period = cast_to_schema(obs_period, OBSERVATION_PERIOD_SCHEMA)

    # ── condition_occurrence (breast cancer + stage) ──────────────
    condition = raw.select(
        pl.col("patient_id").alias("person_id"),
        pl.lit(CONDITION["malignant_neoplasm_breast"]).cast(pl.Int64).alias("condition_concept_id"),
        pl.col("year_of_diagnosis").map_elements(
            _dx_date, return_dtype=pl.Date
        ).alias("condition_start_date"),
        pl.lit(None).cast(pl.Date).alias("condition_end_date"),
        pl.lit(TYPE_CONCEPT["registry"]).cast(pl.Int64).alias("condition_type_concept_id"),
        pl.col("primary_site").alias("condition_source_value"),
    )
    condition = cast_to_schema(condition, CONDITION_OCCURRENCE_SCHEMA)

    # ── measurement (ER, PR, HER2, tumor size) ───────────────────
    meas_rows: list[pl.DataFrame] = []
    for biomarker, concept_key in [("er_status", "er_status"),
                                    ("pr_status", "pr_status"),
                                    ("her2_status", "her2_status")]:
        m = raw.select(
            pl.col("patient_id").alias("person_id"),
            pl.lit(MEASUREMENT[concept_key]).cast(pl.Int64).alias("measurement_concept_id"),
            pl.col("year_of_diagnosis").map_elements(
                _dx_date, return_dtype=pl.Date
            ).alias("measurement_date"),
            pl.lit(None).cast(pl.Float64).alias("value_as_number"),
            pl.col(biomarker).map_elements(
                lambda v: BIOMARKER_VALUE.get(v, 0), return_dtype=pl.Int64
            ).alias("value_as_concept_id"),
            pl.lit(0).cast(pl.Int64).alias("unit_concept_id"),
            pl.col(biomarker).alias("measurement_source_value"),
        )
        meas_rows.append(m)

    # Tumor size
    tumor = raw.select(
        pl.col("patient_id").alias("person_id"),
        pl.lit(MEASUREMENT["tumor_size_cm"]).cast(pl.Int64).alias("measurement_concept_id"),
        pl.col("year_of_diagnosis").map_elements(
            _dx_date, return_dtype=pl.Date
        ).alias("measurement_date"),
        (pl.col("tumor_size_mm").cast(pl.Float64) / 10.0).alias("value_as_number"),
        pl.lit(0).cast(pl.Int64).alias("value_as_concept_id"),
        pl.lit(0).cast(pl.Int64).alias("unit_concept_id"),
        pl.lit("tumor_size_cm").alias("measurement_source_value"),
    )
    meas_rows.append(tumor)
    measurement = cast_to_schema(pl.concat(meas_rows), MEASUREMENT_SCHEMA)

    # ── procedure_occurrence ─────────────────────────────────────
    proc_map = {"Mastectomy": PROCEDURE["mastectomy"],
                "Lumpectomy": PROCEDURE["lumpectomy"]}
    procs = raw.filter(pl.col("surgery") != "None").select(
        pl.col("patient_id").alias("person_id"),
        pl.col("surgery").map_elements(
            lambda s: proc_map.get(s, 0), return_dtype=pl.Int64
        ).alias("procedure_concept_id"),
        pl.col("year_of_diagnosis").map_elements(
            _dx_date, return_dtype=pl.Date
        ).alias("procedure_date"),
        pl.lit(TYPE_CONCEPT["registry"]).cast(pl.Int64).alias("procedure_type_concept_id"),
        pl.col("surgery").alias("procedure_source_value"),
    )
    # Radiation
    rad = raw.filter(pl.col("radiation") == "Yes").select(
        pl.col("patient_id").alias("person_id"),
        pl.lit(PROCEDURE["radiation_therapy"]).cast(pl.Int64).alias("procedure_concept_id"),
        pl.col("year_of_diagnosis").map_elements(
            _dx_date, return_dtype=pl.Date
        ).alias("procedure_date"),
        pl.lit(TYPE_CONCEPT["registry"]).cast(pl.Int64).alias("procedure_type_concept_id"),
        pl.lit("Radiation").alias("procedure_source_value"),
    )
    procedure = cast_to_schema(pl.concat([procs, rad]), PROCEDURE_OCCURRENCE_SCHEMA)

    # ── drug_exposure (chemo flag only in SEER) ──────────────────
    chemo = raw.filter(pl.col("chemotherapy") == "Yes").select(
        pl.col("patient_id").alias("person_id"),
        pl.lit(PROCEDURE["chemotherapy"]).cast(pl.Int64).alias("drug_concept_id"),
        pl.col("year_of_diagnosis").map_elements(
            _dx_date, return_dtype=pl.Date
        ).alias("drug_exposure_start_date"),
        pl.lit(None).cast(pl.Date).alias("drug_exposure_end_date"),
        pl.lit(TYPE_CONCEPT["registry"]).cast(pl.Int64).alias("drug_type_concept_id"),
        pl.lit("chemotherapy_flag").alias("drug_source_value"),
    )
    drug_exposure = cast_to_schema(chemo, DRUG_EXPOSURE_SCHEMA)

    # ── death ────────────────────────────────────────────────────
    deaths = raw.filter(pl.col("vital_status") == "Dead").select(
        pl.col("patient_id").alias("person_id"),
        pl.struct(["year_of_diagnosis", "survival_months"]).map_elements(
            lambda s: _dx_date(s["year_of_diagnosis"]).replace(
                year=s["year_of_diagnosis"] + s["survival_months"] // 12,
                month=min(12, 1 + (6 + s["survival_months"]) % 12),
            ),
            return_dtype=pl.Date,
        ).alias("death_date"),
        pl.lit(TYPE_CONCEPT["registry"]).cast(pl.Int64).alias("death_type_concept_id"),
        pl.lit(CONDITION["malignant_neoplasm_breast"]).cast(pl.Int64).alias("cause_concept_id"),
        pl.lit("breast_cancer").alias("cause_source_value"),
    )
    death = cast_to_schema(deaths, DEATH_SCHEMA)

    return {
        "person": person,
        "observation_period": obs_period,
        "condition_occurrence": condition,
        "measurement": measurement,
        "procedure_occurrence": procedure,
        "drug_exposure": drug_exposure,
        "death": death,
    }


def run(data_path: str | Path) -> dict[str, pl.DataFrame]:
    """Full extract-transform pipeline for SEER data."""
    raw = extract(data_path)
    return transform(raw)
