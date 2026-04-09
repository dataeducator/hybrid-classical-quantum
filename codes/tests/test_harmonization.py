"""Tests for Phase 1 - Data Harmonization.

Each module is tested independently:
  - Synthetic generators produce correct shapes and values
  - ETL pipelines produce valid OMOP CDM tables
  - Cohort builder merges all sources into a single wide table

Note: The package 1_data_harmonization starts with a digit so it
cannot be imported with a normal import statement.  We use
importlib.import_module throughout.
"""

from __future__ import annotations

import importlib
from pathlib import Path

import polars as pl
import pytest

_pkg = "1_data_harmonization"


def _import(submodule: str):
    return importlib.import_module(f"{_pkg}.{submodule}")


_omop = _import("omop_schema")

PERSON_SCHEMA = _omop.PERSON_SCHEMA
cast_to_schema = _omop.cast_to_schema
empty_frame = _omop.empty_frame


class TestOMOPSchema:
    def test_empty_frame_has_correct_columns(self) -> None:
        df = empty_frame(PERSON_SCHEMA)
        assert df.columns == list(PERSON_SCHEMA.keys())
        assert df.height == 0

    def test_cast_to_schema_fills_missing_cols(self) -> None:
        df = pl.DataFrame({"person_id": ["P1"], "gender_concept_id": [8532]})
        result = cast_to_schema(df, PERSON_SCHEMA)
        assert set(result.columns) == set(PERSON_SCHEMA.keys())
        assert result["race_concept_id"].null_count() == 1

    def test_cast_to_schema_drops_extra_cols(self) -> None:
        df = pl.DataFrame({
            "person_id": ["P1"],
            "gender_concept_id": [8532],
            "extra_col": ["foo"],
        })
        result = cast_to_schema(df, PERSON_SCHEMA)
        assert "extra_col" not in result.columns


class TestSEERSynth:
    def test_generates_correct_count(self, tmp_path: Path) -> None:
        mod = _import("synthetic.seer_synth")
        df = mod.generate_seer(n=50, seed=1, output_dir=tmp_path)
        assert df.height == 50

    def test_all_tnbc(self, tmp_path: Path) -> None:
        mod = _import("synthetic.seer_synth")
        df = mod.generate_seer(n=100, seed=2, output_dir=tmp_path)
        assert (df["er_status"] == "Negative").all()
        assert (df["pr_status"] == "Negative").all()
        assert (df["her2_status"] == "Negative").all()

    def test_csv_written(self, tmp_path: Path) -> None:
        mod = _import("synthetic.seer_synth")
        mod.generate_seer(n=10, seed=3, output_dir=tmp_path)
        assert (tmp_path / "seer_tnbc.csv").exists()


class TestMIMICSynth:
    def test_generates_all_tables(self, tmp_path: Path) -> None:
        mod = _import("synthetic.mimic_synth")
        frames = mod.generate_mimic(n=30, seed=1, output_dir=tmp_path)
        expected = {"patients", "admissions", "diagnoses_icd",
                    "labevents", "prescriptions", "discharge_notes"}
        assert set(frames.keys()) == expected

    def test_patient_count(self, tmp_path: Path) -> None:
        mod = _import("synthetic.mimic_synth")
        frames = mod.generate_mimic(n=50, seed=2, output_dir=tmp_path)
        assert frames["patients"].height == 50


class TestTCGASynth:
    def test_generates_all_tables(self, tmp_path: Path) -> None:
        mod = _import("synthetic.tcga_synth")
        frames = mod.generate_tcga(n=40, seed=1, output_dir=tmp_path)
        assert "patient_clinical" in frames
        assert "sample_clinical" in frames
        assert "mutations" in frames

    def test_patient_count(self, tmp_path: Path) -> None:
        mod = _import("synthetic.tcga_synth")
        frames = mod.generate_tcga(n=50, seed=2, output_dir=tmp_path)
        assert frames["patient_clinical"].height == 50

    def test_all_tnbc(self, tmp_path: Path) -> None:
        mod = _import("synthetic.tcga_synth")
        frames = mod.generate_tcga(n=20, seed=3, output_dir=tmp_path)
        samples = frames["sample_clinical"]
        assert (samples["ER_STATUS_BY_IHC"] == "Negative").all()
        assert (samples["PR_STATUS_BY_IHC"] == "Negative").all()
        assert (samples["HER2_FISH_STATUS"] == "Negative").all()

    def test_has_survival_data(self, tmp_path: Path) -> None:
        mod = _import("synthetic.tcga_synth")
        frames = mod.generate_tcga(n=20, seed=4, output_dir=tmp_path)
        patients = frames["patient_clinical"]
        assert "OS_STATUS" in patients.columns
        assert "OS_MONTHS" in patients.columns
        assert "MUTATION_COUNT" in patients.columns

    def test_has_mutations(self, tmp_path: Path) -> None:
        mod = _import("synthetic.tcga_synth")
        frames = mod.generate_tcga(n=20, seed=5, output_dir=tmp_path)
        mutations = frames["mutations"]
        assert mutations.height > 0
        assert "gene" in mutations.columns
        assert "mutationType" in mutations.columns

    def test_csvs_written(self, tmp_path: Path) -> None:
        mod = _import("synthetic.tcga_synth")
        mod.generate_tcga(n=10, seed=6, output_dir=tmp_path)
        assert (tmp_path / "patient_clinical.csv").exists()
        assert (tmp_path / "sample_clinical.csv").exists()
        assert (tmp_path / "mutations.csv").exists()


@pytest.fixture()
def seer_data(tmp_path: Path) -> Path:
    mod = _import("synthetic.seer_synth")
    mod.generate_seer(n=100, seed=10, output_dir=tmp_path)
    return tmp_path


@pytest.fixture()
def mimic_data(tmp_path: Path) -> Path:
    mod = _import("synthetic.mimic_synth")
    mod.generate_mimic(n=80, seed=11, output_dir=tmp_path)
    return tmp_path


@pytest.fixture()
def tcga_data(tmp_path: Path) -> Path:
    mod = _import("synthetic.tcga_synth")
    mod.generate_tcga(n=60, seed=12, output_dir=tmp_path)
    return tmp_path


class TestSEERETL:
    def test_produces_omop_tables(self, seer_data: Path) -> None:
        mod = _import("etl.seer_etl")
        result = mod.run(seer_data)
        assert "person" in result
        assert "condition_occurrence" in result
        assert "measurement" in result

    def test_person_schema_valid(self, seer_data: Path) -> None:
        mod = _import("etl.seer_etl")
        result = mod.run(seer_data)
        person = result["person"]
        assert set(person.columns) == set(PERSON_SCHEMA.keys())
        assert person["data_source"].unique().to_list() == ["SEER"]

    def test_person_count_matches(self, seer_data: Path) -> None:
        mod = _import("etl.seer_etl")
        result = mod.run(seer_data)
        assert result["person"].height == 100


class TestMIMICETL:
    def test_produces_omop_tables(self, mimic_data: Path) -> None:
        mod = _import("etl.mimic_etl")
        result = mod.run(mimic_data)
        assert "person" in result
        assert "measurement" in result
        assert "drug_exposure" in result

    def test_person_source_tag(self, mimic_data: Path) -> None:
        mod = _import("etl.mimic_etl")
        result = mod.run(mimic_data)
        assert result["person"]["data_source"].unique().to_list() == ["MIMIC-IV"]


class TestTCGAETL:
    def test_produces_omop_tables(self, tcga_data: Path) -> None:
        mod = _import("etl.tcga_etl")
        result = mod.run(tcga_data)
        assert "person" in result
        assert "condition_occurrence" in result

    def test_person_source_tag(self, tcga_data: Path) -> None:
        mod = _import("etl.tcga_etl")
        result = mod.run(tcga_data)
        assert result["person"]["data_source"].unique().to_list() == ["TCGA-BRCA"]

    def test_person_count_matches(self, tcga_data: Path) -> None:
        mod = _import("etl.tcga_etl")
        result = mod.run(tcga_data)
        assert result["person"].height == 60

    def test_has_measurements(self, tcga_data: Path) -> None:
        mod = _import("etl.tcga_etl")
        result = mod.run(tcga_data)
        assert "measurement" in result
        meas = result["measurement"]
        sources = meas["measurement_source_value"].unique().to_list()
        assert "mutation_count" in sources
        assert "er_status_by_ihc" in sources

    def test_has_observations(self, tcga_data: Path) -> None:
        mod = _import("etl.tcga_etl")
        result = mod.run(tcga_data)
        assert "observation" in result
        obs = result["observation"]
        sources = obs["observation_source_value"].unique().to_list()
        assert "os_status" in sources
        assert "ajcc_pathologic_stage" in sources

    def test_person_ids_prefixed(self, tcga_data: Path) -> None:
        mod = _import("etl.tcga_etl")
        result = mod.run(tcga_data)
        ids = result["person"]["person_id"].to_list()
        assert all(pid.startswith("TCGA_") for pid in ids)


class TestCohortBuilder:
    def test_build_merges_all_sources(
        self, seer_data: Path, mimic_data: Path,
        tcga_data: Path, tmp_path: Path,
    ) -> None:
        builder = _import("cohort_builder")
        seer_etl = _import("etl.seer_etl")
        mimic_etl = _import("etl.mimic_etl")
        tcga_etl = _import("etl.tcga_etl")

        source_tables = {
            "seer": seer_etl.run(seer_data),
            "mimic": mimic_etl.run(mimic_data),
            "tcga": tcga_etl.run(tcga_data),
        }
        out_file = tmp_path / "cohort.parquet"
        cohort = builder.build(source_tables, output_path=out_file)

        expected_total = 100 + 80 + 60
        assert cohort.height == expected_total
        assert out_file.exists()

    def test_parquet_readable(
        self, seer_data: Path, tmp_path: Path,
    ) -> None:
        builder = _import("cohort_builder")
        seer_etl = _import("etl.seer_etl")

        source_tables = {"seer": seer_etl.run(seer_data)}
        out_file = tmp_path / "test.parquet"
        builder.build(source_tables, output_path=out_file)

        reloaded = pl.read_parquet(out_file)
        assert reloaded.height == 100
        assert "person_id" in reloaded.columns

    def test_data_source_column_present(
        self, seer_data: Path, mimic_data: Path, tmp_path: Path,
    ) -> None:
        builder = _import("cohort_builder")
        seer_etl = _import("etl.seer_etl")
        mimic_etl = _import("etl.mimic_etl")

        source_tables = {
            "seer": seer_etl.run(seer_data),
            "mimic": mimic_etl.run(mimic_data),
        }
        out_file = tmp_path / "test2.parquet"
        cohort = builder.build(source_tables, output_path=out_file)
        assert "data_source" in cohort.columns
        sources = cohort["data_source"].unique().sort().to_list()
        assert "MIMIC-IV" in sources
        assert "SEER" in sources
