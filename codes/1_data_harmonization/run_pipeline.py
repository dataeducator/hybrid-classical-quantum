"""CLI entry-point: generate synthetic data, run all ETL pipelines,
and produce the unified TNBC cohort Parquet file.

Usage:
    python -m 1_data_harmonization.run_pipeline [--n-seer 2000] [--n-mimic 1500] ...

Data-lineage:
    synthetic generators -> source CSVs
    source CSVs -> ETL mappers -> OMOP tables
    OMOP tables -> cohort_builder -> tnbc_cohort.parquet
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from dotenv import load_dotenv

from .cohort_builder import build
from .etl import mimic_etl, seer_etl, tcga_etl
from .synthetic.mimic_synth import generate_mimic
from .synthetic.seer_synth import generate_seer
from .synthetic.tcga_synth import generate_tcga


def main() -> None:
    parser = argparse.ArgumentParser(
        description="TNBC Data Harmonization Pipeline"
    )
    parser.add_argument("--n-seer", type=int, default=2000)
    parser.add_argument("--n-mimic", type=int, default=1500)
    parser.add_argument("--n-tcga", type=int, default=160)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=str, default="data/output/tnbc_cohort.parquet")
    parser.add_argument("--skip-synth", action="store_true",
                        help="Skip synthetic generation; read existing CSVs")
    parser.add_argument("--tcga-api", action="store_true",
                        help="Fetch TCGA-BRCA data from cBioPortal API instead of synthetic")
    args = parser.parse_args()

    load_dotenv()

    seer_path = Path("data/synthetic/seer")
    mimic_path = Path("data/synthetic/mimic")
    tcga_path = Path("data/synthetic/tcga")

    t0 = time.perf_counter()

    # ── Step 1: Generate synthetic data ───────────────────────────
    if not args.skip_synth and not args.tcga_api:
        print("[1/3] Generating synthetic data ...")
        generate_seer(n=args.n_seer, seed=args.seed, output_dir=seer_path)
        print(f"      SEER:      {args.n_seer} patients")
        generate_mimic(n=args.n_mimic, seed=args.seed + 1, output_dir=mimic_path)
        print(f"      MIMIC:     {args.n_mimic} patients")
        generate_tcga(n=args.n_tcga, seed=args.seed + 2, output_dir=tcga_path)
        print(f"      TCGA-BRCA: {args.n_tcga} patients")
    else:
        print("[1/3] Skipping synthetic generation")

    # ── Step 2: Run ETL pipelines ─────────────────────────────────
    print("[2/3] Running ETL pipelines -> OMOP CDM v5.4 ...")
    seer_omop = seer_etl.run(seer_path)
    print(f"      SEER:      {seer_omop['person'].height} persons mapped")
    mimic_omop = mimic_etl.run(mimic_path)
    print(f"      MIMIC:     {mimic_omop['person'].height} persons mapped")

    if args.tcga_api:
        print("      TCGA-BRCA: Fetching from cBioPortal API ...")
        tcga_omop = tcga_etl.run(None)  # API mode
    else:
        tcga_omop = tcga_etl.run(tcga_path)
    print(f"      TCGA-BRCA: {tcga_omop['person'].height} persons mapped")

    # ── Step 3: Build unified cohort ──────────────────────────────
    print("[3/3] Building unified TNBC cohort ...")
    source_tables = {
        "seer": seer_omop,
        "mimic": mimic_omop,
        "tcga": tcga_omop,
    }
    cohort = build(source_tables, output_path=args.output)

    elapsed = time.perf_counter() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Cohort: {cohort.height} patients x {cohort.width} columns")
    print(f"Output: {Path(args.output).resolve()}")
    print("\nData sources in cohort:")
    if "data_source" in cohort.columns:
        for row in cohort.group_by("data_source").len().sort("data_source").iter_rows():
            print(f"  {row[0]}: {row[1]} patients")


if __name__ == "__main__":
    main()
