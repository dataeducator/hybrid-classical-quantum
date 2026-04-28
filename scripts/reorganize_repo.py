"""Reorganize repository into proper folder structure.

Before:           After:
.                 .
├── *.json   →    ├── results/*.json
├── *.csv    →    ├── results/*.csv
├── *.py     →    ├── scripts/*.py (helpers)
├── run_*.py →    ├── run_*.py (entry points stay at root)
├── *.tex    →    ├── paper/*.tex
├── *.pdf    →    ├── paper/*.pdf
"""
import os
import shutil
import subprocess

# What goes where
RESULTS_FILES = [
    'results/TNBC_Survival_Ablation_Clean.csv',
    'results/TNBC_Survival_Ablation_Results.csv',
    'results/TNBC_Binary_Ablation_Clean.csv',
    'results/TNBC_Binary_Honest_Results.csv',
    'results/cox_paper_numbers.json',
    'results/survival_results.json',
    'results/survival_v4_results.json',
    'results/survival_residual_results.json',
    'results/survival_v1_full_result.json',
    'results/survival_v2_full_result.json',
    'results/binary_honest_results.json',
    'results/binary_v2_result.json',
    'results/binary_v3_result.json',
    'results/binary_v3_full_result.json',
    'results/binary_v4_result.json',
]

# Helpers move to scripts/; run_*.py stays at root as documented entry points
SCRIPT_FILES = [
    'consolidate_results.py',
    'consolidate_binary.py',
    'extract_paper_numbers.py',
    'fix_race_encoding.py',
    'update_notebook.py',
    'generate_paper_figures.py',
    'generate_presentation.py',
    'reorganize_repo.py',  # this script too
]

PAPER_FILES = [
    'updated_main.tex',
    'updated_main.pdf',
]


def git_mv(src, dst_dir):
    """Use git mv to preserve history."""
    if not os.path.exists(src):
        print(f"  SKIP (not found): {src}")
        return False
    os.makedirs(dst_dir, exist_ok=True)
    dst = os.path.join(dst_dir, os.path.basename(src))
    try:
        subprocess.run(['git', 'mv', src, dst], check=True, capture_output=True)
        print(f"  moved: {src} -> {dst}")
        return True
    except subprocess.CalledProcessError as e:
        # Fallback if file not tracked
        try:
            shutil.move(src, dst)
            print(f"  moved (untracked): {src} -> {dst}")
            return True
        except Exception as ee:
            print(f"  ERROR: {src}: {ee}")
            return False


def main():
    print("Creating directories...")
    os.makedirs('results', exist_ok=True)
    os.makedirs('scripts', exist_ok=True)
    os.makedirs('paper', exist_ok=True)

    print("\n[results/]")
    for f in RESULTS_FILES:
        git_mv(f, 'results')

    print("\n[scripts/]")
    for f in SCRIPT_FILES:
        git_mv(f, 'scripts')

    print("\n[paper/]")
    for f in PAPER_FILES:
        git_mv(f, 'paper')

    print("\nDone.")


if __name__ == '__main__':
    main()
