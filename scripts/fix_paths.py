"""Fix file paths in scripts after reorganization.

Scripts are now in scripts/ and outputs go to results/.
This patches all the *.py files to reference the new locations.

Run from project root: python scripts/fix_paths.py
"""
import os
import re
import glob

# Project root = parent of this script's dir
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
os.chdir(ROOT)

# Files to update
PY_FILES = (
    glob.glob('run_*.py') +
    glob.glob('scripts/*.py')
)

# Output filenames that should be redirected to results/
OUTPUT_FILES = [
    # CSV outputs
    'results/TNBC_Survival_Ablation_Clean.csv',
    'results/TNBC_Survival_Ablation_Results.csv',
    'results/TNBC_Binary_Ablation_Clean.csv',
    'results/TNBC_Binary_Honest_Results.csv',
    # JSON outputs
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


def patch_file(path):
    with open(path, 'r', encoding='utf-8') as f:
        content = f.read()
    original = content

    for fname in OUTPUT_FILES:
        # Replace 'fname' (in quotes) with 'results/fname' but skip if already prefixed
        # Match: 'filename.json' or "filename.json" not preceded by results/
        pattern = re.compile(r"(?<!results[/\\])(?<!results/)([\"'])({})\1".format(re.escape(fname)))
        content = pattern.sub(r"\1results/\2\1", content)

    if content != original:
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  patched: {path}")
    else:
        print(f"  unchanged: {path}")


def main():
    print(f"Project root: {ROOT}")
    print(f"Patching {len(PY_FILES)} files...")
    for p in PY_FILES:
        patch_file(p)


if __name__ == '__main__':
    main()
