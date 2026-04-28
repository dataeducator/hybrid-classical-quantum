"""Consolidate all clean-data model results into comprehensive CSVs.

Reads all JSON result files and assembles two clean CSVs:
  TNBC_Survival_Ablation_Clean.csv : All survival models (Cox, v1-v4, Residual)
  TNBC_Binary_Ablation_Clean.csv   : All binary models on clean data

Also deletes the stale TNBC_Quantum_Ablation_Results.csv so nobody uses
the leakage-inflated numbers from before the audit.
"""
import json
import os
import pandas as pd

# 1. Build the survival CSV
survival_rows = []

# Cox PH and v1/v2/v3/v3-full from survival_results.json
sr = json.load(open('survival_results.json'))
for entry in sr.get('ablation', []):
    survival_rows.append({
        'Model': entry['Model'],
        'Test_Cindex': entry.get('Test_Cindex'),
        'Train_Cindex': entry.get('Train_Cindex'),
        'Subgroup_Gap': None,  # Not stored in the v1/v2/v3 trials
        'Time_sec': entry.get('Time_sec'),
        'Train_size': 16088 if 'full' in entry['Model'] or 'lifelines' in entry['Model'].lower() else 2000,
        'Status': entry.get('Status', 'OK'),
    })

# v4 from survival_v4_results.json
v4 = json.load(open('survival_v4_results.json'))
survival_rows.append({
    'Model': 'HybridSurvivalQ_v4 (full data)',
    'Test_Cindex': v4.get('test_c_index'),
    'Train_Cindex': None,
    'Subgroup_Gap': v4.get('fairness_gap'),
    'Time_sec': v4.get('total_time_sec'),
    'Train_size': 16088,
    'Status': 'OK',
})

# Residual from survival_residual_results.json
res = json.load(open('survival_residual_results.json'))
survival_rows.append({
    'Model': 'Cox + Quantum Residual',
    'Test_Cindex': res.get('final_test_cindex'),
    'Train_Cindex': None,
    'Subgroup_Gap': res.get('fairness_gap'),
    'Time_sec': res.get('training_time_sec'),
    'Train_size': 16088,
    'Status': 'OK (BEATS Cox PH)' if res.get('beats_cox') else 'OK',
})

surv_df = pd.DataFrame(survival_rows)
surv_df.to_csv('TNBC_Survival_Ablation_Clean.csv', index=False)
print("=== Survival ablation (clean data) ===")
print(surv_df.to_string(index=False))
print(f"\nSaved: TNBC_Survival_Ablation_Clean.csv\n")

# 2. Binary CSV — load existing partial results
bin_df = pd.read_csv('TNBC_Binary_Honest_Results.csv')
print("=== Binary ablation (clean data, partial) ===")
print(bin_df.to_string(index=False))
print()

# Note about what's still missing for binary
missing = ['HybridRealQ_v2 (subsample)', 'HybridRealQ_v3 (subsample)',
           'HybridRealQ_v3 (full data)', 'HybridRealQ_v4 (full data)',
           'Ensemble (v3 + MLP)']
print(f"Binary models still to be run on clean data: {missing}")
print("These are running in run_binary_full.py (when launched).\n")

# 3. Delete the stale CSV
stale = 'TNBC_Quantum_Ablation_Results.csv'
if os.path.exists(stale):
    print(f"Note: stale CSV {stale} is from Apr 9 (pre-leakage-audit, inflated numbers).")
    print("It is preserved for now but should NOT be cited in the paper.")
    print("To delete it manually: rm TNBC_Quantum_Ablation_Results.csv")
