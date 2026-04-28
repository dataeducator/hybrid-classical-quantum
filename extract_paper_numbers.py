"""Extract additional numbers for the paper:
1. Cox PH subgroup C-indices (with proper race labels)
2. Cox PH top coefficients by |β|
3. Top features by hazard ratio with 95% CI
"""
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index

RACE_LABELS = {
    'race_0': 'Non-Hispanic White',
    'race_1': 'Non-Hispanic Black',
    'race_2': 'Hispanic',
    'race_3': 'Non-Hispanic Asian/Pacific Islander',
    'race_4': 'Non-Hispanic American Indian/Alaska Native',
    'race_5': 'Non-Hispanic Unknown Race',
}

# Reproduce the same preprocessing as run_survival_residual.py
df = pd.read_csv('breast_cancer_4quantum.csv')
durations_all = pd.to_numeric(df['Survival_months'], errors='coerce')
events_all = (df['Vital_status_recode_study_cutoff_used_'].astype(str).str.lower() == 'dead').astype(float)
mask = ~durations_all.isna() & (durations_all > 0)
df = df[mask].copy()
df['_duration'] = durations_all[mask].values
df['_event'] = events_all[mask].values

initial_drops = [
    'Patient_ID', 'age_group', 'Sex_no_total_', 'RX_Summ_Systemic_Sur_Seq_2007_',
    'Diagnostic_Confirmation', 'Histologic_Type_ICD_O_3', 'Primary_Site_labeled',
    'Combined_Summary_Stage_with_Expanded_Regional_Codes_2004_',
    'Reason_no_cancer_directed_surgery', 'Grade_Clinical_2018_',
    'COD_to_site_recode', 'Vital_status_recode_study_cutoff_used_',
    'Grade_Pathological_2018_', 'Race_and_origin_recode_NHW,_NHB,_NHAIAN,_NHAPI,_Hispanic_no_total',
    'Median_household_income_inflation_adj_to_2023', 'Age_recode_with_<1_year_olds_and_90_',
    'Rural_Urban_Continuum_Code', 'stage_encoded',
    'Survival_months', 'survival_60_months', 'Year_of_follow_up_recode',
]
df = df.drop(columns=[c for c in initial_drops if c in df.columns])

text_cols = ['Summary_stage_2000_1998_2017_', 'race_encoded', 'Laterality',
             'Marital_status_at_diagnosis', 'Breast_Subtype_2010_']
for col in text_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.lower()

stage_map = {'localized': 0, 'regional': 1, 'distant': 2}
df = df[df['Summary_stage_2000_1998_2017_'].isin(stage_map.keys())]
df['stage_cleaned'] = df['Summary_stage_2000_1998_2017_'].map(stage_map)

df['Time_from_diagnosis_to_treatment_in_days_recode'] = pd.to_numeric(
    df['Time_from_diagnosis_to_treatment_in_days_recode'], errors='coerce').fillna(0)

col_tumor = 'Tumor_Size_Summary_2016_'
df[col_tumor] = pd.to_numeric(df[col_tumor], errors='coerce')
if 'CS_tumor_size_2004_2015_' in df.columns:
    cs = pd.to_numeric(df['CS_tumor_size_2004_2015_'], errors='coerce')
    df[col_tumor] = df[col_tumor].fillna(cs)
df[col_tumor] = df[col_tumor].fillna(df[col_tumor].median())

laterality_map = {'left - origin of primary': 0, 'left': 0,
                  'right - origin of primary': 1, 'right': 1}
df['Laterality'] = df['Laterality'].map(laterality_map).fillna(-1)

df = pd.get_dummies(df, columns=['race_encoded', 'Marital_status_at_diagnosis', 'Breast_Subtype_2010_'],
                    prefix=['race', 'marital', 'subtype'], dtype=float)
if 'Summary_stage_2000_1998_2017_' in df.columns:
    df = df.drop(columns=['Summary_stage_2000_1998_2017_'])

quantum_features = [
    'numeric_age', 'Tumor_Size_Summary_2016_', 'income_encoded',
    'years_since_2010', 'Time_from_diagnosis_to_treatment_in_days_recode',
    'income_age_ratio', 'stage_cleaned'
]
exclude = quantum_features + ['_duration', '_event']
classical_features = [c for c in df.columns
                      if c not in exclude and df[c].dtype in ['float64', 'float32', 'int64', 'int32']]
df_clean = df.dropna(subset=quantum_features + ['_duration', '_event'])
for col in classical_features:
    if df_clean[col].isna().any():
        m = df_clean[col].median()
        if pd.isna(m):
            m = 0.0
        df_clean[col] = df_clean[col].fillna(m)

train_idx, test_idx = train_test_split(
    np.arange(len(df_clean)), test_size=0.2, random_state=42,
    stratify=df_clean['_event']
)
df_train = df_clean.iloc[train_idx].copy()
df_test = df_clean.iloc[test_idx].copy()

q_scaler = MinMaxScaler()
c_scaler = StandardScaler()
Xq_train_scaled = q_scaler.fit_transform(df_train[quantum_features])
Xc_train_scaled = c_scaler.fit_transform(df_train[classical_features])
Xq_test_scaled = q_scaler.transform(df_test[quantum_features])
Xc_test_scaled = c_scaler.transform(df_test[classical_features])

all_feature_names = quantum_features + classical_features
cox_train_df = pd.DataFrame(np.hstack([Xq_train_scaled, Xc_train_scaled]), columns=all_feature_names)
cox_train_df['duration'] = df_train['_duration'].values
cox_train_df['event'] = df_train['_event'].values
cox_test_df = pd.DataFrame(np.hstack([Xq_test_scaled, Xc_test_scaled]), columns=all_feature_names)
cox_test_df['duration'] = df_test['_duration'].values
cox_test_df['event'] = df_test['_event'].values

keep_cols = [c for c in all_feature_names if cox_train_df[c].std() > 1e-6]

cph = CoxPHFitter(penalizer=0.01)
cph.fit(cox_train_df[keep_cols + ['duration', 'event']],
        duration_col='duration', event_col='event', show_progress=False)

# 1. Overall Cox PH C-index
cox_logh_test = cph.predict_log_partial_hazard(cox_test_df[keep_cols]).values
test_cidx = concordance_index(df_test['_duration'].values, -cox_logh_test, df_test['_event'].values)
print(f"Cox PH overall test C-index: {test_cidx:.4f}")

# 2. Cox PH SUBGROUP C-indices (with proper race labels)
race_cols = [c for c in classical_features if c.startswith('race_')]
race_idx_in_classical = [classical_features.index(c) for c in race_cols]
test_race_groups = Xc_test_scaled[:, race_idx_in_classical].argmax(axis=1)

print(f"\n=== Cox PH SUBGROUP C-INDEX (with proper labels) ===")
print(f"  {'Subgroup':<45} {'N':>5} {'Events':>7} {'C-index':>9}")
print("  " + "-" * 70)
cox_subgroup_results = {}
cindices_cox = []
for g, col in enumerate(race_cols):
    label = RACE_LABELS.get(col, col)
    mask = test_race_groups == g
    n = int(mask.sum())
    n_evt = int(df_test['_event'].values[mask].sum()) if n > 0 else 0
    if n < 20 or n_evt < 5:
        print(f"  SKIP {label}: N={n}, events={n_evt}")
        continue
    ci = concordance_index(df_test['_duration'].values[mask], -cox_logh_test[mask], df_test['_event'].values[mask])
    cox_subgroup_results[label] = {'n': n, 'events': n_evt, 'c_index': float(ci)}
    cindices_cox.append(ci)
    print(f"  {label:<45} {n:>5} {n_evt:>7} {ci:>9.4f}")
cox_gap = max(cindices_cox) - min(cindices_cox) if len(cindices_cox) >= 2 else float('nan')
print(f"\n  Cox PH subgroup C-index gap: {cox_gap:.4f}")

# 3. Cox PH top coefficients
print(f"\n=== Cox PH TOP COEFFICIENTS (by |β|) ===")
summary_df = cph.summary.copy()
summary_df['abs_coef'] = summary_df['coef'].abs()
top_features = summary_df.sort_values('abs_coef', ascending=False).head(15)

print(f"  {'Feature':<45} {'β (logHR)':>10} {'HR':>7} {'95% CI':>20} {'p':>8}")
print("  " + "-" * 95)
cox_coefs = []
for feat, row in top_features.iterrows():
    hr = row['exp(coef)']
    lo = row['exp(coef) lower 95%']
    hi = row['exp(coef) upper 95%']
    p = row['p']
    coef = row['coef']
    print(f"  {feat:<45} {coef:>10.4f} {hr:>7.3f} ({lo:.3f}, {hi:.3f})  {p:>8.2e}")
    cox_coefs.append({
        'feature': feat,
        'coef': float(coef),
        'hr': float(hr),
        'hr_ci_low': float(lo),
        'hr_ci_high': float(hi),
        'p_value': float(p),
    })

# 4. Save everything
output = {
    'cox_overall_test_cindex': float(test_cidx),
    'cox_subgroup_cindex': cox_subgroup_results,
    'cox_subgroup_gap': float(cox_gap) if not np.isnan(cox_gap) else None,
    'cox_top_coefficients': cox_coefs,
    'data': {
        'train_size': len(df_train),
        'test_size': len(df_test),
        'event_rate_train': float(df_train['_event'].mean()),
        'event_rate_test': float(df_test['_event'].mean()),
    },
}
with open('cox_paper_numbers.json', 'w') as f:
    json.dump(output, f, indent=2)
print(f"\nSaved: cox_paper_numbers.json")
