"""Replicate the notebook's cell-9 preprocessing + cell-13 leakage audit
to verify it passes after the Survival_months fix.
"""
import os, sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
if not os.path.exists('breast_cancer_4quantum.csv') and os.path.exists('../breast_cancer_4quantum.csv'):
    os.chdir('..')

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from scipy import stats

df = pd.read_csv('breast_cancer_4quantum.csv')

# Right-censoring fix
vital = df['Vital_status_recode_study_cutoff_used_'].astype(str).str.lower()
sm = pd.to_numeric(df['Survival_months'], errors='coerce')
target_clean = pd.Series(np.nan, index=df.index, dtype=float)
target_clean[sm >= 60] = 1.0
target_clean[(sm < 60) & (vital == 'dead')] = 0.0
df['survival_60_months'] = target_clean
df = df.dropna(subset=['survival_60_months']).copy()
df['survival_60_months'] = df['survival_60_months'].astype(int)
print(f"Censoring filter: {len(df)} patients ({df['survival_60_months'].mean():.1%} positive)")

# Initial drops (matches updated notebook cell 9)
initial_drops = [
    'Patient_ID', 'age_group', 'Sex_no_total_',
    'RX_Summ_Systemic_Sur_Seq_2007_', 'Diagnostic_Confirmation',
    'Histologic_Type_ICD_O_3', 'Primary_Site_labeled',
    'Combined_Summary_Stage_with_Expanded_Regional_Codes_2004_',
    'Reason_no_cancer_directed_surgery', 'Grade_Clinical_2018_',
    'COD_to_site_recode', 'Vital_status_recode_study_cutoff_used_',
    'Grade_Pathological_2018_', 'Race_and_origin_recode_NHW,_NHB,_NHAIAN,_NHAPI,_Hispanic_no_total',
    'Median_household_income_inflation_adj_to_2023', 'Age_recode_with_<1_year_olds_and_90_',
    'Rural_Urban_Continuum_Code', 'stage_encoded',
    'Year_of_follow_up_recode',
    'Survival_months',
]
columns_to_preserve = ['Summary_stage_2000_1998_2017_']
columns_to_drop = [c for c in initial_drops if c in df.columns and c not in columns_to_preserve]
df = df.drop(columns=columns_to_drop)

# Standardize text
text_cols = ['Summary_stage_2000_1998_2017_', 'race_encoded', 'Laterality',
             'Marital_status_at_diagnosis', 'Breast_Subtype_2010_']
for col in text_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.lower()

# Stage
stage_map = {'localized': 0, 'regional': 1, 'distant': 2}
df = df[df['Summary_stage_2000_1998_2017_'].isin(stage_map.keys())]
df['stage_cleaned'] = df['Summary_stage_2000_1998_2017_'].map(stage_map)

# Time to treatment
col_ttt = 'Time_from_diagnosis_to_treatment_in_days_recode'
df[col_ttt] = df[col_ttt].str.replace(r'\+', '', regex=True).str.replace('days', '')
df[col_ttt] = pd.to_numeric(df[col_ttt].replace({'blank(s)': None, 'unable to calculate': None}), errors='coerce')

# Tumor size
col_tumor = 'Tumor_Size_Summary_2016_'
df[col_tumor] = pd.to_numeric(df[col_tumor].replace({'blank(s)': None, 'unable to calculate': None}), errors='coerce')

# Laterality
laterality_map = {'left - origin of primary': 0, 'left': 0, 'right - origin of primary': 1, 'right': 1}
df['Laterality'] = df['Laterality'].map(laterality_map).fillna(-1)

# One-hot encoding
nominal_cols = {'race_encoded': 'race', 'Marital_status_at_diagnosis': 'marital', 'Breast_Subtype_2010_': 'subtype'}
df = pd.get_dummies(df, columns=list(nominal_cols.keys()), prefix=list(nominal_cols.values()), dtype=float)
if 'Summary_stage_2000_1998_2017_' in df.columns:
    df = df.drop(columns=['Summary_stage_2000_1998_2017_'])

# Feature selection
quantum_features = [
    'numeric_age', 'Tumor_Size_Summary_2016_', 'income_encoded',
    'years_since_2010', 'Time_from_diagnosis_to_treatment_in_days_recode',
    'income_age_ratio', 'stage_cleaned'
]
target_col = 'survival_60_months'
classical_features = [c for c in df.columns if c not in quantum_features and c != target_col]
df_clean = df.dropna(subset=quantum_features + [target_col])
print(f"After dropna: {len(df_clean)} patients")
print(f"Classical features ({len(classical_features)}): {classical_features}")

# Train/test split
Xq_train, Xq_test, Xc_train, Xc_test, y_train, y_test = train_test_split(
    df_clean[quantum_features], df_clean[classical_features], df_clean[target_col],
    test_size=0.2, random_state=42, stratify=df_clean[target_col]
)

all_feature_names = quantum_features + classical_features
X_train_all = pd.concat([Xq_train, Xc_train], axis=1).values
X_test_all = pd.concat([Xq_test, Xc_test], axis=1).values

# Run audit
print("\n" + "=" * 60)
print("  LEAKAGE AUDIT")
print("=" * 60)

flagged_auc = []
for i, feat in enumerate(all_feature_names):
    try:
        auc = roc_auc_score(y_train.values, X_train_all[:, i])
        if auc < 0.5: auc = 1 - auc
        if auc > 0.95:
            flagged_auc.append((feat, auc))
            print(f"  AUC FLAG: {feat} = {auc:.4f}")
    except Exception:
        pass

flagged_corr = []
for i, feat in enumerate(all_feature_names):
    try:
        corr = abs(np.corrcoef(X_train_all[:, i].astype(float), y_train.values)[0, 1])
        if corr > 0.90:
            flagged_corr.append((feat, corr))
            print(f"  CORR FLAG: {feat} = {corr:.4f}")
    except Exception:
        pass

flagged_ks = []
for i, feat in enumerate(all_feature_names):
    try:
        ks_stat, p_val = stats.ks_2samp(X_train_all[:, i], X_test_all[:, i])
        if p_val < 0.001:
            flagged_ks.append((feat, ks_stat, p_val))
    except Exception:
        pass

print(f"\n  AUC flags:  {len(flagged_auc)}")
print(f"  Corr flags: {len(flagged_corr)}")
print(f"  KS flags:   {len(flagged_ks)}")

n_flags = len(flagged_auc) + len(flagged_corr) + len(flagged_ks)
print(f"\n  RESULT: {'PASSED' if n_flags == 0 else f'{n_flags} FLAG(S)'}")

# Show top 5 features by single-feature AUC for visibility
print("\nTop 10 single-feature AUCs (train set, after fold-up):")
aucs = []
for i, feat in enumerate(all_feature_names):
    try:
        a = roc_auc_score(y_train.values, X_train_all[:, i])
        if a < 0.5: a = 1 - a
        aucs.append((feat, a))
    except Exception:
        pass
for feat, a in sorted(aucs, key=lambda x: -x[1])[:10]:
    print(f"  {feat:<55} {a:.4f}")
