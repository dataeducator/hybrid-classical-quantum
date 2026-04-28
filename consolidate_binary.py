"""Consolidate all binary classification results into a clean CSV.

Reads:
  - binary_honest_results.json (v1, MLP, XGBoost, LightGBM)
  - binary_v2_result.json
  - binary_v3_result.json
  - binary_v3_full_result.json
  - binary_v4_result.json

Computes:
  - Ensemble (v3-full + MLP probabilities averaged)

Writes:
  - TNBC_Binary_Ablation_Clean.csv
"""
import json
import numpy as np
import pandas as pd
from sklearn.metrics import (roc_auc_score, roc_curve, precision_score,
                              recall_score, f1_score)

rows = []

# 1. From binary_honest_results.json: MLP, v1, XGBoost, LightGBM
honest = json.load(open('binary_honest_results.json'))
for entry in honest['ablation']:
    rows.append({
        'Model': entry['Model'],
        'AUC': entry['AUC'],
        'Precision': entry['Precision'],
        'Recall': entry['Recall'],
        'F1': entry['F1'],
        'Time_sec': entry['Time_sec'],
        'Train_size': entry['Train_size'],
        'Status': entry['Status'],
    })

# 2. v2, v3, v3-full, v4 from individual JSONs
file_map = [
    ('binary_v2_result.json', 'HybridRealQ_v2 (subsample, class-weighted)', 2000),
    ('binary_v3_result.json', 'HybridRealQ_v3 (subsample, class-weighted)', 2000),
    ('binary_v3_full_result.json', 'HybridRealQ_v3 (full data, class-weighted)', 15673),
    ('binary_v4_result.json', 'HybridRealQ_v4 (full data, output scaling)', 15673),
]
for fname, label, train_size in file_map:
    r = json.load(open(fname))
    rows.append({
        'Model': label,
        'AUC': round(r['auc'], 4),
        'Precision': round(r['precision'], 4),
        'Recall': round(r['recall'], 4),
        'F1': round(r['f1'], 4),
        'Time_sec': round(r['time_sec'], 1),
        'Train_size': train_size,
        'Status': 'OK',
    })

# 3. Compute Ensemble (v3-full + MLP) — needs the saved probs from v3-full
v3_full = json.load(open('binary_v3_full_result.json'))
v3_probs = np.array(v3_full['probs']) if v3_full.get('probs') else None

# We need MLP probs too; let's recompute them quickly from the saved test set
if v3_probs is not None:
    # We need MLP probs on the same FULL test set. The honest run saved the MLP results
    # but not its probs in the same format. Let me recompute.
    # Actually, the simplest path: re-run MLP very quickly (it took 2.9s), get probs, ensemble.
    print("Computing ensemble (re-running MLP for probs, ~3s)...")
    import os
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import random
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MinMaxScaler, StandardScaler

    def set_determinism(seed=42):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True

    set_determinism(42)

    # Quick reproduction of preprocessing
    df = pd.read_csv('breast_cancer_4quantum.csv')
    vital = df['Vital_status_recode_study_cutoff_used_'].astype(str).str.lower()
    sm = pd.to_numeric(df['Survival_months'], errors='coerce')
    target = pd.Series(np.nan, index=df.index, dtype=float)
    target[sm >= 60] = 1.0
    target[(sm < 60) & (vital == 'dead')] = 0.0
    df['survival_60_months'] = target
    df = df.dropna(subset=['survival_60_months'])
    df['survival_60_months'] = df['survival_60_months'].astype(int)

    initial_drops = ['Patient_ID', 'age_group', 'Sex_no_total_', 'RX_Summ_Systemic_Sur_Seq_2007_',
        'Diagnostic_Confirmation', 'Histologic_Type_ICD_O_3', 'Primary_Site_labeled',
        'Combined_Summary_Stage_with_Expanded_Regional_Codes_2004_',
        'Reason_no_cancer_directed_surgery', 'Grade_Clinical_2018_',
        'COD_to_site_recode', 'Vital_status_recode_study_cutoff_used_',
        'Grade_Pathological_2018_', 'Race_and_origin_recode_NHW,_NHB,_NHAIAN,_NHAPI,_Hispanic_no_total',
        'Median_household_income_inflation_adj_to_2023', 'Age_recode_with_<1_year_olds_and_90_',
        'Rural_Urban_Continuum_Code', 'stage_encoded', 'Year_of_follow_up_recode']
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

    quantum_features = ['numeric_age', 'Tumor_Size_Summary_2016_', 'income_encoded',
                        'years_since_2010', 'Time_from_diagnosis_to_treatment_in_days_recode',
                        'income_age_ratio', 'stage_cleaned']
    target_col = 'survival_60_months'
    exclude = quantum_features + [target_col, 'Survival_months', 'Vital_status_recode_study_cutoff_used_']
    classical_features = [c for c in df.columns if c not in exclude
                          and df[c].dtype in ['float64', 'float32', 'int64', 'int32']]
    df_clean = df.dropna(subset=quantum_features + [target_col])
    for col in classical_features:
        if df_clean[col].isna().any():
            m = df_clean[col].median()
            if pd.isna(m): m = 0.0
            df_clean[col] = df_clean[col].fillna(m)

    Xq_train, Xq_test, Xc_train, Xc_test, y_train, y_test = train_test_split(
        df_clean[quantum_features], df_clean[classical_features], df_clean[target_col],
        test_size=0.2, random_state=42, stratify=df_clean[target_col])

    q_scaler = MinMaxScaler()
    c_scaler = StandardScaler()
    Xq_train_scaled = q_scaler.fit_transform(Xq_train)
    Xc_train_scaled = c_scaler.fit_transform(Xc_train)
    Xq_test_scaled = q_scaler.transform(Xq_test)
    Xc_test_scaled = c_scaler.transform(Xc_test)

    X_train_q = torch.tensor(np.pi * Xq_train_scaled, dtype=torch.float32)
    X_test_q = torch.tensor(np.pi * Xq_test_scaled, dtype=torch.float32)
    X_train_c = torch.tensor(Xc_train_scaled, dtype=torch.float32)
    X_test_c = torch.tensor(Xc_test_scaled, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.values, dtype=torch.float32).reshape(-1, 1)
    y_test_t = torch.tensor(y_test.values, dtype=torch.float32).reshape(-1, 1)

    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32)

    n_classical = X_train_c.shape[1]

    class ClassicalMLP(nn.Module):
        def __init__(self, n_features):
            super().__init__()
            self.fc1 = nn.Linear(n_features, 32)
            self.fc2 = nn.Linear(32, 16)
            self.fc3 = nn.Linear(16, 1)
        def forward(self, x_q, x_c):
            x = F.relu(self.fc1(x_c))
            x = F.relu(self.fc2(x))
            return self.fc3(x)

    set_determinism(42)
    mlp = ClassicalMLP(n_classical)
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = torch.optim.Adam(mlp.parameters(), lr=0.001)
    for _ in range(100):
        mlp.train()
        opt.zero_grad()
        loss = crit(mlp(X_train_q, X_train_c).squeeze(), y_train_t.squeeze())
        loss.backward()
        opt.step()
    mlp.eval()
    with torch.no_grad():
        mlp_probs = torch.sigmoid(mlp(X_test_q, X_test_c).squeeze()).cpu().numpy()

    # Ensemble
    ens_probs = (v3_probs + mlp_probs) / 2.0
    y_test_np = y_test.values
    ens_auc = roc_auc_score(y_test_np, ens_probs)
    fpr, tpr, thresh = roc_curve(y_test_np, ens_probs)
    ens_t = thresh[(tpr - fpr).argmax()]
    ens_preds = (ens_probs >= ens_t).astype(int)
    ens_prec = precision_score(y_test_np, ens_preds, zero_division=0)
    ens_rec = recall_score(y_test_np, ens_preds, zero_division=0)
    ens_f1 = f1_score(y_test_np, ens_preds, zero_division=0)
    print(f"Ensemble: AUC={ens_auc:.4f} Prec={ens_prec:.4f} Rec={ens_rec:.4f} F1={ens_f1:.4f}")

    rows.append({
        'Model': 'Ensemble (HybridRealQ_v3-full + Classical MLP)',
        'AUC': round(ens_auc, 4),
        'Precision': round(ens_prec, 4),
        'Recall': round(ens_rec, 4),
        'F1': round(ens_f1, 4),
        'Time_sec': round(v3_full['time_sec'] + 3, 1),
        'Train_size': 15673,
        'Status': 'OK',
    })


# Reorder rows by AUC (high to low) for clarity
df_out = pd.DataFrame(rows)
# But preserve original ordering — show in a logical groupings
order = ['Classical MLP', 'HybridRealQ v1', 'HybridRealQ_v2', 'HybridRealQ_v3 (subsample',
         'HybridRealQ_v3 (full', 'HybridRealQ_v4', 'Ensemble', 'XGBoost', 'LightGBM']
def sort_key(model_name):
    for i, prefix in enumerate(order):
        if model_name.startswith(prefix):
            return i
    return len(order)
df_out = df_out.sort_values('Model', key=lambda c: c.map(sort_key)).reset_index(drop=True)

print("\n=== FINAL BINARY ABLATION (clean data) ===")
print(df_out.to_string(index=False))
df_out.to_csv('TNBC_Binary_Ablation_Clean.csv', index=False)
print(f"\nSaved: TNBC_Binary_Ablation_Clean.csv ({len(df_out)} rows)")
