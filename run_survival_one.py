"""Run ONE survival model on full data with mini-batch Cox loss.

Usage:
    python run_survival_one.py --model v1
    python run_survival_one.py --model v2
"""
import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, choices=['v1', 'v2'])
args = parser.parse_args()
MODEL_NAME = args.model

import time
import json
import gc
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import pennylane as qml
import random

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from lifelines.utils import concordance_index


def set_determinism(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


set_determinism(42)
print(f"=== Running survival model: {MODEL_NAME} (full data, mini-batch Cox) ===")

# 1. Data preprocessing (same as run_survival_experiments.py)
df = pd.read_csv('breast_cancer_4quantum.csv')
durations_all = pd.to_numeric(df['Survival_months'], errors='coerce')
events_all = (df['Vital_status_recode_study_cutoff_used_'].astype(str).str.lower() == 'dead').astype(float)
mask = ~durations_all.isna() & (durations_all > 0)
df = df[mask].copy()
df['_duration'] = durations_all[mask].values
df['_event'] = events_all[mask].values

initial_drops = ['Patient_ID', 'age_group', 'Sex_no_total_', 'RX_Summ_Systemic_Sur_Seq_2007_',
    'Diagnostic_Confirmation', 'Histologic_Type_ICD_O_3', 'Primary_Site_labeled',
    'Combined_Summary_Stage_with_Expanded_Regional_Codes_2004_',
    'Reason_no_cancer_directed_surgery', 'Grade_Clinical_2018_',
    'COD_to_site_recode', 'Vital_status_recode_study_cutoff_used_',
    'Grade_Pathological_2018_', 'Race_and_origin_recode_NHW,_NHB,_NHAIAN,_NHAPI,_Hispanic_no_total',
    'Median_household_income_inflation_adj_to_2023', 'Age_recode_with_<1_year_olds_and_90_',
    'Rural_Urban_Continuum_Code', 'stage_encoded',
    'Survival_months', 'survival_60_months', 'Year_of_follow_up_recode']
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
exclude = quantum_features + ['_duration', '_event']
classical_features = [c for c in df.columns if c not in exclude
                      and df[c].dtype in ['float64', 'float32', 'int64', 'int32']]
df_clean = df.dropna(subset=quantum_features + ['_duration', '_event'])
for col in classical_features:
    if df_clean[col].isna().any():
        m = df_clean[col].median()
        if pd.isna(m): m = 0.0
        df_clean[col] = df_clean[col].fillna(m)

train_idx, test_idx = train_test_split(
    np.arange(len(df_clean)), test_size=0.2, random_state=42,
    stratify=df_clean['_event'])
df_train = df_clean.iloc[train_idx].copy()
df_test = df_clean.iloc[test_idx].copy()

q_scaler = MinMaxScaler()
c_scaler = StandardScaler()
Xq_train_scaled = q_scaler.fit_transform(df_train[quantum_features])
Xc_train_scaled = c_scaler.fit_transform(df_train[classical_features])
Xq_test_scaled = q_scaler.transform(df_test[quantum_features])
Xc_test_scaled = c_scaler.transform(df_test[classical_features])

X_train_q = torch.tensor(np.pi * Xq_train_scaled, dtype=torch.float32)
X_test_q = torch.tensor(np.pi * Xq_test_scaled, dtype=torch.float32)
X_train_c = torch.tensor(Xc_train_scaled, dtype=torch.float32)
X_test_c = torch.tensor(Xc_test_scaled, dtype=torch.float32)
durations_train = torch.tensor(df_train['_duration'].values, dtype=torch.float32)
events_train = torch.tensor(df_train['_event'].values, dtype=torch.float32)
durations_test = torch.tensor(df_test['_duration'].values, dtype=torch.float32)
events_test = torch.tensor(df_test['_event'].values, dtype=torch.float32)
n_classical = X_train_c.shape[1]


# 2. Quantum circuits
n_qubits = 7
try:
    dev = qml.device("lightning.qubit", wires=n_qubits)
except Exception:
    dev = qml.device("default.qubit", wires=n_qubits)


@qml.qnode(dev, interface="torch")
def quantum_circuit_v1(inputs, weights):
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)
    for i in range(n_qubits):
        qml.RY(weights[i], wires=i)
    for i in range(n_qubits):
        qml.CNOT(wires=[i, (i + 1) % n_qubits])
    return qml.expval(qml.PauliZ(0))


@qml.qnode(dev, interface="torch")
def quantum_circuit_v2(inputs, weights_ry, weights_rz):
    n_layers = 3
    for layer in range(n_layers):
        for i in range(n_qubits):
            qml.RY(inputs[i], wires=i)
        for i in range(n_qubits):
            qml.RY(weights_ry[layer][i], wires=i)
            qml.RZ(weights_rz[layer][i], wires=i)
        shift = layer + 1
        for i in range(n_qubits):
            qml.CNOT(wires=[i, (i + shift) % n_qubits])
    return tuple(qml.expval(qml.PauliZ(i)) for i in range(n_qubits))


class HybridSurvivalQ_v1(nn.Module):
    def __init__(self, n_classical_features):
        super().__init__()
        self.q_params = nn.Parameter(torch.randn(n_qubits))
        self.fc1 = nn.Linear(1 + n_classical_features, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x_q, x_c):
        x_q = x_q.to(torch.float32)
        x_c = x_c.to(torch.float32)
        q_out = []
        for i in range(x_q.shape[0]):
            res = quantum_circuit_v1(x_q[i], self.q_params)
            q_out.append(res)
        q_out = torch.stack(q_out).unsqueeze(1).to(torch.float32)
        combined = torch.cat([q_out, x_c], dim=1)
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        return self.fc3(x).squeeze(-1)


class HybridSurvivalQ_v2(nn.Module):
    def __init__(self, n_classical_features):
        super().__init__()
        self.q_params_ry = nn.Parameter(torch.randn(3, n_qubits) * 0.1)
        self.q_params_rz = nn.Parameter(torch.randn(3, n_qubits) * 0.1)
        self.classical_encoder = nn.Sequential(
            nn.Linear(n_classical_features, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU())
        self.fusion = nn.Sequential(
            nn.Linear(7 + 32, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1))

    def forward(self, x_q, x_c):
        x_q = x_q.to(torch.float32)
        x_c = x_c.to(torch.float32)
        q_out = []
        for i in range(x_q.shape[0]):
            res = quantum_circuit_v2(x_q[i], self.q_params_ry, self.q_params_rz)
            q_out.append(torch.stack(list(res)))
        q_out = torch.stack(q_out).to(torch.float32)
        c_out = self.classical_encoder(x_c)
        return self.fusion(torch.cat([q_out, c_out], dim=1)).squeeze(-1)


def cox_ph_loss(log_hazards, durations, events):
    idx = torch.argsort(durations, descending=True)
    h = log_hazards[idx]
    e = events[idx]
    log_cumsum = torch.logcumsumexp(h, dim=0)
    n_events = e.sum()
    if n_events == 0:
        return torch.tensor(0.0, requires_grad=True)
    return -((h - log_cumsum) * e).sum() / n_events


def train_minibatch(model, X_q, X_c, dur, evt, X_q_v, X_c_v, dur_v, evt_v,
                    epochs=20, lr=0.001, batch_size=256, time_budget=900):
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    n = len(X_q)
    print(f"  Mini-batch: N={n}, batch={batch_size}")
    t0 = time.time()
    for epoch in range(epochs):
        if (time.time() - t0) > time_budget:
            print(f"  Time budget at epoch {epoch}"); break
        model.train()
        perm = torch.randperm(n)
        losses = []
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            if evt[idx].sum() == 0:
                continue
            opt.zero_grad()
            log_h = model(X_q[idx], X_c[idx])
            loss = cox_ph_loss(log_h, dur[idx], evt[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            losses.append(loss.item())
        sched.step()
        if (epoch + 1) % 2 == 0:
            model.eval()
            with torch.no_grad():
                v_h = model(X_q_v, X_c_v).cpu().numpy()
                v_cidx = concordance_index(dur_v.cpu().numpy(), -v_h, evt_v.cpu().numpy())
            print(f"  Epoch {epoch+1:02d} | Loss: {np.mean(losses):.4f} | Val C-idx: {v_cidx:.4f}")


# 3. Run
t_start = time.time()
if MODEL_NAME == 'v1':
    model = HybridSurvivalQ_v1(n_classical)
    train_minibatch(model, X_train_q, X_train_c, durations_train, events_train,
                    X_test_q, X_test_c, durations_test, events_test,
                    epochs=20, lr=0.001, batch_size=256, time_budget=900)
elif MODEL_NAME == 'v2':
    model = HybridSurvivalQ_v2(n_classical)
    train_minibatch(model, X_train_q, X_train_c, durations_train, events_train,
                    X_test_q, X_test_c, durations_test, events_test,
                    epochs=20, lr=0.001, batch_size=256, time_budget=1200)

# Evaluate
model.eval()
with torch.no_grad():
    log_h = model(X_test_q, X_test_c).cpu().numpy()
final_cidx = concordance_index(durations_test.cpu().numpy(), -log_h, events_test.cpu().numpy())
elapsed = time.time() - t_start

print(f"\n=== {MODEL_NAME}-FULL FINAL ===")
print(f"Test C-index: {final_cidx:.4f}")
print(f"Time: {elapsed:.1f}s")

result = {
    'model': f'{MODEL_NAME}_full',
    'test_c_index': float(final_cidx),
    'time_sec': float(elapsed),
    'train_size': len(X_train_q),
}
with open(f'survival_{MODEL_NAME}_full_result.json', 'w') as f:
    json.dump(result, f, indent=2)
print(f"Saved: survival_{MODEL_NAME}_full_result.json")
