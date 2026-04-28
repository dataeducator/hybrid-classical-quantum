"""Run ONE binary classification model on clean data. For parallel execution.

Usage:
    python run_binary_one.py --model v2
    python run_binary_one.py --model v3
    python run_binary_one.py --model v3-full
    python run_binary_one.py --model v4
"""
import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True)

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--model', required=True, choices=['v2', 'v3', 'v3-full', 'v4'])
args = parser.parse_args()
MODEL_NAME = args.model

import time
import gc
import json
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
from sklearn.metrics import (roc_auc_score, roc_curve, precision_score,
                             recall_score, f1_score)


def set_determinism(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


set_determinism(42)
print(f"=== Running model: {MODEL_NAME} ===")

# 1. Load and preprocess data (same as run_binary_honest.py)
df = pd.read_csv('breast_cancer_4quantum.csv')
vital = df['Vital_status_recode_study_cutoff_used_'].astype(str).str.lower()
sm = pd.to_numeric(df['Survival_months'], errors='coerce')
target = pd.Series(np.nan, index=df.index, dtype=float)
target[sm >= 60] = 1.0
target[(sm < 60) & (vital == 'dead')] = 0.0
df['survival_60_months'] = target
df = df.dropna(subset=['survival_60_months'])
df['survival_60_months'] = df['survival_60_months'].astype(int)

initial_drops = [
    'Patient_ID', 'age_group', 'Sex_no_total_', 'RX_Summ_Systemic_Sur_Seq_2007_',
    'Diagnostic_Confirmation', 'Histologic_Type_ICD_O_3', 'Primary_Site_labeled',
    'Combined_Summary_Stage_with_Expanded_Regional_Codes_2004_',
    'Reason_no_cancer_directed_surgery', 'Grade_Clinical_2018_',
    'COD_to_site_recode', 'Vital_status_recode_study_cutoff_used_',
    'Grade_Pathological_2018_', 'Race_and_origin_recode_NHW,_NHB,_NHAIAN,_NHAPI,_Hispanic_no_total',
    'Median_household_income_inflation_adj_to_2023', 'Age_recode_with_<1_year_olds_and_90_',
    'Rural_Urban_Continuum_Code', 'stage_encoded', 'Year_of_follow_up_recode',
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
n_classical = X_train_c.shape[1]

pos_count = y_train.sum()
neg_count = len(y_train) - pos_count
pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32)

# Subsample
MAX_TRAIN, MAX_TEST = 2000, 500
np.random.seed(42)
idx_tr = np.random.choice(len(X_train_q), MAX_TRAIN, replace=False)
idx_te = np.random.choice(len(X_test_q), MAX_TEST, replace=False)
X_train_q_sub = X_train_q[idx_tr]
X_train_c_sub = X_train_c[idx_tr]
y_train_sub = y_train_t[idx_tr]
X_test_q_sub = X_test_q[idx_te]
X_test_c_sub = X_test_c[idx_te]
y_test_sub = y_test_t[idx_te]
pos_sub = y_train_sub.sum().item()
neg_sub = len(y_train_sub) - pos_sub
pos_weight_sub = torch.tensor([neg_sub / pos_sub], dtype=torch.float32)

# 2. Quantum circuit + models
n_qubits = 7
try:
    dev = qml.device("lightning.qubit", wires=n_qubits)
except Exception:
    dev = qml.device("default.qubit", wires=n_qubits)


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


@qml.qnode(dev, interface="torch")
def quantum_circuit_v3(inputs, input_scales, input_biases, weights_ry, weights_rz):
    n_layers = 3
    for layer in range(n_layers):
        for i in range(n_qubits):
            qml.RY(input_scales[i] * inputs[i] + input_biases[i], wires=i)
        for i in range(n_qubits):
            qml.RY(weights_ry[layer][i], wires=i)
            qml.RZ(weights_rz[layer][i], wires=i)
        shift = layer + 1
        for i in range(n_qubits):
            qml.CNOT(wires=[i, (i + shift) % n_qubits])
    single = [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
    corr = [qml.expval(qml.PauliZ(i) @ qml.PauliZ((i + 1) % n_qubits))
            for i in range(n_qubits)]
    return tuple(single + corr)


class HybridRealQ_v2(nn.Module):
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
        return self.fusion(torch.cat([q_out, c_out], dim=1))


class HybridRealQ_v3(nn.Module):
    def __init__(self, n_classical_features):
        super().__init__()
        self.input_scales = nn.Parameter(torch.ones(n_qubits) * np.pi)
        self.input_biases = nn.Parameter(torch.zeros(n_qubits))
        self.q_params_ry = nn.Parameter(torch.randn(3, n_qubits) * 0.1)
        self.q_params_rz = nn.Parameter(torch.randn(3, n_qubits) * 0.1)
        self.classical_encoder = nn.Sequential(
            nn.Linear(n_classical_features, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU())
        fusion_in = 14 + 32 + n_classical_features
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1))

    def forward(self, x_q, x_c):
        x_q = x_q.to(torch.float32)
        x_c = x_c.to(torch.float32)
        q_out = []
        for i in range(x_q.shape[0]):
            res = quantum_circuit_v3(x_q[i], self.input_scales, self.input_biases,
                                     self.q_params_ry, self.q_params_rz)
            q_out.append(torch.stack(list(res)))
        q_out = torch.stack(q_out).to(torch.float32)
        c_encoded = self.classical_encoder(x_c)
        return self.fusion(torch.cat([q_out, c_encoded, x_c], dim=1))


class HybridRealQ_v4(HybridRealQ_v3):
    def __init__(self, n_classical_features):
        super().__init__(n_classical_features)
        # Override with smaller init
        self.q_params_ry = nn.Parameter(torch.randn(3, n_qubits) * 0.05)
        self.q_params_rz = nn.Parameter(torch.randn(3, n_qubits) * 0.05)
        self.output_scale = nn.Parameter(torch.tensor(3.0))
        self.output_bias = nn.Parameter(torch.tensor(0.0))

    def forward(self, x_q, x_c):
        out = super().forward(x_q, x_c).squeeze(-1)
        return (self.output_scale * out + self.output_bias).unsqueeze(-1)


def train_full_batch(model, X_q_tr, X_c_tr, y_tr, X_q_te, X_c_te, y_te,
                     epochs, lr, pos_w, time_budget):
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    t0 = time.time()
    for epoch in range(epochs):
        if (time.time() - t0) > time_budget:
            print(f"  Time budget reached at epoch {epoch}"); break
        model.train()
        opt.zero_grad()
        logits = model(X_q_tr, X_c_tr).squeeze()
        loss = crit(logits, y_tr.squeeze())
        loss.backward()
        opt.step()
        sched.step()
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                vl = crit(model(X_q_te, X_c_te).squeeze(), y_te.squeeze()).item()
            print(f"  Epoch {epoch+1:02d} | Loss: {loss.item():.4f} | Val: {vl:.4f}")


def train_minibatch(model, X_q, X_c, y, X_q_v, X_c_v, y_v,
                    epochs, lr, batch_size, pos_w, time_budget):
    crit = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=1e-5)
    n = len(X_q)
    print(f"  Mini-batch: N={n}, batch={batch_size}")
    t0 = time.time()
    for epoch in range(epochs):
        if (time.time() - t0) > time_budget:
            print(f"  Time budget reached at epoch {epoch}"); break
        model.train()
        perm = torch.randperm(n)
        losses = []
        for s in range(0, n, batch_size):
            idx = perm[s:s + batch_size]
            opt.zero_grad()
            logits = model(X_q[idx], X_c[idx]).squeeze()
            loss = crit(logits, y[idx].squeeze())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            losses.append(loss.item())
        sched.step()
        if (epoch + 1) % 2 == 0:
            model.eval()
            with torch.no_grad():
                vl = crit(model(X_q_v, X_c_v).squeeze(), y_v.squeeze()).item()
            print(f"  Epoch {epoch+1:02d} | Loss: {np.mean(losses):.4f} | Val: {vl:.4f}")


def evaluate(model, X_q, X_c, y_np):
    model.eval()
    y_np = y_np.reshape(-1)
    with torch.no_grad():
        probs = torch.sigmoid(model(X_q, X_c).squeeze()).cpu().numpy()
    auc = roc_auc_score(y_np, probs)
    fpr, tpr, thr = roc_curve(y_np, probs)
    t_opt = thr[(tpr - fpr).argmax()]
    preds = (probs >= t_opt).astype(int)
    return {
        'auc': float(auc),
        'precision': float(precision_score(y_np, preds, zero_division=0)),
        'recall': float(recall_score(y_np, preds, zero_division=0)),
        'f1': float(f1_score(y_np, preds, zero_division=0)),
        'threshold': float(t_opt),
        'probs': probs.tolist(),
    }


# 3. Run requested model
t_start = time.time()
if MODEL_NAME == 'v2':
    model = HybridRealQ_v2(n_classical)
    train_full_batch(model, X_train_q_sub, X_train_c_sub, y_train_sub,
                     X_test_q_sub, X_test_c_sub, y_test_sub,
                     epochs=75, lr=0.001, pos_w=pos_weight_sub, time_budget=570)
    metrics = evaluate(model, X_test_q_sub, X_test_c_sub, y_test_sub.numpy())
    train_size = MAX_TRAIN
    full_metrics = None
elif MODEL_NAME == 'v3':
    model = HybridRealQ_v3(n_classical)
    train_full_batch(model, X_train_q_sub, X_train_c_sub, y_train_sub,
                     X_test_q_sub, X_test_c_sub, y_test_sub,
                     epochs=75, lr=0.001, pos_w=pos_weight_sub, time_budget=600)
    metrics = evaluate(model, X_test_q_sub, X_test_c_sub, y_test_sub.numpy())
    train_size = MAX_TRAIN
    full_metrics = None
elif MODEL_NAME == 'v3-full':
    model = HybridRealQ_v3(n_classical)
    train_minibatch(model, X_train_q, X_train_c, y_train_t,
                    X_test_q, X_test_c, y_test_t,
                    epochs=20, lr=0.001, batch_size=256,
                    pos_w=pos_weight, time_budget=900)
    metrics = evaluate(model, X_test_q, X_test_c, y_test_t.numpy())
    train_size = len(X_train_q)
    full_metrics = metrics  # Save full-data probs for ensemble
elif MODEL_NAME == 'v4':
    model = HybridRealQ_v4(n_classical)
    train_minibatch(model, X_train_q, X_train_c, y_train_t,
                    X_test_q, X_test_c, y_test_t,
                    epochs=20, lr=0.001, batch_size=256,
                    pos_w=pos_weight, time_budget=1200)
    metrics = evaluate(model, X_test_q, X_test_c, y_test_t.numpy())
    train_size = len(X_train_q)
    full_metrics = metrics

elapsed = time.time() - t_start
print(f"\n=== {MODEL_NAME} FINAL ===")
print(f"AUC: {metrics['auc']:.4f}")
print(f"Precision: {metrics['precision']:.4f}")
print(f"Recall: {metrics['recall']:.4f}")
print(f"F1: {metrics['f1']:.4f}")
print(f"Time: {elapsed:.1f}s")

# 4. Save
result = {
    'model': MODEL_NAME,
    'auc': metrics['auc'],
    'precision': metrics['precision'],
    'recall': metrics['recall'],
    'f1': metrics['f1'],
    'threshold': metrics['threshold'],
    'time_sec': elapsed,
    'train_size': train_size,
    'probs': metrics['probs'] if MODEL_NAME in ('v3-full', 'v4') else None,
}
with open(f'binary_{MODEL_NAME.replace("-", "_")}_result.json', 'w') as f:
    json.dump(result, f, indent=2)
print(f"Saved: binary_{MODEL_NAME.replace('-', '_')}_result.json")
