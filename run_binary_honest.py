"""
Honest binary classification baselines on the leakage-free, censoring-corrected
TNBC cohort.

Design choices that distinguish this from the original run_experiments.py:

1. XGBoost and LightGBM train on the FULL training set (16,088 patients), not
   a 2,000-patient subsample. Tree models scale to full data instantly; the
   subsample-only setup of the older run handicapped the honest classical
   baseline.

2. Neural quantum-classical models (v1, v2, v3) keep the 2,000-patient
   subsample because per-sample quantum simulation is the real bottleneck.
   That asymmetry is reported transparently: tree models test the
   "what's the data ceiling" question; quantum models test the "what's
   feasible to train" question.

3. Class imbalance after censoring fix is severe (~75% positive). All neural
   binary models use BCEWithLogitsLoss with pos_weight matching the inverse
   class frequency in the training set. Without this, the small-subsample
   models collapse to predicting the majority class and AUC is ~0.5.

4. Single train/test split with seed 42, stratified by the binary outcome.
"""
import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True)

import time
import platform
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
from sklearn.metrics import (roc_auc_score, confusion_matrix, roc_curve,
                             precision_score, recall_score, f1_score, accuracy_score)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


def set_determinism(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


set_determinism(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")

RACE_LABELS = {
    'race_0': 'Non-Hispanic White',
    'race_1': 'Non-Hispanic Black',
    'race_2': 'Hispanic',
    'race_3': 'Non-Hispanic Asian/Pacific Islander',
    'race_4': 'Non-Hispanic American Indian/Alaska Native',
    'race_5': 'Non-Hispanic Unknown Race',
}

# 1. Data preprocessing (matches run_survival_experiments.py for direct comparability)
print("\n" + "=" * 60)
print("  DATA LOADING & PREPROCESSING (HONEST BINARY)")
print("=" * 60)

df = pd.read_csv('breast_cancer_4quantum.csv')
print(f"Loaded: {df.shape}")

# Re-derive binary survival_60_months excluding censored patients
vital = df['Vital_status_recode_study_cutoff_used_'].astype(str).str.lower()
sm = pd.to_numeric(df['Survival_months'], errors='coerce')
target = pd.Series(np.nan, index=df.index, dtype=float)
target[sm >= 60] = 1.0
target[(sm < 60) & (vital == 'dead')] = 0.0
df['survival_60_months'] = target
n_before = len(df)
df = df.dropna(subset=['survival_60_months'])
df['survival_60_months'] = df['survival_60_months'].astype(int)
print(f"  Re-labeled survival_60_months: dropped {n_before - len(df)} censored patients")
print(f"  Clean cohort: {len(df)} patients ({df['survival_60_months'].mean():.1%} positive)")

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
target_col = 'survival_60_months'
exclude = quantum_features + [target_col, 'Survival_months', 'Vital_status_recode_study_cutoff_used_']
classical_features = [c for c in df.columns
                      if c not in exclude and df[c].dtype in ['float64', 'float32', 'int64', 'int32']]

df_clean = df.dropna(subset=quantum_features + [target_col])
for col in classical_features:
    if df_clean[col].isna().any():
        m = df_clean[col].median()
        if pd.isna(m):
            m = 0.0
        df_clean[col] = df_clean[col].fillna(m)

print(f"Clean data: {df_clean.shape}, n_classical={len(classical_features)}")
print(f"Target distribution: {dict(df_clean[target_col].value_counts())}")

# 2. Train/test split (stratified)
Xq_train, Xq_test, Xc_train, Xc_test, y_train, y_test = train_test_split(
    df_clean[quantum_features], df_clean[classical_features], df_clean[target_col],
    test_size=0.2, random_state=42, stratify=df_clean[target_col]
)
print(f"Train: {len(Xq_train)} ({y_train.mean():.1%} positive)")
print(f"Test:  {len(Xq_test)}  ({y_test.mean():.1%} positive)")

# 3. Scaling
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

# Class imbalance handling: pos_weight = N_neg / N_pos
pos_count = y_train.sum()
neg_count = len(y_train) - pos_count
pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32)
print(f"\nClass weighting: pos_weight = {pos_weight.item():.4f} "
      f"(neg/pos = {int(neg_count)}/{int(pos_count)})")

# 4. Leakage audit
print("\n" + "=" * 60)
print("  DATA LEAKAGE AUDIT (binary outcome)")
print("=" * 60)

from scipy import stats
all_feature_names = quantum_features + classical_features
X_train_all = pd.concat([Xq_train, Xc_train], axis=1).values
X_test_all = pd.concat([Xq_test, Xc_test], axis=1).values

flagged_auc = []
for i, feat in enumerate(all_feature_names):
    try:
        a = roc_auc_score(y_train, X_train_all[:, i])
        if a < 0.5:
            a = 1 - a
        if a > 0.95:
            flagged_auc.append((feat, a))
    except Exception:
        pass
print(f"Per-feature AUC scan (>0.95): {len(flagged_auc)} flag(s)")
for f, a in flagged_auc:
    print(f"  WARNING: {f}: {a:.4f}")
if not flagged_auc:
    print("  PASS")

flagged_corr = []
for i, feat in enumerate(all_feature_names):
    try:
        c = abs(np.corrcoef(X_train_all[:, i], y_train.values)[0, 1])
        if c > 0.90:
            flagged_corr.append((feat, c))
    except Exception:
        pass
print(f"Correlation scan (>0.90): {len(flagged_corr)} flag(s)")
if not flagged_corr:
    print("  PASS")

flagged_ks = []
for i, feat in enumerate(all_feature_names):
    try:
        _, p = stats.ks_2samp(X_train_all[:, i], X_test_all[:, i])
        if p < 0.001:
            flagged_ks.append((feat, p))
    except Exception:
        pass
print(f"KS distribution shift (p<0.001): {len(flagged_ks)} flag(s)")
if not flagged_ks:
    print("  PASS")

n_flags = len(flagged_auc) + len(flagged_corr) + len(flagged_ks)
print(f"  RESULT: {'ALL CHECKS PASSED' if n_flags == 0 else f'{n_flags} flags'}")

# 5. Hardware logging
def log_hardware_info():
    info = {"platform": platform.platform(), "torch_version": torch.__version__}
    if torch.cuda.is_available():
        info["device"] = "GPU"
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_gb"] = round(torch.cuda.get_device_properties(0).total_memory / 1e9, 2)
        info["cuda_version"] = torch.version.cuda
    else:
        info["device"] = "CPU"
    info["pennylane_version"] = qml.__version__
    return info


# 6. Quantum circuit and models
n_qubits = 7
try:
    dev = qml.device("lightning.qubit", wires=n_qubits)
    print("\nUsing lightning.qubit (fast CPU simulator)")
except Exception:
    dev = qml.device("default.qubit", wires=n_qubits)


@qml.qnode(dev, interface="torch")
def quantum_circuit(inputs, weights):
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)
    for i in range(n_qubits):
        qml.RY(weights[i], wires=i)
    for i in range(n_qubits):
        qml.CNOT(wires=[i, (i + 1) % n_qubits])
    return qml.expval(qml.PauliZ(0))


class HybridRealQ(nn.Module):
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
            res = quantum_circuit(x_q[i], self.q_params)
            q_out.append(res)
        q_out = torch.stack(q_out).unsqueeze(1).to(torch.float32)
        combined = torch.cat([q_out, x_c], dim=1)
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


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
            nn.Linear(64, 32), nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(7 + 32, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

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
    """v3: trainable input scaling, ZZ correlators, residual classical features."""
    def __init__(self, n_classical_features):
        super().__init__()
        self.input_scales = nn.Parameter(torch.ones(n_qubits) * np.pi)
        self.input_biases = nn.Parameter(torch.zeros(n_qubits))
        self.q_params_ry = nn.Parameter(torch.randn(3, n_qubits) * 0.1)
        self.q_params_rz = nn.Parameter(torch.randn(3, n_qubits) * 0.1)
        self.classical_encoder = nn.Sequential(
            nn.Linear(n_classical_features, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
        )
        fusion_in = 14 + 32 + n_classical_features
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

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


class HybridRealQ_v4(nn.Module):
    """v4: v3 + output scaling + smaller init."""
    def __init__(self, n_classical_features):
        super().__init__()
        self.input_scales = nn.Parameter(torch.ones(n_qubits) * np.pi)
        self.input_biases = nn.Parameter(torch.zeros(n_qubits))
        self.q_params_ry = nn.Parameter(torch.randn(3, n_qubits) * 0.05)
        self.q_params_rz = nn.Parameter(torch.randn(3, n_qubits) * 0.05)
        self.classical_encoder = nn.Sequential(
            nn.Linear(n_classical_features, 64), nn.ReLU(),
            nn.Linear(64, 32), nn.ReLU(),
        )
        fusion_in = 14 + 32 + n_classical_features
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.output_scale = nn.Parameter(torch.tensor(3.0))
        self.output_bias = nn.Parameter(torch.tensor(0.0))

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
        fused = self.fusion(torch.cat([q_out, c_encoded, x_c], dim=1)).squeeze(-1)
        return (self.output_scale * fused + self.output_bias).unsqueeze(-1)


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


# 7. Training and evaluation utilities
def train_model(model, X_q_tr, X_c_tr, y_tr, X_q_te, X_c_te, y_te,
                epochs=50, lr=0.001, pos_weight_tensor=None, verbose=True,
                use_scheduler=False, time_budget=None):
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-5)
    history = {'train_loss': [], 'val_loss': []}
    total_start = time.time()
    for epoch in range(epochs):
        if time_budget and (time.time() - total_start) > time_budget:
            if verbose:
                print(f"  Time budget reached at epoch {epoch}")
            break
        model.train()
        optimizer.zero_grad()
        logits = model(X_q_tr, X_c_tr).squeeze()
        loss = criterion(logits, y_tr.squeeze())
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
        model.eval()
        with torch.no_grad():
            val_logits = model(X_q_te, X_c_te).squeeze()
            val_loss = criterion(val_logits, y_te.squeeze())
        history['train_loss'].append(loss.item())
        history['val_loss'].append(val_loss.item())
        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:02d} | Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f}")
    history['total_time'] = time.time() - total_start
    history['hardware'] = log_hardware_info()
    return history


def train_model_minibatch(model, X_q, X_c, y, X_q_val, X_c_val, y_val,
                          epochs=20, lr=0.001, batch_size=256,
                          pos_weight_tensor=None, time_budget=900, verbose=True):
    """Mini-batch BCE training for full-data quantum models."""
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    n = len(X_q)
    history = {'train_loss': [], 'val_loss': []}
    total_start = time.time()
    if verbose:
        print(f"  Mini-batch: N={n}, batch_size={batch_size}, batches/epoch={n // batch_size + 1}")
    for epoch in range(epochs):
        if (time.time() - total_start) > time_budget:
            if verbose:
                print(f"  Time budget reached at epoch {epoch}")
            break
        model.train()
        perm = torch.randperm(n)
        losses = []
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            optimizer.zero_grad()
            logits = model(X_q[idx], X_c[idx]).squeeze()
            loss = criterion(logits, y[idx].squeeze())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()
        avg_loss = float(np.mean(losses)) if losses else 0.0
        model.eval()
        with torch.no_grad():
            val_logits = model(X_q_val, X_c_val).squeeze()
            val_loss = criterion(val_logits, y_val.squeeze()).item()
        history['train_loss'].append(avg_loss)
        history['val_loss'].append(val_loss)
        if verbose and (epoch + 1) % 2 == 0:
            print(f"  Epoch {epoch+1:02d} | Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f}")
    history['total_time'] = time.time() - total_start
    return history


def evaluate_model(model, X_q, X_c, y_np):
    model.eval()
    y_np = y_np.reshape(-1)
    with torch.no_grad():
        logits = model(X_q, X_c).squeeze()
        probs = torch.sigmoid(logits).cpu().numpy()
    auc = roc_auc_score(y_np, probs)
    fpr, tpr, thresholds = roc_curve(y_np, probs)
    j = tpr - fpr
    best_t = thresholds[j.argmax()]
    preds = (probs >= best_t).astype(int)
    return {
        'auc': auc,
        'precision': precision_score(y_np, preds, zero_division=0),
        'recall': recall_score(y_np, preds, zero_division=0),
        'f1': f1_score(y_np, preds, zero_division=0),
        'threshold': float(best_t),
        'probs': probs,
        'preds': preds,
    }


# 8. Subsample setup for quantum models
MAX_TRAIN = 2000
MAX_TEST = 500
np.random.seed(42)
if len(X_train_q) > MAX_TRAIN:
    print(f"\nQuantum subsample: {MAX_TRAIN} train, {MAX_TEST} test "
          f"(from {len(X_train_q)}/{len(X_test_q)})")
    idx_tr = np.random.choice(len(X_train_q), MAX_TRAIN, replace=False)
    idx_te = np.random.choice(len(X_test_q), MAX_TEST, replace=False)
    X_train_q_sub = X_train_q[idx_tr]
    X_train_c_sub = X_train_c[idx_tr]
    y_train_sub = y_train_t[idx_tr]
    X_test_q_sub = X_test_q[idx_te]
    X_test_c_sub = X_test_c[idx_te]
    y_test_sub = y_test_t[idx_te]
    # pos_weight for the subsample (same imbalance ratio in expectation, but recompute exactly)
    pos_sub = y_train_sub.sum().item()
    neg_sub = len(y_train_sub) - pos_sub
    pos_weight_sub = torch.tensor([neg_sub / pos_sub], dtype=torch.float32)
    print(f"  Subsample pos_weight: {pos_weight_sub.item():.4f} "
          f"(neg/pos = {int(neg_sub)}/{int(pos_sub)})")
else:
    X_train_q_sub, X_train_c_sub, y_train_sub = X_train_q, X_train_c, y_train_t
    X_test_q_sub, X_test_c_sub, y_test_sub = X_test_q, X_test_c, y_test_t
    pos_weight_sub = pos_weight


# 9. Run all baselines
print("\n" + "=" * 80)
print("  HONEST BINARY ABLATION")
print("=" * 80)

ablation = []
xgb_trained = None
lgbm_trained = None
mlp_full_probs = None  # for fairness audit on full test


# 9a. Classical MLP on full data with class weighting
print("\n--- Classical MLP (FULL data, class-weighted) ---")
set_determinism(42)
mlp = ClassicalMLP(n_classical)
t0 = time.time()
mlp_history = train_model(mlp, X_train_q, X_train_c, y_train_t,
                          X_test_q, X_test_c, y_test_t,
                          epochs=100, lr=0.001, pos_weight_tensor=pos_weight, verbose=True)
mlp_metrics = evaluate_model(mlp, X_test_q, X_test_c, y_test_t.numpy())
mlp_time = time.time() - t0
print(f"  Classical MLP (full): AUC={mlp_metrics['auc']:.4f} "
      f"Prec={mlp_metrics['precision']:.4f} Rec={mlp_metrics['recall']:.4f} "
      f"F1={mlp_metrics['f1']:.4f} Time={mlp_time:.1f}s")
mlp.eval()
with torch.no_grad():
    mlp_full_probs = torch.sigmoid(mlp(X_test_q, X_test_c).squeeze()).cpu().numpy()
ablation.append({
    'Model': 'Classical MLP (full data, class-weighted)',
    'AUC': round(mlp_metrics['auc'], 4),
    'Precision': round(mlp_metrics['precision'], 4),
    'Recall': round(mlp_metrics['recall'], 4),
    'F1': round(mlp_metrics['f1'], 4),
    'Time_sec': round(mlp_time, 1),
    'Train_size': len(X_train_q),
    'Status': 'OK',
})


# 9b. HybridRealQ on subsample with class weighting
print("\n--- HybridRealQ v1 (subsample, class-weighted) ---")
set_determinism(42)
hyb = HybridRealQ(n_classical)
t0 = time.time()
hyb_history = train_model(hyb, X_train_q_sub, X_train_c_sub, y_train_sub,
                          X_test_q_sub, X_test_c_sub, y_test_sub,
                          epochs=50, lr=0.001, pos_weight_tensor=pos_weight_sub, verbose=True)
hyb_metrics = evaluate_model(hyb, X_test_q_sub, X_test_c_sub, y_test_sub.numpy())
hyb_time = time.time() - t0
print(f"  HybridRealQ (subsample): AUC={hyb_metrics['auc']:.4f} "
      f"Prec={hyb_metrics['precision']:.4f} Rec={hyb_metrics['recall']:.4f} "
      f"F1={hyb_metrics['f1']:.4f} Time={hyb_time:.1f}s")
ablation.append({
    'Model': 'HybridRealQ v1 (subsample, class-weighted)',
    'AUC': round(hyb_metrics['auc'], 4),
    'Precision': round(hyb_metrics['precision'], 4),
    'Recall': round(hyb_metrics['recall'], 4),
    'F1': round(hyb_metrics['f1'], 4),
    'Time_sec': round(hyb_time, 1),
    'Train_size': len(X_train_q_sub),
    'Status': 'OK',
})


# 9b2. HybridRealQ_v2 on subsample with class weighting
print("\n--- HybridRealQ_v2 (subsample, class-weighted) ---")
set_determinism(42)
gc.collect()
v2 = HybridRealQ_v2(n_classical)
t0 = time.time()
v2_history = train_model(v2, X_train_q_sub, X_train_c_sub, y_train_sub,
                         X_test_q_sub, X_test_c_sub, y_test_sub,
                         epochs=75, lr=0.001, pos_weight_tensor=pos_weight_sub,
                         use_scheduler=True, time_budget=570, verbose=True)
v2_metrics = evaluate_model(v2, X_test_q_sub, X_test_c_sub, y_test_sub.numpy())
v2_time = time.time() - t0
print(f"  HybridRealQ_v2: AUC={v2_metrics['auc']:.4f} "
      f"Prec={v2_metrics['precision']:.4f} Rec={v2_metrics['recall']:.4f} "
      f"F1={v2_metrics['f1']:.4f} Time={v2_time:.1f}s")
ablation.append({
    'Model': 'HybridRealQ_v2 (subsample, class-weighted)',
    'AUC': round(v2_metrics['auc'], 4),
    'Precision': round(v2_metrics['precision'], 4),
    'Recall': round(v2_metrics['recall'], 4),
    'F1': round(v2_metrics['f1'], 4),
    'Time_sec': round(v2_time, 1),
    'Train_size': len(X_train_q_sub),
    'Status': 'OK',
})

# 9b3. HybridRealQ_v3 on subsample with class weighting
print("\n--- HybridRealQ_v3 (subsample, class-weighted) ---")
set_determinism(42)
gc.collect()
v3 = HybridRealQ_v3(n_classical)
t0 = time.time()
v3_history = train_model(v3, X_train_q_sub, X_train_c_sub, y_train_sub,
                         X_test_q_sub, X_test_c_sub, y_test_sub,
                         epochs=75, lr=0.001, pos_weight_tensor=pos_weight_sub,
                         use_scheduler=True, time_budget=600, verbose=True)
v3_metrics = evaluate_model(v3, X_test_q_sub, X_test_c_sub, y_test_sub.numpy())
v3_time = time.time() - t0
print(f"  HybridRealQ_v3: AUC={v3_metrics['auc']:.4f} "
      f"Prec={v3_metrics['precision']:.4f} Rec={v3_metrics['recall']:.4f} "
      f"F1={v3_metrics['f1']:.4f} Time={v3_time:.1f}s")
ablation.append({
    'Model': 'HybridRealQ_v3 (subsample, class-weighted)',
    'AUC': round(v3_metrics['auc'], 4),
    'Precision': round(v3_metrics['precision'], 4),
    'Recall': round(v3_metrics['recall'], 4),
    'F1': round(v3_metrics['f1'], 4),
    'Time_sec': round(v3_time, 1),
    'Train_size': len(X_train_q_sub),
    'Status': 'OK',
})

# 9b4. HybridRealQ_v3 on FULL data with mini-batch
print("\n--- HybridRealQ_v3 (FULL data, mini-batch, class-weighted) ---")
set_determinism(42)
gc.collect()
v3_full = HybridRealQ_v3(n_classical)
t0 = time.time()
v3_full_history = train_model_minibatch(
    v3_full, X_train_q, X_train_c, y_train_t,
    X_test_q, X_test_c, y_test_t,
    epochs=20, lr=0.001, batch_size=256,
    pos_weight_tensor=pos_weight, time_budget=900, verbose=True)
v3_full_metrics = evaluate_model(v3_full, X_test_q, X_test_c, y_test_t.numpy())
v3_full_time = time.time() - t0
print(f"  HybridRealQ_v3 (full): AUC={v3_full_metrics['auc']:.4f} "
      f"Prec={v3_full_metrics['precision']:.4f} Rec={v3_full_metrics['recall']:.4f} "
      f"F1={v3_full_metrics['f1']:.4f} Time={v3_full_time:.1f}s")
v3_full.eval()
with torch.no_grad():
    v3_full_probs = torch.sigmoid(v3_full(X_test_q, X_test_c).squeeze()).cpu().numpy()
ablation.append({
    'Model': 'HybridRealQ_v3 (full data, class-weighted)',
    'AUC': round(v3_full_metrics['auc'], 4),
    'Precision': round(v3_full_metrics['precision'], 4),
    'Recall': round(v3_full_metrics['recall'], 4),
    'F1': round(v3_full_metrics['f1'], 4),
    'Time_sec': round(v3_full_time, 1),
    'Train_size': len(X_train_q),
    'Status': 'OK',
})

# 9b5. HybridRealQ_v4 on FULL data with mini-batch (output scaling)
print("\n--- HybridRealQ_v4 (FULL data, output scaling, class-weighted) ---")
set_determinism(42)
gc.collect()
v4_model = HybridRealQ_v4(n_classical)
t0 = time.time()
v4_history = train_model_minibatch(
    v4_model, X_train_q, X_train_c, y_train_t,
    X_test_q, X_test_c, y_test_t,
    epochs=20, lr=0.001, batch_size=256,
    pos_weight_tensor=pos_weight, time_budget=1200, verbose=True)
v4_metrics = evaluate_model(v4_model, X_test_q, X_test_c, y_test_t.numpy())
v4_time = time.time() - t0
print(f"  HybridRealQ_v4 (full): AUC={v4_metrics['auc']:.4f} "
      f"Prec={v4_metrics['precision']:.4f} Rec={v4_metrics['recall']:.4f} "
      f"F1={v4_metrics['f1']:.4f} Time={v4_time:.1f}s")
v4_model.eval()
with torch.no_grad():
    v4_probs = torch.sigmoid(v4_model(X_test_q, X_test_c).squeeze()).cpu().numpy()
ablation.append({
    'Model': 'HybridRealQ_v4 (full data, output scaling)',
    'AUC': round(v4_metrics['auc'], 4),
    'Precision': round(v4_metrics['precision'], 4),
    'Recall': round(v4_metrics['recall'], 4),
    'F1': round(v4_metrics['f1'], 4),
    'Time_sec': round(v4_time, 1),
    'Train_size': len(X_train_q),
    'Status': 'OK',
})

# 9c. XGBoost on full data with scale_pos_weight
print("\n--- XGBoost (FULL data, scale_pos_weight) ---")
X_train_combined = np.hstack([X_train_q.numpy() / np.pi, X_train_c.numpy()])
X_test_combined = np.hstack([X_test_q.numpy() / np.pi, X_test_c.numpy()])
y_train_np = y_train.values
y_test_np = y_test.values

t0 = time.time()
xgb_trained = XGBClassifier(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    scale_pos_weight=pos_weight.item(),
    eval_metric='logloss', random_state=42, use_label_encoder=False,
    n_jobs=-1,
)
xgb_trained.fit(X_train_combined, y_train_np)
xgb_time = time.time() - t0
xgb_probs = xgb_trained.predict_proba(X_test_combined)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test_np, xgb_probs)
xgb_thresh = thresholds[(tpr - fpr).argmax()]
xgb_preds = (xgb_probs >= xgb_thresh).astype(int)
xgb_metrics = {
    'auc': roc_auc_score(y_test_np, xgb_probs),
    'precision': precision_score(y_test_np, xgb_preds, zero_division=0),
    'recall': recall_score(y_test_np, xgb_preds, zero_division=0),
    'f1': f1_score(y_test_np, xgb_preds, zero_division=0),
    'threshold': float(xgb_thresh),
    'probs': xgb_probs,
    'preds': xgb_preds,
}
print(f"  XGBoost (full): AUC={xgb_metrics['auc']:.4f} "
      f"Prec={xgb_metrics['precision']:.4f} Rec={xgb_metrics['recall']:.4f} "
      f"F1={xgb_metrics['f1']:.4f} Time={xgb_time:.1f}s")
ablation.append({
    'Model': 'XGBoost (full data, balanced)',
    'AUC': round(xgb_metrics['auc'], 4),
    'Precision': round(xgb_metrics['precision'], 4),
    'Recall': round(xgb_metrics['recall'], 4),
    'F1': round(xgb_metrics['f1'], 4),
    'Time_sec': round(xgb_time, 2),
    'Train_size': len(X_train_q),
    'Status': 'OK',
})


# 9d. LightGBM on full data with class weighting
print("\n--- LightGBM (FULL data, class_weight balanced) ---")
t0 = time.time()
lgbm_trained = LGBMClassifier(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    class_weight='balanced',
    random_state=42, verbose=-1, n_jobs=-1,
)
lgbm_trained.fit(X_train_combined, y_train_np)
lgbm_time = time.time() - t0
lgbm_probs = lgbm_trained.predict_proba(X_test_combined)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test_np, lgbm_probs)
lgbm_thresh = thresholds[(tpr - fpr).argmax()]
lgbm_preds = (lgbm_probs >= lgbm_thresh).astype(int)
lgbm_metrics = {
    'auc': roc_auc_score(y_test_np, lgbm_probs),
    'precision': precision_score(y_test_np, lgbm_preds, zero_division=0),
    'recall': recall_score(y_test_np, lgbm_preds, zero_division=0),
    'f1': f1_score(y_test_np, lgbm_preds, zero_division=0),
    'probs': lgbm_probs,
    'preds': lgbm_preds,
}
print(f"  LightGBM (full): AUC={lgbm_metrics['auc']:.4f} "
      f"Prec={lgbm_metrics['precision']:.4f} Rec={lgbm_metrics['recall']:.4f} "
      f"F1={lgbm_metrics['f1']:.4f} Time={lgbm_time:.1f}s")
ablation.append({
    'Model': 'LightGBM (full data, balanced)',
    'AUC': round(lgbm_metrics['auc'], 4),
    'Precision': round(lgbm_metrics['precision'], 4),
    'Recall': round(lgbm_metrics['recall'], 4),
    'F1': round(lgbm_metrics['f1'], 4),
    'Time_sec': round(lgbm_time, 2),
    'Train_size': len(X_train_q),
    'Status': 'OK',
})


# 10. Subgroup fairness audit (on full test set, using XGBoost as the headline binary model)
print("\n" + "=" * 75)
print("  BINARY FAIRNESS AUDIT (using XGBoost full-data model)")
print("=" * 75)

race_cols = [c for c in classical_features if c.startswith('race_')]
race_idx = [classical_features.index(c) for c in race_cols]
test_race_groups = X_test_c.cpu().numpy()[:, race_idx].argmax(axis=1)

print(f"  Threshold (Youden's J): {xgb_thresh:.4f}")
print(f"  {'Subgroup':<45} {'N':>5} {'Acc':>6} {'AUC':>6} {'TPR':>6} {'FPR':>6} {'PosRate':>8}")
print("  " + "-" * 90)

fairness_xgb = {}
for g, col in enumerate(race_cols):
    label = RACE_LABELS.get(col, col)
    mask = test_race_groups == g
    n = int(mask.sum())
    if n < 20:
        print(f"  SKIP {label}: N={n}")
        continue
    y_sub = y_test_np[mask]
    p_sub = xgb_preds[mask]
    pr_sub = xgb_probs[mask]
    acc = accuracy_score(y_sub, p_sub)
    cm = confusion_matrix(y_sub, p_sub, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        tpr_v = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
        fpr_v = fp / (fp + tn) if (fp + tn) > 0 else float('nan')
    else:
        tpr_v = fpr_v = float('nan')
    try:
        sub_auc = roc_auc_score(y_sub, pr_sub)
    except ValueError:
        sub_auc = float('nan')
    pos_rate = p_sub.mean()
    fairness_xgb[label] = {
        'n': n, 'accuracy': float(acc), 'auc': float(sub_auc),
        'tpr': float(tpr_v) if not np.isnan(tpr_v) else None,
        'fpr': float(fpr_v) if not np.isnan(fpr_v) else None,
        'pos_rate': float(pos_rate),
    }
    print(f"  {label:<45} {n:>5} {acc:>6.4f} {sub_auc:>6.4f} {tpr_v:>6.4f} {fpr_v:>6.4f} {pos_rate:>8.4f}")

# Compute gaps
tprs = [v['tpr'] for v in fairness_xgb.values() if v['tpr'] is not None]
fprs = [v['fpr'] for v in fairness_xgb.values() if v['fpr'] is not None]
prs = [v['pos_rate'] for v in fairness_xgb.values()]
gap_dp = max(prs) - min(prs) if len(prs) >= 2 else float('nan')
gap_tpr = max(tprs) - min(tprs) if len(tprs) >= 2 else float('nan')
gap_fpr = max(fprs) - min(fprs) if len(fprs) >= 2 else float('nan')
print(f"\n  Demographic Parity Difference: {gap_dp:.4f}")
print(f"  Equal Opportunity Diff (TPR gap): {gap_tpr:.4f}")
print(f"  FPR Parity Difference: {gap_fpr:.4f}")


# 11. Feature importance
print("\n" + "=" * 75)
print("  FEATURE IMPORTANCE (XGBoost gain)")
print("=" * 75)

xgb_importance = dict(zip(all_feature_names, xgb_trained.feature_importances_))
lgbm_importance = dict(zip(all_feature_names, lgbm_trained.feature_importances_ /
                            lgbm_trained.feature_importances_.sum()))

top = sorted(xgb_importance.items(), key=lambda kv: -kv[1])[:12]
print(f"  {'Feature':<45} {'XGB gain':>10} {'LGBM gain':>10}")
print("  " + "-" * 65)
for feat, gain in top:
    lg = lgbm_importance.get(feat, 0)
    print(f"  {feat:<45} {gain:>10.4f} {lg:>10.4f}")


# 12. Save
results = {
    'task': 'binary_classification_honest',
    'data': {
        'total_clean_rows': len(df_clean),
        'train_size_full': len(X_train_q),
        'test_size': len(X_test_q),
        'subsample_train': MAX_TRAIN,
        'subsample_test': MAX_TEST,
        'positive_rate_train': float(y_train.mean()),
        'positive_rate_test': float(y_test.mean()),
        'pos_weight': float(pos_weight.item()),
        'n_quantum_features': len(quantum_features),
        'n_classical_features': n_classical,
    },
    'leakage_audit': {
        'auc_flags': len(flagged_auc),
        'corr_flags': len(flagged_corr),
        'ks_flags': len(flagged_ks),
        'all_passed': n_flags == 0,
    },
    'ablation': ablation,
    'fairness_xgboost': {
        'subgroups': fairness_xgb,
        'demographic_parity_diff': float(gap_dp) if not np.isnan(gap_dp) else None,
        'equal_opportunity_tpr_gap': float(gap_tpr) if not np.isnan(gap_tpr) else None,
        'fpr_parity_diff': float(gap_fpr) if not np.isnan(gap_fpr) else None,
    },
    'feature_importance': {
        'xgboost_gain': {k: float(v) for k, v in xgb_importance.items()},
        'lightgbm_gain': {k: float(v) for k, v in lgbm_importance.items()},
    },
    'hardware': log_hardware_info(),
}

ablation_df = pd.DataFrame(ablation)
print("\n\n" + "=" * 80)
print("  SUMMARY")
print("=" * 80)
print(ablation_df.to_string(index=False))
ablation_df.to_csv('results/TNBC_Binary_Honest_Results.csv', index=False)
with open('results/binary_honest_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nSaved: binary_honest_results.json")
print(f"Saved: TNBC_Binary_Honest_Results.csv")
print("\nDone!")
