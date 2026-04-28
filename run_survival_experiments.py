"""
Survival Analysis: Cox PH for TNBC with quantum-classical hybrid models.

This script reframes the prediction task from binary classification to proper
time-to-event survival analysis, using Cox proportional hazards. Censored
patients are retained (they contribute partial information to the partial
likelihood). The primary metric is Harrell's concordance index (C-index).
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
from scipy import stats
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index


# 0. Determinism
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


# 1. Data Loading & Preprocessing
print("\n" + "=" * 60)
print("  DATA LOADING & PREPROCESSING (SURVIVAL ANALYSIS)")
print("=" * 60)

df = pd.read_csv('breast_cancer_4quantum.csv')
print(f"Loaded: {df.shape}")

# Build (duration, event) pair from raw data
durations_all = pd.to_numeric(df['Survival_months'], errors='coerce')
events_all = (df['Vital_status_recode_study_cutoff_used_'].astype(str).str.lower() == 'dead').astype(float)
mask = ~durations_all.isna() & (durations_all > 0)
df = df[mask].copy()
df['_duration'] = durations_all[mask].values
df['_event'] = events_all[mask].values
print(f"After duration/event extraction: {len(df)} patients")
print(f"  Event rate (deaths): {df['_event'].mean():.2%}")
print(f"  Median follow-up: {df['_duration'].median():.1f} months")

# Drop columns that aren't features (including leakage and outcome columns)
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
    'Survival_months',  # used for _duration only
    'survival_60_months',  # binary version not used here
    'Year_of_follow_up_recode',  # leakage
]
df = df.drop(columns=[c for c in initial_drops if c in df.columns])

# Standardize text columns
text_cols = ['Summary_stage_2000_1998_2017_', 'race_encoded', 'Laterality',
             'Marital_status_at_diagnosis', 'Breast_Subtype_2010_']
for col in text_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.strip().str.lower()

# Stage cleaning
stage_map = {'localized': 0, 'regional': 1, 'distant': 2}
df = df[df['Summary_stage_2000_1998_2017_'].isin(stage_map.keys())]
df['stage_cleaned'] = df['Summary_stage_2000_1998_2017_'].map(stage_map)

# Time to treatment
df['Time_from_diagnosis_to_treatment_in_days_recode'] = pd.to_numeric(
    df['Time_from_diagnosis_to_treatment_in_days_recode'], errors='coerce').fillna(0)

# Tumor size: backfill from CS column then median
col_tumor = 'Tumor_Size_Summary_2016_'
df[col_tumor] = pd.to_numeric(df[col_tumor], errors='coerce')
if 'CS_tumor_size_2004_2015_' in df.columns:
    cs_tumor = pd.to_numeric(df['CS_tumor_size_2004_2015_'], errors='coerce')
    df[col_tumor] = df[col_tumor].fillna(cs_tumor)
df[col_tumor] = df[col_tumor].fillna(df[col_tumor].median())

# Laterality
laterality_map = {'left - origin of primary': 0, 'left': 0,
                  'right - origin of primary': 1, 'right': 1}
df['Laterality'] = df['Laterality'].map(laterality_map).fillna(-1)

# One-hot encoding
nominal_cols = {'race_encoded': 'race', 'Marital_status_at_diagnosis': 'marital',
                'Breast_Subtype_2010_': 'subtype'}
df = pd.get_dummies(df, columns=list(nominal_cols.keys()),
                    prefix=list(nominal_cols.values()), dtype=float)

if 'Summary_stage_2000_1998_2017_' in df.columns:
    df = df.drop(columns=['Summary_stage_2000_1998_2017_'])

# Feature partitioning
quantum_features = [
    'numeric_age', 'Tumor_Size_Summary_2016_', 'income_encoded',
    'years_since_2010', 'Time_from_diagnosis_to_treatment_in_days_recode',
    'income_age_ratio', 'stage_cleaned'
]
exclude = quantum_features + ['_duration', '_event']
classical_features = [c for c in df.columns
                      if c not in exclude
                      and df[c].dtype in ['float64', 'float32', 'int64', 'int32']]

# Drop rows with missing quantum features
df_clean = df.dropna(subset=quantum_features + ['_duration', '_event'])

# Median-fill remaining classical NaN
for col in classical_features:
    if df_clean[col].isna().any():
        m = df_clean[col].median()
        if pd.isna(m):
            m = 0.0
        df_clean[col] = df_clean[col].fillna(m)

print(f"Clean cohort: {df_clean.shape}")
print(f"  Event rate: {df_clean['_event'].mean():.2%}")
print(f"  Quantum features ({len(quantum_features)}): {quantum_features}")
print(f"  Classical features ({len(classical_features)}): {len(classical_features)} columns")


# 2. Leakage Audit (still relevant)
print("\n" + "=" * 60)
print("  DATA LEAKAGE AUDIT")
print("=" * 60)
all_feature_names = quantum_features + classical_features
X_all = df_clean[all_feature_names].values
event_arr = df_clean['_event'].values
duration_arr = df_clean['_duration'].values

# For survival, leakage = features predicting duration or event too well
from sklearn.metrics import roc_auc_score
flagged_auc = []
print("\n--- Per-feature AUC vs event indicator (threshold=0.95) ---")
for i, feat in enumerate(all_feature_names):
    try:
        auc = roc_auc_score(event_arr, X_all[:, i])
        if auc < 0.5:
            auc = 1 - auc
        if auc > 0.95:
            flagged_auc.append((feat, auc))
            print(f"  WARNING: {feat} has AUC={auc:.4f}")
    except Exception:
        pass
if not flagged_auc:
    print("  All features below AUC threshold. PASS.")

flagged_corr = []
print("\n--- Feature-duration correlation (threshold=0.90) ---")
for i, feat in enumerate(all_feature_names):
    try:
        c = abs(np.corrcoef(X_all[:, i], duration_arr)[0, 1])
        if c > 0.90:
            flagged_corr.append((feat, c))
            print(f"  WARNING: {feat} has |corr|={c:.4f}")
    except Exception:
        pass
if not flagged_corr:
    print("  All features below correlation threshold. PASS.")

n_flags = len(flagged_auc) + len(flagged_corr)
print(f"\n  LEAKAGE AUDIT RESULT: {'ALL PASSED' if n_flags == 0 else f'{n_flags} FLAGS'}")


# 3. Train/test split (stratified by event)
train_idx, test_idx = train_test_split(
    np.arange(len(df_clean)), test_size=0.2, random_state=42,
    stratify=df_clean['_event']
)
df_train = df_clean.iloc[train_idx].copy()
df_test = df_clean.iloc[test_idx].copy()
print(f"\nTrain: {len(df_train)} ({df_train['_event'].mean():.2%} events)")
print(f"Test: {len(df_test)} ({df_test['_event'].mean():.2%} events)")


# 4. Scaling
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

print(f"\nQuantum tensor: {X_train_q.shape}")
print(f"Classical tensor: {X_train_c.shape}")
assert not torch.isnan(X_train_q).any() and not torch.isnan(X_train_c).any()
print("NaN check: clean")


# 5. Model definitions (same architectures, output now interpreted as log-hazard)
n_qubits = 7
try:
    dev = qml.device("lightning.qubit", wires=n_qubits)
    print("Using lightning.qubit")
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


class HybridSurvivalQ_v1(nn.Module):
    """v1: 1-layer VQC, single-qubit measurement, log-hazard output."""
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
        return self.fc3(x).squeeze(-1)  # log-hazard, no sigmoid


class HybridSurvivalQ_v2(nn.Module):
    """v2: 3-layer VQC with data re-uploading + classical encoder."""
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
        combined = torch.cat([q_out, c_out], dim=1)
        return self.fusion(combined).squeeze(-1)


class HybridSurvivalQ_v3(nn.Module):
    """v3: trainable encoding, ZZ correlators, residual connection."""
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
        combined = torch.cat([q_out, c_encoded, x_c], dim=1)
        return self.fusion(combined).squeeze(-1)


# 6. Cox partial likelihood loss
def cox_ph_loss(log_hazards, durations, events):
    """Negative log partial likelihood (Cox PH).

    Sorts by descending duration, then for each event the loss is
    log_hazard - logsumexp(log_hazards of all patients with duration >= current).
    """
    idx = torch.argsort(durations, descending=True)
    h = log_hazards[idx]
    e = events[idx]
    log_cumsum = torch.logcumsumexp(h, dim=0)
    n_events = e.sum()
    if n_events == 0:
        return torch.tensor(0.0, requires_grad=True)
    return -((h - log_cumsum) * e).sum() / n_events


# 7. Hardware logging
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


# 8. Training loop
def train_cox(model, X_q_tr, X_c_tr, dur_tr, evt_tr, X_q_te, X_c_te, dur_te, evt_te,
              epochs=50, lr=0.001, verbose=True, use_scheduler=False, time_budget=None):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-5)
    history = {'train_loss': [], 'val_loss': [], 'val_cidx': [], 'epoch_time': []}
    hw = log_hardware_info()
    if verbose:
        print(f"  Hardware: {hw['device']} | torch {hw['torch_version']}")
    total_start = time.time()
    for epoch in range(epochs):
        if time_budget and (time.time() - total_start) > time_budget:
            if verbose:
                print(f"  Time budget reached at epoch {epoch}.")
            break
        epoch_start = time.time()
        model.train()
        optimizer.zero_grad()
        log_h = model(X_q_tr, X_c_tr)
        loss = cox_ph_loss(log_h, dur_tr, evt_tr)
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
        model.eval()
        with torch.no_grad():
            val_h = model(X_q_te, X_c_te).cpu().numpy()
            val_loss = cox_ph_loss(model(X_q_te, X_c_te), dur_te, evt_te).item()
            val_cidx = concordance_index(dur_te.cpu().numpy(), -val_h, evt_te.cpu().numpy())
        history['train_loss'].append(loss.item())
        history['val_loss'].append(val_loss)
        history['val_cidx'].append(val_cidx)
        history['epoch_time'].append(time.time() - epoch_start)
        if verbose and (epoch + 1) % 10 == 0:
            print(f"  Epoch {epoch+1:02d} | Loss: {loss.item():.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val C-idx: {val_cidx:.4f} | "
                  f"Time: {history['epoch_time'][-1]:.1f}s")
    history['total_time'] = time.time() - total_start
    history['hardware'] = hw
    if verbose:
        print(f"  Total: {history['total_time']:.1f}s ({history['total_time']/60:.1f}m)")
    return history


def train_cox_minibatch(model, X_q_tr, X_c_tr, dur_tr, evt_tr,
                         X_q_te, X_c_te, dur_te, evt_te,
                         epochs=30, lr=0.001, batch_size=256,
                         verbose=True, use_scheduler=True, time_budget=None):
    """Mini-batch Cox training (DeepSurv-style). Each batch computes its own
    partial likelihood using only the patients in the batch as the risk set.
    This is an approximation but converges to the true Cox loss in expectation."""
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-5)
    n = len(X_q_tr)
    history = {'train_loss': [], 'val_loss': [], 'val_cidx': [], 'epoch_time': []}
    hw = log_hardware_info()
    if verbose:
        print(f"  Hardware: {hw['device']} | Batch size: {batch_size} | Train N: {n}")
    total_start = time.time()
    for epoch in range(epochs):
        if time_budget and (time.time() - total_start) > time_budget:
            if verbose:
                print(f"  Time budget reached at epoch {epoch}.")
            break
        epoch_start = time.time()
        model.train()
        perm = torch.randperm(n)
        batch_losses = []
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            if evt_tr[idx].sum() == 0:
                continue  # Skip batches with no events (no signal for Cox)
            optimizer.zero_grad()
            log_h = model(X_q_tr[idx], X_c_tr[idx])
            loss = cox_ph_loss(log_h, dur_tr[idx], evt_tr[idx])
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        if scheduler:
            scheduler.step()
        avg_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
        model.eval()
        with torch.no_grad():
            val_h = model(X_q_te, X_c_te).cpu().numpy()
            val_loss = cox_ph_loss(model(X_q_te, X_c_te), dur_te, evt_te).item()
            val_cidx = concordance_index(dur_te.cpu().numpy(), -val_h, evt_te.cpu().numpy())
        history['train_loss'].append(avg_loss)
        history['val_loss'].append(val_loss)
        history['val_cidx'].append(val_cidx)
        history['epoch_time'].append(time.time() - epoch_start)
        if verbose and (epoch + 1) % 5 == 0:
            print(f"  Epoch {epoch+1:02d} | Train Loss: {avg_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val C-idx: {val_cidx:.4f} | "
                  f"Time: {history['epoch_time'][-1]:.1f}s")
    history['total_time'] = time.time() - total_start
    history['hardware'] = hw
    if verbose:
        print(f"  Total: {history['total_time']:.1f}s ({history['total_time']/60:.1f}m)")
    return history


def evaluate_cox(model, X_q, X_c, durations, events):
    """Compute C-index on test set."""
    model.eval()
    with torch.no_grad():
        log_h = model(X_q, X_c).cpu().numpy()
    cidx = concordance_index(durations.cpu().numpy(), -log_h, events.cpu().numpy())
    return {'c_index': cidx, 'log_hazards': log_h}


# 9. Subsampling for hybrid models (PennyLane is slow per-sample)
MAX_TRAIN = 2000
MAX_TEST = 500
if len(X_train_q) > MAX_TRAIN:
    print(f"\nSubsampling: {MAX_TRAIN} train, {MAX_TEST} test (from {len(X_train_q)}/{len(X_test_q)})")
    np.random.seed(42)
    idx_tr = np.random.choice(len(X_train_q), MAX_TRAIN, replace=False)
    idx_te = np.random.choice(len(X_test_q), MAX_TEST, replace=False)
    X_train_q_sub = X_train_q[idx_tr]
    X_train_c_sub = X_train_c[idx_tr]
    durations_train_sub = durations_train[idx_tr]
    events_train_sub = events_train[idx_tr]
    X_test_q_sub = X_test_q[idx_te]
    X_test_c_sub = X_test_c[idx_te]
    durations_test_sub = durations_test[idx_te]
    events_test_sub = events_test[idx_te]
else:
    X_train_q_sub, X_train_c_sub = X_train_q, X_train_c
    durations_train_sub, events_train_sub = durations_train, events_train
    X_test_q_sub, X_test_c_sub = X_test_q, X_test_c
    durations_test_sub, events_test_sub = durations_test, events_test


# 10. Cox PH classical baseline (lifelines, full data)
print("\n" + "=" * 60)
print("  COX PH BASELINE (LIFELINES)")
print("=" * 60)
ablation = []

cox_train_df = pd.DataFrame(
    np.hstack([Xq_train_scaled, Xc_train_scaled]),
    columns=all_feature_names
)
cox_train_df['duration'] = df_train['_duration'].values
cox_train_df['event'] = df_train['_event'].values
cox_test_df = pd.DataFrame(
    np.hstack([Xq_test_scaled, Xc_test_scaled]),
    columns=all_feature_names
)
cox_test_df['duration'] = df_test['_duration'].values
cox_test_df['event'] = df_test['_event'].values

# Drop low-variance features that break Cox PH
keep_cols = [c for c in all_feature_names if cox_train_df[c].std() > 1e-6]
cox_keep = keep_cols + ['duration', 'event']

cox_start = time.time()
cph = CoxPHFitter(penalizer=0.01)
cph.fit(cox_train_df[cox_keep], duration_col='duration', event_col='event',
        show_progress=False)
cox_time = time.time() - cox_start
cox_train_cidx = concordance_index(
    cox_train_df['duration'], -cph.predict_log_partial_hazard(cox_train_df[keep_cols]),
    cox_train_df['event']
)
cox_test_cidx = concordance_index(
    cox_test_df['duration'], -cph.predict_log_partial_hazard(cox_test_df[keep_cols]),
    cox_test_df['event']
)
print(f"Cox PH (lifelines, full {len(df_train)}): "
      f"Train C-idx={cox_train_cidx:.4f}, Test C-idx={cox_test_cidx:.4f}, Time={cox_time:.1f}s")
ablation.append({'Model': 'Cox PH (lifelines)', 'Test_Cindex': round(cox_test_cidx, 4),
                 'Train_Cindex': round(cox_train_cidx, 4), 'Time_sec': round(cox_time, 1),
                 'Status': 'OK'})


# 11. Hybrid models with Cox loss
print("\n" + "=" * 80)
print("  HYBRID SURVIVAL MODELS (COX PARTIAL LIKELIHOOD)")
print("=" * 80)


def run_survival_trial(name, model, ep=50, lr=0.001, scheduler=False, tb=None):
    set_determinism(42)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    start = time.time()
    try:
        train_cox(model, X_train_q_sub, X_train_c_sub, durations_train_sub, events_train_sub,
                  X_test_q_sub, X_test_c_sub, durations_test_sub, events_test_sub,
                  epochs=ep, lr=lr, verbose=True, use_scheduler=scheduler, time_budget=tb)
        metrics = evaluate_cox(model, X_test_q_sub, X_test_c_sub,
                               durations_test_sub, events_test_sub)
        elapsed = time.time() - start
        print(f"  {name}: C-idx={metrics['c_index']:.4f} Time={elapsed:.1f}s")
        return {'Model': name, 'Test_Cindex': round(metrics['c_index'], 4),
                'Train_Cindex': None, 'Time_sec': round(elapsed, 1), 'Status': 'OK'}
    except Exception as e:
        elapsed = time.time() - start
        print(f"  {name}: FAILED ({e})")
        return {'Model': name, 'Test_Cindex': None, 'Train_Cindex': None,
                'Time_sec': round(elapsed, 1), 'Status': f'FAILED: {e}'}


print("\nTrial 1: HybridSurvivalQ_v1 (1-layer VQC)")
ablation.append(run_survival_trial("HybridSurvivalQ_v1", HybridSurvivalQ_v1(n_classical), ep=50))

print("\nTrial 2: HybridSurvivalQ_v2 (3-layer VQC, all-qubit measurement)")
ablation.append(run_survival_trial("HybridSurvivalQ_v2", HybridSurvivalQ_v2(n_classical),
                                    ep=75, scheduler=True, tb=570))

print("\nTrial 3: HybridSurvivalQ_v3 (trainable encoding, ZZ correlators, residual)")
ablation.append(run_survival_trial("HybridSurvivalQ_v3", HybridSurvivalQ_v3(n_classical),
                                    ep=75, scheduler=True, tb=600))

# Trial 4: HybridSurvivalQ_v3 trained on FULL DATA with mini-batch Cox loss
print("\nTrial 4: HybridSurvivalQ_v3 (FULL DATA, mini-batch Cox)")
set_determinism(42)
if torch.cuda.is_available():
    torch.cuda.empty_cache()
gc.collect()
v3_full_model = HybridSurvivalQ_v3(n_classical)
v3_full_start = time.time()
try:
    v3_full_history = train_cox_minibatch(
        v3_full_model, X_train_q, X_train_c, durations_train, events_train,
        X_test_q, X_test_c, durations_test, events_test,
        epochs=20, lr=0.001, batch_size=256,
        use_scheduler=True, time_budget=900
    )
    v3_full_metrics = evaluate_cox(v3_full_model, X_test_q, X_test_c,
                                    durations_test, events_test)
    v3_full_time = time.time() - v3_full_start
    print(f"  HybridSurvivalQ_v3 (full): C-idx={v3_full_metrics['c_index']:.4f} Time={v3_full_time:.1f}s")
    ablation.append({'Model': 'HybridSurvivalQ_v3 (full data)',
                     'Test_Cindex': round(v3_full_metrics['c_index'], 4),
                     'Train_Cindex': None,
                     'Time_sec': round(v3_full_time, 1), 'Status': 'OK'})
except Exception as e:
    v3_full_time = time.time() - v3_full_start
    print(f"  v3 (full data) FAILED: {e}")
    ablation.append({'Model': 'HybridSurvivalQ_v3 (full data)',
                     'Test_Cindex': None, 'Train_Cindex': None,
                     'Time_sec': round(v3_full_time, 1),
                     'Status': f'FAILED: {e}'})


# 12. Subgroup C-index (fairness audit)
print("\n" + "=" * 75)
print("  FAIRNESS AUDIT (SUBGROUP C-INDEX)")
print("=" * 75)


def subgroup_cindex(log_hazards, durations, events, groups, group_names):
    """Compute C-index for each subgroup."""
    print(f"  {'Subgroup':<35} {'N':>5} {'Events':>7} {'C-index':>9}")
    print("  " + "-" * 60)
    results = {}
    cindices = []
    for g, name in group_names.items():
        mask = groups == g
        n = mask.sum()
        n_evt = events[mask].sum() if n > 0 else 0
        if n < 20 or n_evt < 5:
            print(f"  SKIP {name}: N={n}, events={n_evt}")
            continue
        try:
            ci = concordance_index(durations[mask], -log_hazards[mask], events[mask])
            results[name] = {'n': int(n), 'events': int(n_evt), 'c_index': ci}
            cindices.append(ci)
            print(f"  {name:<35} {n:>5} {int(n_evt):>7} {ci:>9.4f}")
        except Exception as ee:
            print(f"  ERROR {name}: {ee}")
    if len(cindices) >= 2:
        gap = max(cindices) - min(cindices)
        print(f"\n  Subgroup C-index gap: {gap:.4f}")
        return results, gap
    return results, float('nan')


# Find race groups in test set
race_cols = [c for c in classical_features if c.startswith('race_')]
race_idx_in_classical = [classical_features.index(c) for c in race_cols]
test_race_onehot = X_test_c_sub.cpu().numpy()[:, race_idx_in_classical]
test_race_groups = test_race_onehot.argmax(axis=1)
race_group_names = {i: race_cols[i] for i in range(len(race_cols))}

# Audit each model
print("\n--- Cox PH (lifelines) ---")
# Need to subsample cox predictions to test_idx_sub
np.random.seed(42)
test_indices_full = np.arange(len(X_test_q))
te_sub_idx = np.random.choice(len(X_test_q), MAX_TEST, replace=False) if len(X_test_q) > MAX_TEST else test_indices_full
cox_test_sub_df = cox_test_df.iloc[te_sub_idx]
# pass raw log-hazards; subgroup_cindex negates internally
cox_logh_sub = cph.predict_log_partial_hazard(cox_test_sub_df[keep_cols]).values
cox_subgroup, cox_gap = subgroup_cindex(
    cox_logh_sub, cox_test_sub_df['duration'].values,
    cox_test_sub_df['event'].values, test_race_groups, race_group_names)

# v3 full-data subgroup audit (on full test set, not subsample)
v3_subgroup = {}
v3_gap = float('nan')
if 'v3_full_model' in dir() and 'v3_full_metrics' in dir():
    print("\n--- HybridSurvivalQ_v3 (full data) ---")
    full_test_race_onehot = X_test_c.cpu().numpy()[:, race_idx_in_classical]
    full_test_race_groups = full_test_race_onehot.argmax(axis=1)
    v3_full_logh = v3_full_metrics['log_hazards']
    v3_subgroup, v3_gap = subgroup_cindex(
        v3_full_logh, durations_test.cpu().numpy(),
        events_test.cpu().numpy(), full_test_race_groups, race_group_names)


# 13. Save results
print("\n" + "=" * 60)
print("  SAVING RESULTS")
print("=" * 60)

ablation_df = pd.DataFrame(ablation)
print("\nSurvival Analysis Results:")
print(ablation_df.to_string(index=False))
ablation_df.to_csv('results/TNBC_Survival_Ablation_Results.csv', index=False)

results = {
    'task': 'survival_analysis_cox_ph',
    'data': {
        'total_patients': int(len(df_clean)),
        'event_rate': float(df_clean['_event'].mean()),
        'median_followup_months': float(df_clean['_duration'].median()),
        'train_size': int(len(df_train)),
        'test_size': int(len(df_test)),
        'subsample_train': MAX_TRAIN,
        'subsample_test': MAX_TEST,
        'n_quantum_features': len(quantum_features),
        'n_classical_features': n_classical,
    },
    'leakage_audit': {
        'auc_flags': len(flagged_auc),
        'corr_flags': len(flagged_corr),
        'all_passed': n_flags == 0,
    },
    'ablation': ablation,
    'fairness_cox': {
        'subgroups': {k: {kk: round(vv, 4) if isinstance(vv, float) else vv
                          for kk, vv in v.items()} for k, v in cox_subgroup.items()},
        'cindex_gap': round(cox_gap, 4) if not np.isnan(cox_gap) else None,
    },
    'fairness_v3_full': {
        'subgroups': {k: {kk: round(vv, 4) if isinstance(vv, float) else vv
                          for kk, vv in v.items()} for k, v in v3_subgroup.items()},
        'cindex_gap': round(v3_gap, 4) if not np.isnan(v3_gap) else None,
    },
    'hardware': log_hardware_info(),
}

with open('results/survival_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print("\nSaved: TNBC_Survival_Ablation_Results.csv")
print("Saved: survival_results.json")
print("\nDone!")
