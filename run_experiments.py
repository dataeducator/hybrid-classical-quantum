"""
Run all experiments from the notebook in a standalone script.
Generates results for the paper (ablation table, fairness audit, leakage audit).
"""
import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True)

import time
import platform
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
import os
import json

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import (roc_auc_score, confusion_matrix, roc_curve,
                             precision_score, recall_score, f1_score, accuracy_score)
from scipy import stats

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
print("  DATA LOADING & PREPROCESSING")
print("=" * 60)

df = pd.read_csv('breast_cancer_4quantum.csv')
print(f"Loaded: {df.shape}")

# Drop non-predictive columns
initial_drops = [
    'Patient_ID', 'age_group', 'Sex_no_total_',
    'RX_Summ_Systemic_Sur_Seq_2007_', 'Diagnostic_Confirmation',
    'Histologic_Type_ICD_O_3', 'Primary_Site_labeled',
    'Combined_Summary_Stage_with_Expanded_Regional_Codes_2004_',
    'Reason_no_cancer_directed_surgery', 'Grade_Clinical_2018_',
    'COD_to_site_recode', 'Vital_status_recode_study_cutoff_used_',
    'Grade_Pathological_2018_', 'Race_and_origin_recode_NHW,_NHB,_NHAIAN,_NHAPI,_Hispanic_no_total',
    'Median_household_income_inflation_adj_to_2023', 'Age_recode_with_<1_year_olds_and_90_',
    'Rural_Urban_Continuum_Code', 'stage_encoded'
]
columns_to_drop = [c for c in initial_drops if c in df.columns]
df = df.drop(columns=columns_to_drop)

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
col_ttt = 'Time_from_diagnosis_to_treatment_in_days_recode'
df[col_ttt] = pd.to_numeric(df[col_ttt], errors='coerce').fillna(0)

# Tumor size
col_tumor = 'Tumor_Size_Summary_2016_'
df[col_tumor] = pd.to_numeric(df[col_tumor], errors='coerce')

# Laterality
laterality_map = {
    'left - origin of primary': 0, 'left': 0,
    'right - origin of primary': 1, 'right': 1
}
df['Laterality'] = df['Laterality'].map(laterality_map).fillna(-1)

# One-hot encode nominal variables
nominal_cols = {'race_encoded': 'race', 'Marital_status_at_diagnosis': 'marital',
                'Breast_Subtype_2010_': 'subtype'}
df = pd.get_dummies(df, columns=list(nominal_cols.keys()),
                    prefix=list(nominal_cols.values()), dtype=float)

if 'Summary_stage_2000_1998_2017_' in df.columns:
    df = df.drop(columns=['Summary_stage_2000_1998_2017_'])

# Feature sets
quantum_features = [
    'numeric_age', 'Tumor_Size_Summary_2016_', 'income_encoded',
    'years_since_2010', 'Time_from_diagnosis_to_treatment_in_days_recode',
    'income_age_ratio', 'stage_cleaned'
]
target_col = 'survival_60_months'

# Drop leaky/non-predictive columns that shouldn't be features
cols_to_exclude = quantum_features + [target_col, 'Survival_months',
    'Vital_status_recode_study_cutoff_used_']

# Only keep numeric columns as classical features (strings need encoding first)
classical_features = [col for col in df.columns
                      if col not in cols_to_exclude
                      and df[col].dtype in ['float64', 'float32', 'int64', 'int32']]

df_clean = df.dropna(subset=quantum_features + [target_col])

# Fill remaining NaN in classical features with column median
for col in classical_features:
    if df_clean[col].isna().any():
        median_val = df_clean[col].median()
        n_filled = df_clean[col].isna().sum()
        df_clean[col] = df_clean[col].fillna(median_val)
        print(f"  Filled {n_filled} NaN in {col} with median={median_val:.2f}")

print(f"Clean data: {df_clean.shape}")
print(f"Target distribution:\n{df_clean[target_col].value_counts()}")

# Train/test split
Xq_train, Xq_test, Xc_train, Xc_test, y_train, y_test = train_test_split(
    df_clean[quantum_features], df_clean[classical_features], df_clean[target_col],
    test_size=0.2, random_state=42, stratify=df_clean[target_col]
)
print(f"Train: {len(Xq_train)}, Test: {len(Xq_test)}")

# 2. Leakage Audit
print("\n" + "=" * 60)
print("  DATA LEAKAGE AUDIT")
print("=" * 60)

all_feature_names = quantum_features + classical_features
X_train_all = pd.concat([Xq_train, Xc_train], axis=1).values
X_test_all = pd.concat([Xq_test, Xc_test], axis=1).values

flagged_auc = []
print("\n--- Check 1: Per-Feature AUC Scan (threshold=0.95) ---")
for i, feat in enumerate(all_feature_names):
    try:
        auc = roc_auc_score(y_train.values, X_train_all[:, i])
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
print("\n--- Check 2: Feature-Target Correlation (threshold=0.90) ---")
for i, feat in enumerate(all_feature_names):
    try:
        corr = abs(np.corrcoef(X_train_all[:, i], y_train.values)[0, 1])
        if corr > 0.90:
            flagged_corr.append((feat, corr))
            print(f"  WARNING: {feat} has |corr|={corr:.4f}")
    except Exception:
        pass
if not flagged_corr:
    print("  All features below correlation threshold. PASS.")

flagged_ks = []
print("\n--- Check 3: Train/Test Distribution Shift (KS test, p<0.001) ---")
for i, feat in enumerate(all_feature_names):
    try:
        ks_stat, p_val = stats.ks_2samp(X_train_all[:, i], X_test_all[:, i])
        if p_val < 0.001:
            flagged_ks.append((feat, ks_stat, p_val))
    except Exception:
        pass
if flagged_ks:
    print(f"  {len(flagged_ks)} feature(s) show significant shift:")
    for feat, ks, p in sorted(flagged_ks, key=lambda x: -x[1])[:5]:
        print(f"    {feat}: KS={ks:.4f}, p={p:.2e}")
else:
    print("  No significant distribution shift detected. PASS.")

n_flags = len(flagged_auc) + len(flagged_corr) + len(flagged_ks)
print(f"\n  LEAKAGE AUDIT: {'ALL CHECKS PASSED' if n_flags == 0 else f'{n_flags} FLAG(S)'}")

# 3. Scaling
q_scaler = MinMaxScaler()
c_scaler = StandardScaler()
Xq_train_scaled = q_scaler.fit_transform(Xq_train)
Xc_train_scaled = c_scaler.fit_transform(Xc_train)
Xq_test_scaled = q_scaler.transform(Xq_test)
Xc_test_scaled = c_scaler.transform(Xc_test)

X_train_q = torch.tensor(np.pi * Xq_train_scaled, dtype=torch.float32)
X_test_q = torch.tensor(np.pi * Xq_test_scaled, dtype=torch.float32)
X_train_classical = torch.tensor(Xc_train_scaled, dtype=torch.float32)
X_test_classical = torch.tensor(Xc_test_scaled, dtype=torch.float32)
y_train_torch = torch.tensor(y_train.values, dtype=torch.float32).reshape(-1, 1)
y_test_torch = torch.tensor(y_test.values, dtype=torch.float32).reshape(-1, 1)
n_classical = X_train_classical.shape[1]

print(f"\nQuantum tensor: {X_train_q.shape}")
print(f"Classical tensor: {X_train_classical.shape}")

# Verify no NaN in tensors
assert not torch.isnan(X_train_q).any(), "NaN in quantum train!"
assert not torch.isnan(X_train_classical).any(), "NaN in classical train!"
assert not torch.isnan(X_test_q).any(), "NaN in quantum test!"
assert not torch.isnan(X_test_classical).any(), "NaN in classical test!"
print("NaN check: all tensors clean")

# 4. Model Definitions
n_qubits = 7
try:
    dev = qml.device("lightning.qubit", wires=n_qubits)
    print("Using lightning.qubit (fast CPU simulator)")
except Exception:
    dev = qml.device("default.qubit", wires=n_qubits)
    print("Using default.qubit simulator")

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
        batch_size = x_q.shape[0]
        q_out = []
        for i in range(batch_size):
            res = quantum_circuit(x_q[i], self.q_params)
            q_out.append(res)
        q_out = torch.stack(q_out).unsqueeze(1).to(torch.float32)
        combined = torch.cat([q_out, x_c], dim=1)
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


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


class QuantumOnly(nn.Module):
    def __init__(self, n_qubits_in):
        super().__init__()
        self.q_params = nn.Parameter(torch.randn(n_qubits))
        self.fc = nn.Linear(1, 1)

    def forward(self, x_q, x_c):
        q_out = []
        for i in range(x_q.shape[0]):
            res = quantum_circuit(x_q[i], self.q_params)
            q_out.append(res)
        q_out = torch.stack(q_out).unsqueeze(1).to(torch.float32)
        return self.fc(q_out)


# Reduced quantum (3-qubit)
n_qubits_reduced = 3
try:
    dev_reduced = qml.device("lightning.qubit", wires=n_qubits_reduced)
except Exception:
    dev_reduced = qml.device("default.qubit", wires=n_qubits_reduced)

@qml.qnode(dev_reduced, interface="torch")
def q_circuit_reduced(inputs, weights):
    for i in range(n_qubits_reduced):
        qml.RY(inputs[i], wires=i)
    for i in range(n_qubits_reduced):
        qml.RY(weights[i], wires=i)
    for i in range(n_qubits_reduced):
        qml.CNOT(wires=[i, (i + 1) % n_qubits_reduced])
    return qml.expval(qml.PauliZ(0))


class HybridReducedQ(nn.Module):
    def __init__(self, n_classical_features):
        super().__init__()
        self.q_params = nn.Parameter(torch.randn(n_qubits_reduced))
        self.fc1 = nn.Linear(1 + n_classical_features, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x_q, x_c):
        q_out = []
        for i in range(x_q.shape[0]):
            res = q_circuit_reduced(x_q[i], self.q_params)
            q_out.append(res)
        q_out = torch.stack(q_out).unsqueeze(1).to(torch.float32)
        combined = torch.cat([q_out, x_c], dim=1)
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# Deep quantum (2-layer)
@qml.qnode(dev, interface="torch")
def q_circuit_deep(inputs, weights):
    for i in range(n_qubits):
        qml.RY(inputs[i], wires=i)
    for layer in range(2):
        for i in range(n_qubits):
            qml.RY(weights[layer][i], wires=i)
        for i in range(n_qubits):
            qml.CNOT(wires=[i, (i + 1) % n_qubits])
    return qml.expval(qml.PauliZ(0))


class HybridDeepQ(nn.Module):
    def __init__(self, n_classical_features):
        super().__init__()
        self.q_params = nn.Parameter(torch.randn(2, n_qubits))
        self.fc1 = nn.Linear(1 + n_classical_features, 32)
        self.fc2 = nn.Linear(32, 16)
        self.fc3 = nn.Linear(16, 1)

    def forward(self, x_q, x_c):
        q_out = []
        for i in range(x_q.shape[0]):
            res = q_circuit_deep(x_q[i], self.q_params)
            q_out.append(res)
        q_out = torch.stack(q_out).unsqueeze(1).to(torch.float32)
        combined = torch.cat([q_out, x_c], dim=1)
        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# 4b. Improved Quantum Circuit (v2): 3-layer, data re-uploading, all-qubit measurement
@qml.qnode(dev, interface="torch")
def quantum_circuit_v2(inputs, weights_ry, weights_rz):
    """3-layer VQC with data re-uploading, RY+RZ variational gates,
    shifted-ring entanglement, and all-qubit measurement."""
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


class HybridRealQ_v2(nn.Module):
    """Improved hybrid model: 3-layer VQC with data re-uploading, all-qubit
    measurement (7D), separate classical encoder, deeper fusion with dropout."""
    def __init__(self, n_classical_features):
        super().__init__()
        self.q_params_ry = nn.Parameter(torch.randn(3, n_qubits) * 0.1)
        self.q_params_rz = nn.Parameter(torch.randn(3, n_qubits) * 0.1)
        self.classical_encoder = nn.Sequential(
            nn.Linear(n_classical_features, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(7 + 32, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x_q, x_c):
        x_q = x_q.to(torch.float32)
        x_c = x_c.to(torch.float32)
        batch_size = x_q.shape[0]
        q_out = []
        for i in range(batch_size):
            res = quantum_circuit_v2(x_q[i], self.q_params_ry, self.q_params_rz)
            q_out.append(torch.stack(list(res)))
        q_out = torch.stack(q_out).to(torch.float32)
        c_out = self.classical_encoder(x_c)
        combined = torch.cat([q_out, c_out], dim=1)
        return self.fusion(combined)


# 5. Hardware Logging + Training
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


def train_model(model, train_data, test_data, epochs=50, lr=0.001, verbose=True,
                use_scheduler=False, time_budget=None):
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = None
    if use_scheduler:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=1e-5)
    xq_train, xc_train, y_train_t = train_data
    xq_test, xc_test, y_test_t = test_data
    history = {'train_loss': [], 'val_loss': [], 'epoch_time': [], 'lr': []}
    hw_info = log_hardware_info()
    if verbose:
        print(f"  Hardware: {hw_info['device']} | torch {hw_info['torch_version']}")
    total_start = time.time()
    for epoch in range(epochs):
        if time_budget and (time.time() - total_start) > time_budget:
            if verbose:
                print(f"  Time budget ({time_budget}s) reached at epoch {epoch}. Stopping.")
            break
        epoch_start = time.time()
        model.train()
        optimizer.zero_grad()
        logits = model(xq_train, xc_train).squeeze()
        loss = criterion(logits, y_train_t.squeeze())
        loss.backward()
        optimizer.step()
        if scheduler:
            scheduler.step()
        model.eval()
        with torch.no_grad():
            val_logits = model(xq_test, xc_test).squeeze()
            val_loss = criterion(val_logits, y_test_t.squeeze())
        epoch_time = time.time() - epoch_start
        history['train_loss'].append(loss.item())
        history['val_loss'].append(val_loss.item())
        history['epoch_time'].append(epoch_time)
        history['lr'].append(optimizer.param_groups[0]['lr'])
        if verbose and (epoch + 1) % 10 == 0:
            cur_lr = optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch+1:02d} | Loss: {loss.item():.4f} | Val Loss: {val_loss.item():.4f} | LR: {cur_lr:.6f} | Time: {epoch_time:.1f}s")
    total_time = time.time() - total_start
    history['total_time'] = total_time
    history['hardware'] = hw_info
    if verbose:
        print(f"  Total: {total_time:.1f}s ({total_time/60:.1f}m)")
    return history


def evaluate_model(model, x_q, x_c, y_true_np):
    model.eval()
    y_true_np = y_true_np.reshape(-1)
    with torch.no_grad():
        logits = model(x_q, x_c).squeeze()
        probs = torch.sigmoid(logits).cpu().numpy()
    auc = roc_auc_score(y_true_np, probs)
    fpr, tpr, thresholds = roc_curve(y_true_np, probs)
    j_scores = tpr - fpr
    best_thresh = thresholds[j_scores.argmax()]
    preds = (probs >= best_thresh).astype(int)
    prec = precision_score(y_true_np, preds, zero_division=0)
    rec = recall_score(y_true_np, preds, zero_division=0)
    f1 = f1_score(y_true_np, preds, zero_division=0)
    return {'auc': auc, 'precision': prec, 'recall': rec, 'f1': f1,
            'threshold': best_thresh, 'probs': probs, 'preds': preds}


# 6. Subsample for feasibility (quantum circuits are slow)
# Use a subsample for training quantum models to keep runtime reasonable
MAX_TRAIN = 2000
MAX_TEST = 500

if len(X_train_q) > MAX_TRAIN:
    print(f"\nSubsampling: {MAX_TRAIN} train, {MAX_TEST} test (from {len(X_train_q)}/{len(X_test_q)})")
    idx_train = np.random.choice(len(X_train_q), MAX_TRAIN, replace=False)
    idx_test = np.random.choice(len(X_test_q), MAX_TEST, replace=False)
    X_train_q_sub = X_train_q[idx_train]
    X_train_c_sub = X_train_classical[idx_train]
    y_train_sub = y_train_torch[idx_train]
    X_test_q_sub = X_test_q[idx_test]
    X_test_c_sub = X_test_classical[idx_test]
    y_test_sub = y_test_torch[idx_test]
else:
    X_train_q_sub = X_train_q
    X_train_c_sub = X_train_classical
    y_train_sub = y_train_torch
    X_test_q_sub = X_test_q
    X_test_c_sub = X_test_classical
    y_test_sub = y_test_torch

train_data = (X_train_q_sub, X_train_c_sub, y_train_sub)
test_data = (X_test_q_sub, X_test_c_sub, y_test_sub)

# Reduced quantum inputs
Xq_train_red = X_train_q_sub[:, :3]
Xq_test_red = X_test_q_sub[:, :3]

# 7. Train main hybrid model
print("\n" + "=" * 60)
print("  TRAINING MAIN HYBRID MODEL")
print("=" * 60)

set_determinism(42)
hybrid_model = HybridRealQ(n_classical)
hybrid_history = train_model(hybrid_model, train_data, test_data)
hybrid_metrics = evaluate_model(hybrid_model, X_test_q_sub, X_test_c_sub, y_test_sub.numpy())
best_thresh = hybrid_metrics['threshold']

print(f"\n  Main Model Results:")
print(f"    AUC:       {hybrid_metrics['auc']:.4f}")
print(f"    Precision: {hybrid_metrics['precision']:.4f}")
print(f"    Recall:    {hybrid_metrics['recall']:.4f}")
print(f"    F1:        {hybrid_metrics['f1']:.4f}")
print(f"    Threshold: {hybrid_metrics['threshold']:.4f}")

# 8. Ablation Studies
print("\n" + "=" * 80)
print("  ABLATION STUDIES")
print("=" * 80)

def run_trial(name, model, tr_data, te_data):
    set_determinism(42)
    gc.collect()
    start = time.time()
    try:
        train_model(model, tr_data, te_data, verbose=False)
        metrics = evaluate_model(model, te_data[0], te_data[1], te_data[2].numpy())
        elapsed = time.time() - start
        print(f"  {name}: AUC={metrics['auc']:.4f} Prec={metrics['precision']:.4f} Rec={metrics['recall']:.4f} F1={metrics['f1']:.4f} Time={elapsed:.1f}s")
        return {'Model': name, 'AUC': round(metrics['auc'], 4),
                'Precision': round(metrics['precision'], 4),
                'Recall': round(metrics['recall'], 4),
                'F1': round(metrics['f1'], 4),
                'Time_sec': round(elapsed, 1), 'Status': 'OK'}
    except Exception as e:
        elapsed = time.time() - start
        print(f"  {name}: FAILED ({e})")
        return {'Model': name, 'AUC': None, 'Precision': None, 'Recall': None,
                'F1': None, 'Time_sec': round(elapsed, 1), 'Status': f'FAILED: {e}'}

ablation = []

# Re-add main model results
ablation.append({'Model': 'HybridRealQ (7-qubit)', 'AUC': round(hybrid_metrics['auc'], 4),
                 'Precision': round(hybrid_metrics['precision'], 4),
                 'Recall': round(hybrid_metrics['recall'], 4),
                 'F1': round(hybrid_metrics['f1'], 4),
                 'Time_sec': round(hybrid_history['total_time'], 1), 'Status': 'OK'})

print("\nRunning Classical MLP...")
ablation.append(run_trial("Classical MLP", ClassicalMLP(n_classical), train_data, test_data))

print("Running Quantum Only...")
ablation.append(run_trial("Quantum Only", QuantumOnly(n_qubits), train_data, test_data))

print("Running Hybrid (3-qubit)...")
tr_red = (Xq_train_red, X_train_c_sub, y_train_sub)
te_red = (Xq_test_red, X_test_c_sub, y_test_sub)
ablation.append(run_trial("Hybrid (3-qubit)", HybridReducedQ(n_classical), tr_red, te_red))

print("Running Hybrid (Deep 2-layer)...")
ablation.append(run_trial("Hybrid (Deep 2-layer)", HybridDeepQ(n_classical), train_data, test_data))

# 8b. Improved Hybrid v2
print("\nRunning HybridRealQ_v2 (improved)...")
set_determinism(42)
gc.collect()
v2_model = HybridRealQ_v2(n_classical)
start_v2 = time.time()
v2_history = train_model(v2_model, train_data, test_data,
                         epochs=75, use_scheduler=True, time_budget=570)
v2_metrics = evaluate_model(v2_model, X_test_q_sub, X_test_c_sub, y_test_sub.numpy())
v2_time = time.time() - start_v2
print(f"  HybridRealQ_v2: AUC={v2_metrics['auc']:.4f} Prec={v2_metrics['precision']:.4f} "
      f"Rec={v2_metrics['recall']:.4f} F1={v2_metrics['f1']:.4f} Time={v2_time:.1f}s")
ablation.append({'Model': 'HybridRealQ_v2 (improved)', 'AUC': round(v2_metrics['auc'], 4),
                 'Precision': round(v2_metrics['precision'], 4),
                 'Recall': round(v2_metrics['recall'], 4),
                 'F1': round(v2_metrics['f1'], 4),
                 'Time_sec': round(v2_time, 1), 'Status': 'OK'})

# 8c. Fairness-aware v2
print("\nRunning HybridRealQ_v2 (fairness-aware)...")
race_cols_in_classical = [c for c in classical_features if c.startswith('race_')]
race_col_indices = [classical_features.index(c) for c in race_cols_in_classical]
race_labels_train = torch.tensor(
    X_train_c_sub.numpy()[:, race_col_indices].argmax(axis=1), dtype=torch.long)

set_determinism(42)
gc.collect()
v2_fair_model = HybridRealQ_v2(n_classical)
fair_criterion = nn.BCEWithLogitsLoss()
fair_optimizer = torch.optim.Adam(v2_fair_model.parameters(), lr=0.001)
fair_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    fair_optimizer, T_max=75, eta_min=1e-5)
fairness_weight = 0.1
n_race_groups = len(race_cols_in_classical)

hw_info = log_hardware_info()
print(f"  Hardware: {hw_info['device']} | Fairness weight: {fairness_weight}")
fair_start = time.time()
for epoch in range(75):
    if (time.time() - fair_start) > 570:
        print(f"  Time budget reached at epoch {epoch}. Stopping.")
        break
    epoch_start = time.time()
    v2_fair_model.train()
    fair_optimizer.zero_grad()
    logits = v2_fair_model(X_train_q_sub, X_train_c_sub).squeeze()
    bce_loss = fair_criterion(logits, y_train_sub.squeeze())
    # Demographic parity regularization: penalize variance of group-mean predictions
    group_means = []
    for g in range(n_race_groups):
        mask = (race_labels_train == g)
        if mask.sum() > 0:
            group_means.append(torch.sigmoid(logits[mask]).mean())
    if len(group_means) > 1:
        group_means_t = torch.stack(group_means)
        fair_penalty = group_means_t.var()
    else:
        fair_penalty = torch.tensor(0.0)
    loss = bce_loss + fairness_weight * fair_penalty
    loss.backward()
    fair_optimizer.step()
    fair_scheduler.step()
    if (epoch + 1) % 10 == 0:
        epoch_time = time.time() - epoch_start
        print(f"  Epoch {epoch+1:02d} | Loss: {loss.item():.4f} | BCE: {bce_loss.item():.4f} | "
              f"Fair: {fair_penalty.item():.4f} | Time: {epoch_time:.1f}s")

fair_total = time.time() - fair_start
print(f"  Total: {fair_total:.1f}s ({fair_total/60:.1f}m)")
v2_fair_metrics = evaluate_model(v2_fair_model, X_test_q_sub, X_test_c_sub, y_test_sub.numpy())
print(f"  HybridRealQ_v2 (fair): AUC={v2_fair_metrics['auc']:.4f} Prec={v2_fair_metrics['precision']:.4f} "
      f"Rec={v2_fair_metrics['recall']:.4f} F1={v2_fair_metrics['f1']:.4f}")
ablation.append({'Model': 'HybridRealQ_v2 (fair)', 'AUC': round(v2_fair_metrics['auc'], 4),
                 'Precision': round(v2_fair_metrics['precision'], 4),
                 'Recall': round(v2_fair_metrics['recall'], 4),
                 'F1': round(v2_fair_metrics['f1'], 4),
                 'Time_sec': round(fair_total, 1), 'Status': 'OK'})

# 9. Gradient Boosting Baselines
print("\n" + "=" * 60)
print("  GRADIENT BOOSTING BASELINES")
print("=" * 60)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

X_train_combined = np.hstack([X_train_q_sub.numpy() / np.pi, X_train_c_sub.numpy()])
X_test_combined = np.hstack([X_test_q_sub.numpy() / np.pi, X_test_c_sub.numpy()])
y_train_np = y_train_sub.numpy().ravel()
y_test_np = y_test_sub.numpy().ravel()

for ModelClass, name, kwargs in [
    (XGBClassifier, "XGBoost", dict(n_estimators=300, max_depth=6, learning_rate=0.1,
                                     subsample=0.8, colsample_bytree=0.8,
                                     eval_metric='logloss', random_state=42,
                                     use_label_encoder=False)),
    (LGBMClassifier, "LightGBM", dict(n_estimators=300, max_depth=6, learning_rate=0.1,
                                       subsample=0.8, colsample_bytree=0.8,
                                       random_state=42, verbose=-1)),
]:
    start = time.time()
    model = ModelClass(**kwargs)
    model.fit(X_train_combined, y_train_np)
    elapsed = time.time() - start
    probs = model.predict_proba(X_test_combined)[:, 1]
    auc = roc_auc_score(y_test_np, probs)
    fpr, tpr, thresholds = roc_curve(y_test_np, probs)
    j = tpr - fpr
    best_t = thresholds[j.argmax()]
    preds = (probs >= best_t).astype(int)
    prec = precision_score(y_test_np, preds, zero_division=0)
    rec = recall_score(y_test_np, preds, zero_division=0)
    f1_val = f1_score(y_test_np, preds, zero_division=0)
    print(f"  {name}: AUC={auc:.4f} Prec={prec:.4f} Rec={rec:.4f} F1={f1_val:.4f} Time={elapsed:.1f}s")
    ablation.append({'Model': name, 'AUC': round(auc, 4), 'Precision': round(prec, 4),
                     'Recall': round(rec, 4), 'F1': round(f1_val, 4),
                     'Time_sec': round(elapsed, 1), 'Status': 'OK'})

# 10. Fairness Audit
print("\n" + "=" * 75)
print("  FAIRNESS AUDIT")
print("=" * 75)

hybrid_model.eval()
with torch.no_grad():
    logits = hybrid_model(X_test_q_sub, X_test_c_sub).squeeze()
    probs_fair = torch.sigmoid(logits).cpu().numpy()

y_fair = y_test_sub.cpu().numpy().ravel()
preds_fair = (probs_fair >= best_thresh).astype(int)

audit_df = pd.DataFrame(X_test_c_sub.cpu().numpy(), columns=classical_features)
audit_df['target'] = y_fair
audit_df['prediction'] = preds_fair
audit_df['prob'] = probs_fair

race_map = {
    'race_0': 'White', 'race_1': 'Black',
    'race_2': 'American Indian/Alaska Native',
    'race_3': 'Asian/Pacific Islander',
    'race_4': 'Hispanic/Other'
}
# Find actual race columns
race_cols = [c for c in classical_features if c.startswith('race_')]
print(f"  Race columns found: {race_cols}")

# Map race columns to labels
actual_race_map = {}
for col in race_cols:
    # Try to match with known race map
    for key, label in race_map.items():
        if col == key or col.endswith(key.split('_')[-1]):
            actual_race_map[col] = label
            break
    else:
        actual_race_map[col] = col

print(f"  Race mapping: {actual_race_map}")

MIN_SUBGROUP = 20
WARN_SUBGROUP = 50

print(f"\n  {'Subgroup':<35} {'N':>5} {'Acc':>7} {'AUC':>7} {'TPR':>7} {'FPR':>7} {'PosRate':>8}")
print("  " + "-" * 75)

fairness_results = {}
for col, label in actual_race_map.items():
    if col not in audit_df.columns:
        continue
    subgroup = audit_df[audit_df[col] > 0.5]
    n = len(subgroup)
    if n < MIN_SUBGROUP:
        print(f"  SKIP {label}: n={n}")
        continue
    warn = " *" if n < WARN_SUBGROUP else ""
    y_sub = subgroup['target'].values
    p_sub = subgroup['prediction'].values
    prob_sub = subgroup['prob'].values
    acc = accuracy_score(y_sub, p_sub)
    pos_rate = p_sub.mean()
    cm = confusion_matrix(y_sub, p_sub, labels=[0, 1])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        tpr = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
        fpr_val = fp / (fp + tn) if (fp + tn) > 0 else float('nan')
    else:
        tpr = fpr_val = float('nan')
    try:
        sub_auc = roc_auc_score(y_sub, prob_sub)
    except ValueError:
        sub_auc = float('nan')
    fairness_results[label] = {'n': n, 'accuracy': acc, 'pos_rate': pos_rate,
                                'tpr': tpr, 'fpr': fpr_val, 'auc': sub_auc}
    print(f"  {label + warn:<35} {n:>5} {acc:>7.4f} {sub_auc:>7.4f} {tpr:>7.4f} {fpr_val:>7.4f} {pos_rate:>8.4f}")

# Gaps
pos_rates = [v['pos_rate'] for v in fairness_results.values()]
tprs = [v['tpr'] for v in fairness_results.values() if not np.isnan(v['tpr'])]
fprs = [v['fpr'] for v in fairness_results.values() if not np.isnan(v['fpr'])]
dp_diff = max(pos_rates) - min(pos_rates) if len(pos_rates) >= 2 else float('nan')
eo_tpr = max(tprs) - min(tprs) if len(tprs) >= 2 else float('nan')
eo_fpr = max(fprs) - min(fprs) if len(fprs) >= 2 else float('nan')

print(f"\n  Demographic Parity Difference:     {dp_diff:.4f}")
print(f"  Equal Opportunity Diff (TPR gap):  {eo_tpr:.4f}")
print(f"  FPR Parity Difference:             {eo_fpr:.4f}")

# 10b. Fairness comparison: v2 vs v2_fair
def quick_fairness_audit(model_to_audit, model_name, threshold_override=None):
    """Run fairness audit on a model and return gap metrics."""
    model_to_audit.eval()
    with torch.no_grad():
        lgt = model_to_audit(X_test_q_sub, X_test_c_sub).squeeze()
        prb = torch.sigmoid(lgt).cpu().numpy()
    y_np = y_test_sub.cpu().numpy().ravel()
    if threshold_override is None:
        fpr_a, tpr_a, thr_a = roc_curve(y_np, prb)
        threshold_override = thr_a[(tpr_a - fpr_a).argmax()]
    prd = (prb >= threshold_override).astype(int)
    adf = pd.DataFrame(X_test_c_sub.cpu().numpy(), columns=classical_features)
    adf['target'] = y_np
    adf['prediction'] = prd
    adf['prob'] = prb
    print(f"\n  --- {model_name} (threshold={threshold_override:.4f}) ---")
    res = {}
    for col in race_cols:
        label = actual_race_map.get(col, col)
        sg = adf[adf[col] > 0.5]
        if len(sg) < MIN_SUBGROUP:
            continue
        y_s, p_s, pr_s = sg['target'].values, sg['prediction'].values, sg['prob'].values
        acc = accuracy_score(y_s, p_s)
        cm = confusion_matrix(y_s, p_s, labels=[0, 1])
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            t = tp / (tp + fn) if (tp + fn) > 0 else float('nan')
            f = fp / (fp + tn) if (fp + tn) > 0 else float('nan')
        else:
            t = f = float('nan')
        try:
            sa = roc_auc_score(y_s, pr_s)
        except ValueError:
            sa = float('nan')
        res[label] = {'n': len(sg), 'tpr': t, 'fpr': f, 'pos_rate': p_s.mean(), 'auc': sa}
        print(f"    {label:<35} N={len(sg):>4} AUC={sa:.4f} TPR={t:.4f} PosRate={p_s.mean():.4f}")
    pr = [v['pos_rate'] for v in res.values()]
    tp = [v['tpr'] for v in res.values() if not np.isnan(v['tpr'])]
    fp = [v['fpr'] for v in res.values() if not np.isnan(v['fpr'])]
    gaps = {
        'dp_diff': max(pr) - min(pr) if len(pr) >= 2 else float('nan'),
        'eo_tpr_diff': max(tp) - min(tp) if len(tp) >= 2 else float('nan'),
        'eo_fpr_diff': max(fp) - min(fp) if len(fp) >= 2 else float('nan'),
    }
    print(f"    DP Diff: {gaps['dp_diff']:.4f} | TPR Gap: {gaps['eo_tpr_diff']:.4f} | FPR Gap: {gaps['eo_fpr_diff']:.4f}")
    return res, gaps

v2_fairness, v2_gaps = quick_fairness_audit(v2_model, "HybridRealQ_v2")
v2f_fairness, v2f_gaps = quick_fairness_audit(v2_fair_model, "HybridRealQ_v2 (fair)")

# 11. Save All Results
print("\n" + "=" * 60)
print("  SAVING RESULTS")
print("=" * 60)

ablation_df = pd.DataFrame(ablation)
print("\nAblation Results:")
print(ablation_df.to_string(index=False))
ablation_df.to_csv('TNBC_Quantum_Ablation_Results.csv', index=False)

results = {
    'main_model': {
        'auc': round(hybrid_metrics['auc'], 4),
        'precision': round(hybrid_metrics['precision'], 4),
        'recall': round(hybrid_metrics['recall'], 4),
        'f1': round(hybrid_metrics['f1'], 4),
        'threshold': round(hybrid_metrics['threshold'], 4),
        'total_training_time_sec': round(hybrid_history['total_time'], 1),
    },
    'ablation': ablation,
    'fairness': {k: {kk: round(vv, 4) if isinstance(vv, float) else vv
                      for kk, vv in v.items()} for k, v in fairness_results.items()},
    'fairness_gaps': {
        'dp_diff': round(dp_diff, 4),
        'eo_tpr_diff': round(eo_tpr, 4),
        'eo_fpr_diff': round(eo_fpr, 4),
    },
    'leakage_audit': {
        'auc_flags': len(flagged_auc),
        'corr_flags': len(flagged_corr),
        'ks_flags': len(flagged_ks),
        'all_passed': n_flags == 0,
    },
    'data': {
        'total_rows': len(df_clean),
        'train_size': len(X_train_q_sub),
        'test_size': len(X_test_q_sub),
        'n_quantum_features': len(quantum_features),
        'n_classical_features': n_classical,
    },
    'hardware': hybrid_history.get('hardware', {}),
}

with open('experiment_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print("\nSaved: TNBC_Quantum_Ablation_Results.csv")
print("Saved: experiment_results.json")
print("\nDone!")
