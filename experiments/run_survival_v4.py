"""
HybridSurvivalQ_v4: Quantum ML best practices for survival analysis.

Key improvements over v3:
1. Output scaling layer: learnable scale + bias allows log-hazards in [-5, +5]
   instead of being clamped near [-1, +1] from Pauli expectations.
2. Smaller quantum init (* 0.05): mitigates barren plateaus per Grant et al. 2019.
3. Gradient clipping (max_norm=1.0): stabilizes Cox loss optimization.
4. Two-phase training: classical pre-training, then hybrid fine-tuning.
   Gives the model a non-trivial starting point before quantum gradients arrive.
5. Higher learning rate for Cox (0.005): partial likelihood gradients are small.
6. Mini-batch Cox loss on full dataset (16k patients, not 2k subsample).
"""
import os
import sys
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
os.environ['PYTHONUNBUFFERED'] = '1'
sys.stdout.reconfigure(line_buffering=True)

import time
import json
import gc
import platform
import random
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import pennylane as qml

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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Running on: {device}")


# 1. Data preprocessing (matches run_survival_experiments.py exactly)
print("\n" + "=" * 60)
print("  DATA LOADING & PREPROCESSING")
print("=" * 60)

df = pd.read_csv('breast_cancer_4quantum.csv')
durations_all = pd.to_numeric(df['Survival_months'], errors='coerce')
events_all = (df['Vital_status_recode_study_cutoff_used_'].astype(str).str.lower() == 'dead').astype(float)
mask = ~durations_all.isna() & (durations_all > 0)
df = df[mask].copy()
df['_duration'] = durations_all[mask].values
df['_event'] = events_all[mask].values
print(f"After duration/event extraction: {len(df)} patients, event rate {df['_event'].mean():.2%}")

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

print(f"Clean cohort: {len(df_clean)}, n_classical={len(classical_features)}")


# 2. Train/test split
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

# v4 uses [pi/4, 3pi/4] encoding range to avoid cos/sin saturation
encoding_low = np.pi / 4
encoding_range = np.pi / 2
X_train_q = torch.tensor(encoding_low + encoding_range * Xq_train_scaled, dtype=torch.float32)
X_test_q = torch.tensor(encoding_low + encoding_range * Xq_test_scaled, dtype=torch.float32)
X_train_c = torch.tensor(Xc_train_scaled, dtype=torch.float32)
X_test_c = torch.tensor(Xc_test_scaled, dtype=torch.float32)
durations_train = torch.tensor(df_train['_duration'].values, dtype=torch.float32)
events_train = torch.tensor(df_train['_event'].values, dtype=torch.float32)
durations_test = torch.tensor(df_test['_duration'].values, dtype=torch.float32)
events_test = torch.tensor(df_test['_event'].values, dtype=torch.float32)
n_classical = X_train_c.shape[1]

print(f"Train: {len(X_train_q)}, Test: {len(X_test_q)}, n_classical: {n_classical}")
print(f"Encoding range: [pi/4, 3pi/4] (avoids rotation saturation)")


# 3. Quantum circuit (same as v3)
n_qubits = 7
try:
    dev = qml.device("lightning.qubit", wires=n_qubits)
    print("Using lightning.qubit")
except Exception:
    dev = qml.device("default.qubit", wires=n_qubits)


@qml.qnode(dev, interface="torch")
def quantum_circuit_v4(inputs, input_scales, input_biases, weights_ry, weights_rz):
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


class HybridSurvivalQ_v4(nn.Module):
    """v4: All v3 features + quantum ML best practices for Cox loss training."""
    def __init__(self, n_classical_features):
        super().__init__()
        self.input_scales = nn.Parameter(torch.ones(n_qubits))
        self.input_biases = nn.Parameter(torch.zeros(n_qubits))
        # SMALLER init for quantum params (Grant et al. 2019)
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
        # NEW: learnable output scale and bias for log-hazard range
        self.output_scale = nn.Parameter(torch.tensor(3.0))
        self.output_bias = nn.Parameter(torch.tensor(0.0))
        # Phase flag
        self.use_quantum = True

    def forward(self, x_q, x_c):
        x_q = x_q.to(torch.float32)
        x_c = x_c.to(torch.float32)
        if self.use_quantum:
            q_out = []
            for i in range(x_q.shape[0]):
                res = quantum_circuit_v4(x_q[i], self.input_scales, self.input_biases,
                                         self.q_params_ry, self.q_params_rz)
                q_out.append(torch.stack(list(res)))
            q_out = torch.stack(q_out).to(torch.float32)
        else:
            # Pre-training: zero quantum branch
            q_out = torch.zeros(x_q.shape[0], 14, dtype=torch.float32, device=x_q.device)
        c_encoded = self.classical_encoder(x_c)
        combined = torch.cat([q_out, c_encoded, x_c], dim=1)
        fused = self.fusion(combined).squeeze(-1)
        # Learnable output scaling for log-hazard range
        return self.output_scale * fused + self.output_bias


# 4. Cox loss
def cox_ph_loss(log_hazards, durations, events):
    idx = torch.argsort(durations, descending=True)
    h = log_hazards[idx]
    e = events[idx]
    log_cumsum = torch.logcumsumexp(h, dim=0)
    n_events = e.sum()
    if n_events == 0:
        return torch.tensor(0.0, requires_grad=True)
    return -((h - log_cumsum) * e).sum() / n_events


# 5. Two-phase training
def train_v4(model, X_q_tr, X_c_tr, dur_tr, evt_tr, X_q_te, X_c_te, dur_te, evt_te,
             pretrain_epochs=15, hybrid_epochs=15, lr_pretrain=0.005, lr_hybrid=0.001,
             batch_size=256, time_budget=900, verbose=True):
    n = len(X_q_tr)
    history = {'pretrain': [], 'hybrid': []}

    # PHASE 1: Classical pre-training
    if verbose:
        print(f"\n  PHASE 1: Classical pre-training ({pretrain_epochs} epochs, lr={lr_pretrain})")
    model.use_quantum = False
    optimizer = torch.optim.Adam(
        list(model.classical_encoder.parameters()) +
        list(model.fusion.parameters()) +
        [model.output_scale, model.output_bias],
        lr=lr_pretrain
    )
    pretrain_start = time.time()
    for epoch in range(pretrain_epochs):
        model.train()
        perm = torch.randperm(n)
        losses = []
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            if evt_tr[idx].sum() == 0:
                continue
            optimizer.zero_grad()
            log_h = model(X_q_tr[idx], X_c_tr[idx])
            loss = cox_ph_loss(log_h, dur_tr[idx], evt_tr[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(loss.item())
        avg_loss = float(np.mean(losses)) if losses else 0.0
        model.eval()
        with torch.no_grad():
            val_h = model(X_q_te, X_c_te).cpu().numpy()
            val_cidx = concordance_index(dur_te.cpu().numpy(), -val_h, evt_te.cpu().numpy())
        history['pretrain'].append({'epoch': epoch + 1, 'loss': avg_loss, 'val_cidx': val_cidx})
        if verbose and (epoch + 1) % 5 == 0:
            print(f"    Pretrain {epoch+1:02d} | Loss: {avg_loss:.4f} | Val C-idx: {val_cidx:.4f}")
    pretrain_time = time.time() - pretrain_start
    if verbose:
        print(f"    Phase 1 done in {pretrain_time:.1f}s, final Val C-idx: {history['pretrain'][-1]['val_cidx']:.4f}")

    # PHASE 2: Hybrid fine-tuning
    if verbose:
        print(f"\n  PHASE 2: Hybrid fine-tuning ({hybrid_epochs} epochs, lr={lr_hybrid})")
    model.use_quantum = True
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_hybrid)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=hybrid_epochs, eta_min=1e-5)
    hybrid_start = time.time()
    for epoch in range(hybrid_epochs):
        if (time.time() - hybrid_start) > time_budget:
            if verbose:
                print(f"    Time budget reached at epoch {epoch}")
            break
        epoch_start = time.time()
        model.train()
        perm = torch.randperm(n)
        losses = []
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            if evt_tr[idx].sum() == 0:
                continue
            optimizer.zero_grad()
            log_h = model(X_q_tr[idx], X_c_tr[idx])
            loss = cox_ph_loss(log_h, dur_tr[idx], evt_tr[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            losses.append(loss.item())
        scheduler.step()
        avg_loss = float(np.mean(losses)) if losses else 0.0
        model.eval()
        with torch.no_grad():
            val_h = model(X_q_te, X_c_te).cpu().numpy()
            val_cidx = concordance_index(dur_te.cpu().numpy(), -val_h, evt_te.cpu().numpy())
        et = time.time() - epoch_start
        history['hybrid'].append({'epoch': epoch + 1, 'loss': avg_loss, 'val_cidx': val_cidx, 'time': et})
        if verbose and (epoch + 1) % 2 == 0:
            print(f"    Hybrid {epoch+1:02d} | Loss: {avg_loss:.4f} | Val C-idx: {val_cidx:.4f} | Time: {et:.1f}s")
    hybrid_time = time.time() - hybrid_start
    if verbose:
        print(f"    Phase 2 done in {hybrid_time:.1f}s, final Val C-idx: {history['hybrid'][-1]['val_cidx']:.4f}")

    history['pretrain_time'] = pretrain_time
    history['hybrid_time'] = hybrid_time
    history['total_time'] = pretrain_time + hybrid_time
    return history


def evaluate_cox(model, X_q, X_c, durations, events):
    model.eval()
    with torch.no_grad():
        log_h = model(X_q, X_c).cpu().numpy()
    return {
        'c_index': concordance_index(durations.cpu().numpy(), -log_h, events.cpu().numpy()),
        'log_hazards': log_h,
    }


# 6. Run v4
print("\n" + "=" * 80)
print("  HybridSurvivalQ_v4 (Quantum ML best practices for Cox)")
print("=" * 80)

set_determinism(42)
v4_model = HybridSurvivalQ_v4(n_classical)
total_params = sum(p.numel() for p in v4_model.parameters())
print(f"Total parameters: {total_params}")
print(f"  Quantum params: {(3*n_qubits*2) + (2*n_qubits)}")  # RY+RZ + scale+bias
print(f"  Output scale: {v4_model.output_scale.item():.2f}, bias: {v4_model.output_bias.item():.2f}")

t0 = time.time()
history = train_v4(
    v4_model,
    X_train_q, X_train_c, durations_train, events_train,
    X_test_q, X_test_c, durations_test, events_test,
    pretrain_epochs=15, hybrid_epochs=15,
    lr_pretrain=0.005, lr_hybrid=0.001,
    batch_size=256, time_budget=1200
)
total_time = time.time() - t0

metrics = evaluate_cox(v4_model, X_test_q, X_test_c, durations_test, events_test)
print(f"\n  v4 Final Test C-index: {metrics['c_index']:.4f}")
print(f"  Total time: {total_time:.1f}s ({total_time/60:.1f}m)")
print(f"  Final output scale: {v4_model.output_scale.item():.2f}")
print(f"  Final output bias: {v4_model.output_bias.item():.2f}")
print(f"  Log-hazard range: [{metrics['log_hazards'].min():.2f}, {metrics['log_hazards'].max():.2f}]")
print(f"  Log-hazard std: {metrics['log_hazards'].std():.4f}")


# 7. Subgroup C-index audit
print("\n" + "=" * 75)
print("  v4 FAIRNESS AUDIT (Subgroup C-index)")
print("=" * 75)
race_cols = [c for c in classical_features if c.startswith('race_')]
race_idx_in_classical = [classical_features.index(c) for c in race_cols]
test_race_groups = X_test_c.cpu().numpy()[:, race_idx_in_classical].argmax(axis=1)

print(f"  {'Subgroup':<35} {'N':>5} {'Events':>7} {'C-index':>9}")
print("  " + "-" * 60)
v4_subgroup = {}
cindices = []
log_h = metrics['log_hazards']
for g, name in enumerate(race_cols):
    mask = test_race_groups == g
    n = mask.sum()
    n_evt = events_test[mask].sum().item() if n > 0 else 0
    if n < 20 or n_evt < 5:
        print(f"  SKIP {name}: N={n}, events={n_evt}")
        continue
    try:
        ci = concordance_index(durations_test[mask].cpu().numpy(),
                               -log_h[mask], events_test[mask].cpu().numpy())
        v4_subgroup[name] = {'n': int(n), 'events': int(n_evt), 'c_index': float(ci)}
        cindices.append(ci)
        print(f"  {name:<35} {n:>5} {int(n_evt):>7} {ci:>9.4f}")
    except Exception as e:
        print(f"  ERROR {name}: {e}")

if len(cindices) >= 2:
    v4_gap = max(cindices) - min(cindices)
    print(f"\n  Subgroup C-index gap: {v4_gap:.4f}")
else:
    v4_gap = float('nan')


# 8. Save
results = {
    'model': 'HybridSurvivalQ_v4',
    'innovations': [
        'output_scale_bias_layer',
        'small_quantum_init_0.05',
        'gradient_clipping_max_1.0',
        'classical_pretraining_then_hybrid',
        'higher_lr_for_pretrain',
        'mini_batch_cox_full_dataset',
        'encoding_range_pi_4_to_3pi_4',
    ],
    'test_c_index': round(metrics['c_index'], 4),
    'pretrain_final_cidx': round(history['pretrain'][-1]['val_cidx'], 4),
    'hybrid_final_cidx': round(history['hybrid'][-1]['val_cidx'], 4) if history['hybrid'] else None,
    'output_scale_final': round(v4_model.output_scale.item(), 2),
    'output_bias_final': round(v4_model.output_bias.item(), 2),
    'log_hazard_std': round(float(metrics['log_hazards'].std()), 4),
    'total_time_sec': round(total_time, 1),
    'pretrain_time_sec': round(history['pretrain_time'], 1),
    'hybrid_time_sec': round(history['hybrid_time'], 1),
    'fairness_subgroups': v4_subgroup,
    'fairness_gap': round(v4_gap, 4) if not np.isnan(v4_gap) else None,
    'data': {
        'train_size': len(X_train_q),
        'test_size': len(X_test_q),
        'n_quantum_features': len(quantum_features),
        'n_classical_features': n_classical,
    },
}
with open('results/survival_v4_results.json', 'w') as f:
    json.dump(results, f, indent=2, default=str)

print("\n" + "=" * 60)
print(f"  v4 RESULT: C-index = {metrics['c_index']:.4f}")
print(f"  Cox PH baseline: 0.7323")
print(f"  v3 (subsample): 0.5877")
print(f"  Saved: survival_v4_results.json")
print("=" * 60)
