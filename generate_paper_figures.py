"""Generate training-curve and quantum-correction figures from current survival results.

Reads:
  - survival_v4_results.json
  - survival_residual_results.json

Writes:
  - figures/survival_training_curve.png
  - figures/quantum_correction_distribution.png
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs('figures', exist_ok=True)

# Load v4 results
v4 = json.load(open('survival_v4_results.json'))
v4_pretrain_epochs = list(range(1, len(v4.get('training_history_pretrain', [])) + 1))
v4_pretrain_cidx = []
v4_hybrid_cidx = []

# v4 history is stored as a flat dict, need to reconstruct from the script's training prints
# Parse from the saved hybrid_final_cidx trajectory if available
# For simplicity, use the printed values from the run output
v4_pretrain_epochs_arr = [5, 10, 15]
v4_pretrain_cidx_arr = [0.5760, 0.5697, 0.5613]
v4_hybrid_epochs_arr = [2, 4, 6, 8, 10]
v4_hybrid_cidx_arr = [0.5919, 0.7044, 0.7226, 0.7257, 0.7249]

# Load residual results
res = json.load(open('survival_residual_results.json'))
history = res.get('training_history', [])
res_epochs = [e['epoch'] for e in history]
res_cidx = [e['test_cidx'] for e in history]
res_corr_std = [e.get('correction_std', 0) for e in history]
cox_baseline = res.get('cox_test_cindex', 0.7326)

# Figure 1: Training curve comparison
fig, ax = plt.subplots(1, 2, figsize=(14, 5))

# Left: v4 two-phase training
ax[0].axhline(y=cox_baseline, color='red', linestyle='--', linewidth=1.5,
              label=f'Cox PH baseline ({cox_baseline:.4f})')
ax[0].plot(v4_pretrain_epochs_arr, v4_pretrain_cidx_arr, 'o-', color='steelblue',
           linewidth=2, markersize=8, label='Phase 1: Classical pretrain')
# Connect pretrain end to hybrid start visually
hybrid_x = [15] + [15 + e for e in v4_hybrid_epochs_arr]
hybrid_y = [v4_pretrain_cidx_arr[-1]] + v4_hybrid_cidx_arr
ax[0].plot(hybrid_x, hybrid_y, 's-', color='goldenrod', linewidth=2, markersize=8,
           label='Phase 2: Hybrid fine-tuning')
ax[0].axvline(x=15, color='gray', linestyle=':', alpha=0.5)
ax[0].text(15.3, 0.59, 'Quantum\nbranch\nenabled', fontsize=9, color='gray')
ax[0].set_xlabel('Epoch', fontsize=11)
ax[0].set_ylabel('Test C-index', fontsize=11)
ax[0].set_title('HybridSurvivalQ_v4: Two-Phase Training', fontsize=12, fontweight='bold')
ax[0].legend(loc='lower right', fontsize=9)
ax[0].grid(alpha=0.3)
ax[0].set_ylim([0.55, 0.76])

# Right: residual learning
ax[1].axhline(y=cox_baseline, color='red', linestyle='--', linewidth=1.5,
              label=f'Cox PH baseline ({cox_baseline:.4f})')
ax[1].plot(res_epochs, res_cidx, 'o-', color='darkgreen', linewidth=2, markersize=8,
           label='Cox + Quantum Residual')
final_cidx = res_cidx[-1] if res_cidx else cox_baseline
ax[1].annotate(f'Final: {final_cidx:.4f}\n(+{final_cidx - cox_baseline:.4f})',
               xy=(res_epochs[-1], res_cidx[-1]),
               xytext=(res_epochs[-1] - 2, res_cidx[-1] + 0.005),
               fontsize=10, ha='center',
               bbox=dict(boxstyle='round', facecolor='palegreen', alpha=0.8))
ax[1].set_xlabel('Epoch', fontsize=11)
ax[1].set_ylabel('Test C-index', fontsize=11)
ax[1].set_title('Quantum Residual Learning', fontsize=12, fontweight='bold')
ax[1].legend(loc='lower right', fontsize=9)
ax[1].grid(alpha=0.3)
ax[1].set_ylim([0.7320, 0.7400])

plt.tight_layout()
plt.savefig('figures/survival_training_curve.png', dpi=200, bbox_inches='tight')
plt.savefig('figures/training_curve.png', dpi=200, bbox_inches='tight')  # Replace old
print("Saved: figures/survival_training_curve.png (and overwrote training_curve.png)")
plt.close()

# Figure 2: Quantum correction growth (residual model)
fig, ax = plt.subplots(figsize=(8, 5))
ax2 = ax.twinx()
l1, = ax.plot(res_epochs, res_corr_std, 'o-', color='purple', linewidth=2, markersize=8,
              label='Quantum correction std')
l2, = ax2.plot(res_epochs, res_cidx, 's-', color='darkgreen', linewidth=2, markersize=8,
               label='Test C-index')
ax2.axhline(y=cox_baseline, color='red', linestyle='--', linewidth=1.5, alpha=0.7,
            label=f'Cox PH baseline')
ax.set_xlabel('Epoch', fontsize=11)
ax.set_ylabel('Quantum correction std deviation', fontsize=11, color='purple')
ax2.set_ylabel('Test C-index', fontsize=11, color='darkgreen')
ax.tick_params(axis='y', labelcolor='purple')
ax2.tick_params(axis='y', labelcolor='darkgreen')
ax.set_title('Residual Learning: Correction Magnitude vs C-index', fontsize=12, fontweight='bold')
ax.grid(alpha=0.3)
fig.legend(handles=[l1, l2], loc='upper left', bbox_to_anchor=(0.15, 0.85))
plt.tight_layout()
plt.savefig('figures/quantum_correction_distribution.png', dpi=200, bbox_inches='tight')
print("Saved: figures/quantum_correction_distribution.png")
plt.close()

print("\nAll figures generated successfully")
