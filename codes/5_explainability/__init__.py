"""Phase 5: Explainability layer for TNBC survival model.

Modules:
  - integrated_gradients: Per-feature attributions via path integration
  - attention: Temporal attention weight extraction from transformer
  - feature_importance: Permutation-based feature importance (model-agnostic)
  - report: Per-patient explanation report (JSON output)
"""
