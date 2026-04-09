"""SHAP explainability for TNBC survival model.

Data-lineage:
  Trained model + feature matrix
    -> SHAP values per feature per patient
    -> Global feature importance rankings
    -> Per-subgroup SHAP value comparison (disparity-aware)

Uses SHAP's KernelExplainer (model-agnostic) for the tabular SEER tower
and GradientExplainer for differentiable neural network towers.

Reference:
  Lundberg & Lee (2017) "A Unified Approach to Interpreting Model
    Predictions" (NIPS)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import shap
import torch
import torch.nn as nn


@dataclass
class SHAPExplanation:
    """Container for SHAP analysis results."""
    shap_values: np.ndarray          # (N, d_features) SHAP values
    feature_names: list[str]         # column names
    base_value: float                # expected model output (mean prediction)
    global_importance: list[tuple[str, float]]  # sorted by |mean SHAP|
    X: np.ndarray                    # (N, d_features) input data


def explain_seer_tower(
    model: nn.Module,
    X: np.ndarray,
    feature_names: list[str],
    background_size: int = 100,
    n_samples: int = 200,
    input_key: str = "x_seer",
) -> SHAPExplanation:
    """Compute SHAP values for the SEER tabular tower.

    Uses KernelExplainer (model-agnostic) to handle the full model
    pipeline including batch normalization and dropout.

    Parameters
    ----------
    model : nn.Module
        Trained survival model accepting ``x_seer`` keyword.
    X : (N, d_features)
        Feature matrix to explain.
    feature_names : list
        Names for each feature column.
    background_size : int
        Number of background samples for KernelExplainer.
    n_samples : int
        Number of perturbation samples per explanation.
    input_key : str
        Keyword argument name for tabular input.

    Returns
    -------
    SHAPExplanation with per-feature SHAP values.
    """
    model.eval()
    device = next(model.parameters()).device

    # Model wrapper: numpy in -> numpy out
    def predict_fn(x_np: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x_t = torch.from_numpy(x_np.astype(np.float32)).to(device)
            kwargs = {input_key: x_t}
            return model(**kwargs).cpu().numpy()

    # Background data (subsample for efficiency)
    rng = np.random.default_rng(42)
    bg_idx = rng.choice(len(X), size=min(background_size, len(X)), replace=False)
    background = X[bg_idx]

    # Build SHAP explainer
    explainer = shap.KernelExplainer(predict_fn, background)

    # Compute SHAP values
    shap_values = explainer.shap_values(X, nsamples=n_samples, silent=True)

    # Global importance: mean |SHAP| per feature
    mean_abs = np.abs(shap_values).mean(axis=0)
    importance = sorted(
        zip(feature_names, mean_abs.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )

    return SHAPExplanation(
        shap_values=shap_values,
        feature_names=feature_names,
        base_value=float(explainer.expected_value),
        global_importance=importance,
        X=X,
    )


def explain_gradient(
    model: nn.Module,
    X: torch.Tensor,
    feature_names: list[str],
    background: torch.Tensor | None = None,
    input_key: str = "x_seer",
) -> SHAPExplanation:
    """Compute GradientSHAP values for differentiable models.

    Faster than KernelExplainer but requires a differentiable model.

    Parameters
    ----------
    model : nn.Module
        Differentiable model.
    X : (N, d_features) tensor to explain.
    feature_names : feature names.
    background : (M, d_features) background distribution.
    input_key : keyword for model forward.

    Returns
    -------
    SHAPExplanation with per-feature SHAP values.
    """
    model.eval()
    device = next(model.parameters()).device

    if background is None:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X), size=min(50, len(X)), replace=False)
        background = X[idx]

    background = background.to(device)
    X = X.to(device)

    # Wrapper model for SHAP
    class _Wrapper(nn.Module):
        def __init__(self, inner, key):
            super().__init__()
            self.inner = inner
            self.key = key

        def forward(self, x):
            return self.inner(**{self.key: x}).unsqueeze(-1)

    wrapper = _Wrapper(model, input_key)

    explainer = shap.GradientExplainer(wrapper, background)
    sv = explainer.shap_values(X)

    if isinstance(sv, list):
        sv = sv[0]
    if isinstance(sv, torch.Tensor):
        sv = sv.cpu().numpy()

    # Squeeze trailing dimension if present
    if sv.ndim == 3 and sv.shape[-1] == 1:
        sv = sv.squeeze(-1)

    mean_abs = np.abs(sv).mean(axis=0)
    importance = sorted(
        zip(feature_names, mean_abs.tolist()),
        key=lambda x: x[1],
        reverse=True,
    )

    with torch.no_grad():
        base = model(**{input_key: background}).mean().item()

    return SHAPExplanation(
        shap_values=sv,
        feature_names=feature_names,
        base_value=base,
        global_importance=importance,
        X=X.cpu().numpy(),
    )


def shap_subgroup_comparison(
    explanation: SHAPExplanation,
    group_labels: np.ndarray,
    group_names: dict[int, str] | None = None,
) -> dict[str, dict[str, float]]:
    """Compare SHAP feature importance across demographic subgroups.

    Reveals whether the model relies on different features for
    different racial/demographic groups -- a key fairness insight.

    Parameters
    ----------
    explanation : SHAPExplanation
        SHAP values for the full dataset.
    group_labels : (N,) integer group codes.
    group_names : dict mapping code -> name.

    Returns
    -------
    Dict[group_name, Dict[feature_name, mean_abs_shap]].
    """
    if group_names is None:
        group_names = {0: "White", 1: "Black", 2: "Asian",
                       3: "Hispanic", 4: "Other"}

    result: dict[str, dict[str, float]] = {}
    sv = explanation.shap_values
    fnames = explanation.feature_names

    for code, name in group_names.items():
        mask = group_labels == code
        if mask.sum() < 5:
            continue
        group_sv = sv[mask]
        mean_abs = np.abs(group_sv).mean(axis=0)
        result[name] = {fn: float(v) for fn, v in zip(fnames, mean_abs)}

    return result


def print_shap_summary(explanation: SHAPExplanation, top_k: int = 10) -> None:
    """Print global SHAP feature importance."""
    print("=== SHAP Feature Importance ===\n")
    print(f"Base value (expected output): {explanation.base_value:.4f}")
    print(f"Samples explained: {explanation.shap_values.shape[0]}\n")
    print(f"Top {top_k} features by mean |SHAP|:")
    for i, (name, val) in enumerate(explanation.global_importance[:top_k]):
        bar = "#" * int(val / explanation.global_importance[0][1] * 30)
        print(f"  {i+1:2d}. {name:30s} {val:.4f}  {bar}")
