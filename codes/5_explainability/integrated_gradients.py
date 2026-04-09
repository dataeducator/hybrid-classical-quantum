"""Integrated Gradients for per-feature attribution of survival predictions.

Data-lineage:
  Trained model + input features
    -> gradient path integration from baseline to input
    -> per-feature attribution scores (same shape as input)

Works for all towers (tabular, temporal, text) since it only requires
differentiable forward passes.

Reference:
  Sundararajan et al. (2017) "Axiomatic Attribution for Deep Networks"
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn


@dataclass
class Attribution:
    """Container for feature attributions."""
    values: np.ndarray          # attribution scores (same shape as input)
    feature_names: list[str]    # names for each feature dimension
    predicted_risk: float       # model output for this input
    baseline_risk: float        # model output for baseline


def integrated_gradients_tabular(
    model: nn.Module,
    x: torch.Tensor,
    feature_names: list[str],
    baseline: torch.Tensor | None = None,
    n_steps: int = 50,
    forward_kwargs: dict | None = None,
    input_key: str = "x_seer",
) -> Attribution:
    """Compute Integrated Gradients for a single tabular input.

    Parameters
    ----------
    model : nn.Module
        Survival model (must accept keyword arguments).
    x : (d_features,) single input tensor.
    feature_names : list of feature names matching x dimensions.
    baseline : (d_features,) baseline input. Defaults to zeros.
    n_steps : number of interpolation steps.
    forward_kwargs : additional keyword arguments for model forward.
    input_key : keyword argument name for the tabular input.

    Returns
    -------
    Attribution with per-feature scores.
    """
    if baseline is None:
        baseline = torch.zeros_like(x)

    if forward_kwargs is None:
        forward_kwargs = {}

    model.eval()
    x = x.detach().clone()
    baseline = baseline.detach().clone()

    # Generate interpolation path
    alphas = torch.linspace(0, 1, n_steps + 1, device=x.device)
    scaled_inputs = baseline.unsqueeze(0) + alphas.unsqueeze(1) * (x - baseline).unsqueeze(0)
    # scaled_inputs: (n_steps+1, d_features)

    # Compute gradients at each step
    gradients = []
    for i in range(n_steps + 1):
        inp = scaled_inputs[i].unsqueeze(0).requires_grad_(True)
        kwargs = {input_key: inp, **forward_kwargs}
        output = model(**kwargs)
        output.sum().backward()
        gradients.append(inp.grad.detach().squeeze(0))

    # Trapezoidal rule for integral
    grads = torch.stack(gradients)  # (n_steps+1, d_features)
    avg_grads = (grads[:-1] + grads[1:]).mean(dim=0) / 2.0

    # Attribution = (input - baseline) * avg_gradient
    attributions = (x - baseline) * avg_grads

    # Get predicted values
    with torch.no_grad():
        pred = model(**{input_key: x.unsqueeze(0), **forward_kwargs}).item()
        base = model(**{input_key: baseline.unsqueeze(0), **forward_kwargs}).item()

    return Attribution(
        values=attributions.cpu().numpy(),
        feature_names=feature_names,
        predicted_risk=pred,
        baseline_risk=base,
    )


def integrated_gradients_batch(
    model: nn.Module,
    X: torch.Tensor,
    feature_names: list[str],
    baseline: torch.Tensor | None = None,
    n_steps: int = 30,
    input_key: str = "x_seer",
) -> np.ndarray:
    """Compute Integrated Gradients for a batch of tabular inputs.

    Parameters
    ----------
    model : nn.Module
    X : (N, d_features) batch of inputs.
    feature_names : feature names.
    baseline : (d_features,) shared baseline. Defaults to zeros.
    n_steps : interpolation steps.
    input_key : model keyword argument name.

    Returns
    -------
    (N, d_features) attribution matrix.
    """
    if baseline is None:
        baseline = torch.zeros(X.shape[1], device=X.device)

    model.eval()
    all_attrs = []

    for i in range(X.shape[0]):
        attr = integrated_gradients_tabular(
            model, X[i], feature_names,
            baseline=baseline, n_steps=n_steps, input_key=input_key,
        )
        all_attrs.append(attr.values)

    return np.stack(all_attrs)


def integrated_gradients_temporal(
    model: nn.Module,
    x_seq: torch.Tensor,
    seq_mask: torch.Tensor | None = None,
    n_steps: int = 30,
    forward_kwargs: dict | None = None,
) -> np.ndarray:
    """Compute Integrated Gradients for temporal lab sequences.

    Parameters
    ----------
    model : nn.Module
        Must accept x_seq keyword argument.
    x_seq : (T, n_labs) single temporal input.
    seq_mask : (T,) valid time-step mask.
    n_steps : interpolation steps.
    forward_kwargs : additional model kwargs.

    Returns
    -------
    (T, n_labs) attribution matrix showing importance of each
    lab value at each time step.
    """
    if forward_kwargs is None:
        forward_kwargs = {}

    model.eval()
    baseline = torch.zeros_like(x_seq)

    alphas = torch.linspace(0, 1, n_steps + 1, device=x_seq.device)

    gradients = []
    for alpha in alphas:
        inp = (baseline + alpha * (x_seq - baseline)).unsqueeze(0).requires_grad_(True)
        mask_arg = seq_mask.unsqueeze(0) if seq_mask is not None else None
        output = model(x_seq=inp, seq_mask=mask_arg, **forward_kwargs)
        output.sum().backward()
        gradients.append(inp.grad.detach().squeeze(0))

    grads = torch.stack(gradients)
    avg_grads = (grads[:-1] + grads[1:]).mean(dim=0) / 2.0
    attributions = (x_seq - baseline) * avg_grads

    return attributions.cpu().numpy()


def summarize_attributions(
    attributions: np.ndarray,
    feature_names: list[str],
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """Rank features by absolute attribution magnitude.

    Parameters
    ----------
    attributions : (d_features,) attribution values.
    feature_names : matching feature names.
    top_k : number of top features to return.

    Returns
    -------
    List of (feature_name, attribution_value) sorted by |attribution|.
    """
    abs_attr = np.abs(attributions)
    indices = np.argsort(abs_attr)[::-1][:top_k]
    return [(feature_names[i], float(attributions[i])) for i in indices]
