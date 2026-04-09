"""LIME explanations for individual TNBC patient predictions.

Data-lineage:
  Trained model + single patient features
    -> local linear surrogate model
    -> per-feature contribution to prediction
    -> human-readable explanation text

LIME is especially useful for clinician-facing explanations because
it produces intuitive "if-then" style feature contributions.

Reference:
  Ribeiro et al. (2016) '"Why Should I Trust You?": Explaining the
    Predictions of Any Classifier'
"""

from __future__ import annotations

from dataclasses import dataclass, field

import lime.lime_tabular
import numpy as np
import torch
import torch.nn as nn


@dataclass
class LIMEExplanation:
    """Container for a single patient's LIME explanation."""
    patient_id: str
    predicted_risk: float
    feature_contributions: list[tuple[str, float]]  # sorted by |weight|
    intercept: float
    local_r2: float
    explanation_text: str


def explain_patient(
    model: nn.Module,
    x: np.ndarray,
    X_train: np.ndarray,
    feature_names: list[str],
    patient_id: str = "unknown",
    num_features: int = 10,
    num_samples: int = 500,
    input_key: str = "x_seer",
) -> LIMEExplanation:
    """Generate a LIME explanation for a single patient.

    Parameters
    ----------
    model : nn.Module
        Trained survival model.
    x : (d_features,) single patient feature vector.
    X_train : (N, d_features) training data for LIME's background.
    feature_names : list of feature names.
    patient_id : patient identifier for the report.
    num_features : number of top features in explanation.
    num_samples : perturbation samples for LIME.
    input_key : keyword argument name for model.

    Returns
    -------
    LIMEExplanation for this patient.
    """
    model.eval()
    device = next(model.parameters()).device

    def predict_fn(x_np: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            x_t = torch.from_numpy(x_np.astype(np.float32)).to(device)
            preds = model(**{input_key: x_t}).cpu().numpy()
        # LIME expects 2D output for regression
        return preds.reshape(-1, 1)

    # Build LIME explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(
        training_data=X_train,
        feature_names=feature_names,
        mode="regression",
        verbose=False,
    )

    # Explain this instance
    exp = explainer.explain_instance(
        x,
        predict_fn,
        num_features=num_features,
        num_samples=num_samples,
    )

    # Extract contributions
    contributions = exp.as_list()
    local_pred = exp.local_pred[0] if hasattr(exp, 'local_pred') and exp.local_pred is not None else 0.0
    intercept = exp.intercept[0] if hasattr(exp, 'intercept') else 0.0
    score = exp.score if hasattr(exp, 'score') else 0.0

    # Get actual prediction
    with torch.no_grad():
        x_t = torch.from_numpy(x.astype(np.float32)).unsqueeze(0).to(device)
        pred_risk = model(**{input_key: x_t}).item()

    # Build human-readable explanation
    text = _build_explanation_text(patient_id, pred_risk, contributions)

    return LIMEExplanation(
        patient_id=patient_id,
        predicted_risk=pred_risk,
        feature_contributions=contributions,
        intercept=intercept,
        local_r2=score,
        explanation_text=text,
    )


def explain_cohort(
    model: nn.Module,
    X: np.ndarray,
    X_train: np.ndarray,
    feature_names: list[str],
    patient_ids: list[str] | None = None,
    num_features: int = 10,
    num_samples: int = 300,
    input_key: str = "x_seer",
    max_patients: int = 100,
) -> list[LIMEExplanation]:
    """Generate LIME explanations for multiple patients.

    Parameters
    ----------
    model : nn.Module
    X : (N, d_features) patients to explain.
    X_train : training background data.
    feature_names : feature names.
    patient_ids : optional list of patient IDs.
    num_features : features per explanation.
    num_samples : LIME perturbation samples.
    input_key : model keyword.
    max_patients : cap on number of patients to explain.

    Returns
    -------
    List of LIMEExplanation objects.
    """
    n = min(len(X), max_patients)
    if patient_ids is None:
        patient_ids = [f"patient_{i}" for i in range(n)]

    explanations = []
    for i in range(n):
        exp = explain_patient(
            model, X[i], X_train, feature_names,
            patient_id=patient_ids[i],
            num_features=num_features,
            num_samples=num_samples,
            input_key=input_key,
        )
        explanations.append(exp)

    return explanations


def aggregate_lime_importance(
    explanations: list[LIMEExplanation],
) -> list[tuple[str, float]]:
    """Aggregate LIME feature importance across patients.

    Averages absolute LIME weights for each feature across all
    explained patients.

    Returns
    -------
    List of (feature_name, mean_abs_weight) sorted descending.
    """
    feature_weights: dict[str, list[float]] = {}

    for exp in explanations:
        for feat_rule, weight in exp.feature_contributions:
            # LIME rules look like "feature_name <= 0.5"
            # Extract the base feature name
            base_name = feat_rule.split(" ")[0] if " " in feat_rule else feat_rule
            feature_weights.setdefault(base_name, []).append(abs(weight))

    importance = [
        (name, float(np.mean(weights)))
        for name, weights in feature_weights.items()
    ]
    importance.sort(key=lambda x: x[1], reverse=True)
    return importance


def _build_explanation_text(
    patient_id: str,
    predicted_risk: float,
    contributions: list[tuple[str, float]],
) -> str:
    """Build human-readable LIME explanation for a patient."""
    risk_level = "HIGH" if predicted_risk > 0 else "LOW"
    lines = [
        f"Patient {patient_id}: Predicted risk = {predicted_risk:.3f} ({risk_level})",
        "",
        "Key factors driving this prediction:",
    ]

    for feat_rule, weight in contributions[:7]:
        direction = "increases" if weight > 0 else "decreases"
        lines.append(f"  - {feat_rule}: {direction} risk by {abs(weight):.3f}")

    return "\n".join(lines)
