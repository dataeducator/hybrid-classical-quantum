"""Per-patient explanation report combining all explainability methods.

Data-lineage:
  Trained model + patient features
    -> Integrated Gradients attributions
    -> SHAP values
    -> LIME local explanations
    -> Temporal attention (if MIMIC data)
    -> Combined JSON report per patient

Output format:
  {
    "patient_id": "SEER_000001",
    "predicted_risk": 1.23,
    "risk_percentile": 85.2,
    "integrated_gradients": {"top_features": [...]},
    "shap": {"top_features": [...]},
    "lime": {"contributions": [...], "explanation_text": "..."},
    "temporal_attention": {"time_importance": [...], "lab_importance": [...]},
    "model_metadata": {"model_type": "...", "c_index": 0.72}
  }
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .integrated_gradients import (
    integrated_gradients_tabular,
    summarize_attributions,
)
from .lime_explainer import explain_patient as lime_explain
from .shap_explainer import explain_seer_tower as shap_explain


@dataclass
class PatientReport:
    """Complete explanation report for a single patient."""
    patient_id: str
    predicted_risk: float
    risk_percentile: float

    # Integrated Gradients
    ig_top_features: list[tuple[str, float]] = field(default_factory=list)
    ig_all_values: list[float] = field(default_factory=list)

    # SHAP
    shap_top_features: list[tuple[str, float]] = field(default_factory=list)
    shap_all_values: list[float] = field(default_factory=list)

    # LIME
    lime_contributions: list[tuple[str, float]] = field(default_factory=list)
    lime_explanation_text: str = ""
    lime_local_r2: float = 0.0

    # Temporal attention (if available)
    time_importance: list[float] = field(default_factory=list)
    lab_importance: list[float] = field(default_factory=list)

    # Metadata
    model_type: str = ""
    feature_names: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        return {
            "patient_id": self.patient_id,
            "predicted_risk": round(self.predicted_risk, 4),
            "risk_percentile": round(self.risk_percentile, 1),
            "integrated_gradients": {
                "top_features": [
                    {"feature": f, "attribution": round(v, 4)}
                    for f, v in self.ig_top_features
                ],
            },
            "shap": {
                "top_features": [
                    {"feature": f, "importance": round(v, 4)}
                    for f, v in self.shap_top_features
                ],
            },
            "lime": {
                "contributions": [
                    {"rule": f, "weight": round(v, 4)}
                    for f, v in self.lime_contributions
                ],
                "explanation_text": self.lime_explanation_text,
                "local_r2": round(self.lime_local_r2, 4),
            },
            "temporal_attention": {
                "time_importance": [round(v, 4) for v in self.time_importance],
                "lab_importance": [round(v, 4) for v in self.lab_importance],
            },
            "model_metadata": {
                "model_type": self.model_type,
                "feature_names": self.feature_names,
            },
        }

    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)


def generate_patient_report(
    model: nn.Module,
    x: np.ndarray,
    X_train: np.ndarray,
    feature_names: list[str],
    patient_id: str,
    all_predictions: np.ndarray | None = None,
    model_type: str = "seer_only",
    top_k: int = 7,
    ig_steps: int = 30,
    lime_samples: int = 300,
    input_key: str = "x_seer",
) -> PatientReport:
    """Generate a comprehensive explanation report for one patient.

    Runs Integrated Gradients, SHAP (single-instance), and LIME
    for a complete multi-method explanation.

    Parameters
    ----------
    model : nn.Module
        Trained survival model.
    x : (d_features,) patient feature vector.
    X_train : (N, d_features) training data for background.
    feature_names : feature names.
    patient_id : patient identifier.
    all_predictions : (M,) all model predictions for percentile calc.
    model_type : description string for metadata.
    top_k : number of top features to include.
    ig_steps : Integrated Gradients interpolation steps.
    lime_samples : LIME perturbation samples.
    input_key : model keyword argument name.

    Returns
    -------
    PatientReport with all explanations.
    """
    device = next(model.parameters()).device
    x_tensor = torch.from_numpy(x.astype(np.float32)).to(device)

    # Get prediction
    model.eval()
    with torch.no_grad():
        pred_risk = model(**{input_key: x_tensor.unsqueeze(0)}).item()

    # Risk percentile
    if all_predictions is not None:
        percentile = float((all_predictions < pred_risk).mean() * 100)
    else:
        percentile = 50.0

    report = PatientReport(
        patient_id=patient_id,
        predicted_risk=pred_risk,
        risk_percentile=percentile,
        model_type=model_type,
        feature_names=feature_names,
    )

    # 1. Integrated Gradients
    ig_attr = integrated_gradients_tabular(
        model, x_tensor, feature_names,
        n_steps=ig_steps, input_key=input_key,
    )
    report.ig_top_features = summarize_attributions(
        ig_attr.values, feature_names, top_k=top_k
    )
    report.ig_all_values = ig_attr.values.tolist()

    # 2. LIME
    lime_exp = lime_explain(
        model, x, X_train, feature_names,
        patient_id=patient_id,
        num_features=top_k,
        num_samples=lime_samples,
        input_key=input_key,
    )
    report.lime_contributions = lime_exp.feature_contributions
    report.lime_explanation_text = lime_exp.explanation_text
    report.lime_local_r2 = lime_exp.local_r2

    # 3. SHAP (single patient - use X_train as background pool)
    # Combine background + patient so explainer has proper background
    shap_input = np.vstack([X_train[:min(50, len(X_train))], x.reshape(1, -1)])
    shap_exp = shap_explain(
        model, shap_input, feature_names,
        background_size=min(50, len(shap_input) - 1),
        n_samples=100,
        input_key=input_key,
    )
    shap_vals = shap_exp.shap_values[-1]  # last row = the patient
    abs_shap = np.abs(shap_vals)
    top_idx = np.argsort(abs_shap)[::-1][:top_k]
    report.shap_top_features = [
        (feature_names[i], float(shap_vals[i])) for i in top_idx
    ]
    report.shap_all_values = shap_vals.tolist()

    return report


def generate_cohort_reports(
    model: nn.Module,
    X: np.ndarray,
    X_train: np.ndarray,
    feature_names: list[str],
    patient_ids: list[str],
    output_dir: str | Path = "data/output/explanations",
    max_patients: int = 50,
    input_key: str = "x_seer",
) -> list[PatientReport]:
    """Generate explanation reports for multiple patients and save as JSON.

    Parameters
    ----------
    model : nn.Module
    X : (N, d_features) patients to explain.
    X_train : training background data.
    feature_names : feature names.
    patient_ids : patient identifiers.
    output_dir : directory for JSON output files.
    max_patients : cap on patients to explain.
    input_key : model keyword argument.

    Returns
    -------
    List of PatientReport objects (also saved as JSON files).
    """
    device = next(model.parameters()).device
    model.eval()

    # Get all predictions for percentile calculation
    with torch.no_grad():
        all_x = torch.from_numpy(X_train.astype(np.float32)).to(device)
        all_preds = model(**{input_key: all_x}).cpu().numpy()

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    n = min(len(X), max_patients)
    reports = []

    for i in range(n):
        report = generate_patient_report(
            model, X[i], X_train, feature_names,
            patient_id=patient_ids[i],
            all_predictions=all_preds,
            input_key=input_key,
        )
        reports.append(report)

        # Save individual JSON
        json_path = out_path / f"{patient_ids[i]}_explanation.json"
        json_path.write_text(report.to_json())

    # Save summary JSON
    summary = {
        "n_patients": n,
        "reports": [r.to_dict() for r in reports],
    }
    (out_path / "cohort_explanations.json").write_text(
        json.dumps(summary, indent=2)
    )

    print(f"  Saved {n} explanation reports to {out_path}")
    return reports


def print_patient_report(report: PatientReport) -> None:
    """Print a human-readable patient explanation report."""
    print(f"=== Explanation Report: {report.patient_id} ===\n")
    print(f"Predicted risk: {report.predicted_risk:.4f} "
          f"(percentile: {report.risk_percentile:.1f}%)\n")

    print("-- Integrated Gradients --")
    for feat, val in report.ig_top_features:
        direction = "+" if val > 0 else "-"
        bar = "#" * min(30, int(abs(val) * 50))
        print(f"  {direction} {feat:30s} {val:+.4f}  {bar}")

    print("\n-- SHAP Values --")
    for feat, val in report.shap_top_features:
        direction = "+" if val > 0 else "-"
        bar = "#" * min(30, int(abs(val) * 50))
        print(f"  {direction} {feat:30s} {val:+.4f}  {bar}")

    print(f"\n-- LIME (local R2={report.lime_local_r2:.3f}) --")
    print(report.lime_explanation_text)

    if report.time_importance:
        print("\n-- Temporal Attention --")
        print(f"  Time steps with highest importance: ", end="")
        imp = np.array(report.time_importance)
        top_t = np.argsort(imp)[::-1][:5]
        print(", ".join(f"t={t}({imp[t]:.3f})" for t in top_t))
