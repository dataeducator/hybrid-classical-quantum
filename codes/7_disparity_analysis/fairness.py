"""Fairness metrics for survival model predictions.

Data-lineage:
  Model predictions + demographic labels
    -> Demographic parity gap
    -> Equalized odds gap
    -> Calibration per subgroup
    -> Comprehensive fairness audit report

Reference:
  Hardt et al. (2016) Equality of Opportunity in Supervised Learning
  Rajkomar et al. (2018) Ensuring Fairness in Machine Learning for Health
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


RACE_NAMES: dict[int, str] = {
    0: "White", 1: "Black", 2: "Asian", 3: "Hispanic", 4: "Other",
}


@dataclass
class FairnessReport:
    """Container for fairness audit results."""
    demographic_parity: dict[str, float] = field(default_factory=dict)
    equalized_odds_tpr: dict[str, float] = field(default_factory=dict)
    equalized_odds_fpr: dict[str, float] = field(default_factory=dict)
    calibration: dict[str, dict[str, float]] = field(default_factory=dict)
    dp_gap: float = 0.0
    eo_gap_tpr: float = 0.0
    eo_gap_fpr: float = 0.0
    overall_fairness_score: float = 0.0

    def summary(self) -> str:
        """Return human-readable summary."""
        lines = [
            "=== Fairness Audit Report ===",
            "",
            "Demographic Parity (P[high-risk] per group):",
        ]
        for group, rate in sorted(self.demographic_parity.items()):
            lines.append(f"  {group}: {rate:.4f}")
        lines.append(f"  Max gap: {self.dp_gap:.4f}")
        lines.append("")

        lines.append("Equalized Odds (TPR per group):")
        for group, tpr in sorted(self.equalized_odds_tpr.items()):
            lines.append(f"  {group}: TPR={tpr:.4f}")
        lines.append(f"  Max TPR gap: {self.eo_gap_tpr:.4f}")
        lines.append("")

        lines.append("Equalized Odds (FPR per group):")
        for group, fpr in sorted(self.equalized_odds_fpr.items()):
            lines.append(f"  {group}: FPR={fpr:.4f}")
        lines.append(f"  Max FPR gap: {self.eo_gap_fpr:.4f}")
        lines.append("")

        lines.append(f"Overall fairness score: {self.overall_fairness_score:.4f}")
        lines.append("  (1.0 = perfectly fair, lower = more disparity)")
        return "\n".join(lines)


def demographic_parity(
    predicted_risk: np.ndarray,
    group_labels: np.ndarray,
    threshold_percentile: float = 75.0,
    group_names: dict[int, str] | None = None,
) -> tuple[dict[str, float], float]:
    """Compute demographic parity: P(high-risk) per group.

    Parameters
    ----------
    predicted_risk : (N,) risk scores (higher = riskier)
    group_labels : (N,) integer group codes
    threshold_percentile : float
        Percentile above which a patient is "high-risk".
    group_names : dict mapping code -> name

    Returns
    -------
    rates : dict[group_name, high_risk_rate]
    gap : max difference between any two groups
    """
    if group_names is None:
        group_names = RACE_NAMES

    threshold = np.percentile(predicted_risk, threshold_percentile)
    high_risk = predicted_risk >= threshold

    rates: dict[str, float] = {}
    for code, name in group_names.items():
        mask = group_labels == code
        if mask.sum() < 5:
            continue
        rates[name] = float(high_risk[mask].mean())

    if len(rates) < 2:
        return rates, 0.0

    values = list(rates.values())
    gap = max(values) - min(values)
    return rates, gap


def equalized_odds(
    predicted_risk: np.ndarray,
    event: np.ndarray,
    group_labels: np.ndarray,
    threshold_percentile: float = 75.0,
    group_names: dict[int, str] | None = None,
) -> tuple[dict[str, float], dict[str, float], float, float]:
    """Compute equalized odds: TPR and FPR per group.

    Parameters
    ----------
    predicted_risk : (N,) risk scores
    event : (N,) true event indicators
    group_labels : (N,) group codes
    threshold_percentile : float
    group_names : dict

    Returns
    -------
    tpr_rates, fpr_rates, tpr_gap, fpr_gap
    """
    if group_names is None:
        group_names = RACE_NAMES

    threshold = np.percentile(predicted_risk, threshold_percentile)
    predicted_high = predicted_risk >= threshold

    tpr_rates: dict[str, float] = {}
    fpr_rates: dict[str, float] = {}

    for code, name in group_names.items():
        mask = group_labels == code
        if mask.sum() < 5:
            continue

        true_pos = event[mask] == 1
        true_neg = event[mask] == 0
        pred_h = predicted_high[mask]

        # TPR = P(predicted_high | event=1)
        if true_pos.sum() > 0:
            tpr_rates[name] = float(pred_h[true_pos].mean())
        else:
            tpr_rates[name] = float("nan")

        # FPR = P(predicted_high | event=0)
        if true_neg.sum() > 0:
            fpr_rates[name] = float(pred_h[true_neg].mean())
        else:
            fpr_rates[name] = float("nan")

    # Compute gaps (ignoring NaN)
    tpr_vals = [v for v in tpr_rates.values() if not np.isnan(v)]
    fpr_vals = [v for v in fpr_rates.values() if not np.isnan(v)]
    tpr_gap = (max(tpr_vals) - min(tpr_vals)) if len(tpr_vals) >= 2 else 0.0
    fpr_gap = (max(fpr_vals) - min(fpr_vals)) if len(fpr_vals) >= 2 else 0.0

    return tpr_rates, fpr_rates, tpr_gap, fpr_gap


def calibration_by_group(
    predicted_risk: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    group_labels: np.ndarray,
    eval_time: float | None = None,
    n_bins: int = 5,
    group_names: dict[int, str] | None = None,
) -> dict[str, dict[str, float]]:
    """Assess calibration per demographic group.

    For each group, bins patients by predicted risk and computes
    the observed event rate in each bin.

    Returns
    -------
    Dict[group_name, Dict[metric_name, value]] with:
      mean_predicted_risk, observed_event_rate, calibration_error
    """
    if group_names is None:
        group_names = RACE_NAMES

    results: dict[str, dict[str, float]] = {}

    for code, name in group_names.items():
        mask = group_labels == code
        if mask.sum() < 10:
            continue

        g_risk = predicted_risk[mask]
        g_event = event[mask]
        g_time = time[mask]

        # If eval_time given, define "event" as event within eval_time
        if eval_time is not None:
            observed = ((g_time <= eval_time) & (g_event == 1)).astype(float)
        else:
            observed = g_event.astype(float)

        # Bin by predicted risk
        bin_edges = np.percentile(g_risk, np.linspace(0, 100, n_bins + 1))
        bin_edges[-1] += 1e-6
        bin_idx = np.digitize(g_risk, bin_edges) - 1
        bin_idx = np.clip(bin_idx, 0, n_bins - 1)

        cal_errors = []
        for b in range(n_bins):
            in_bin = bin_idx == b
            if in_bin.sum() == 0:
                continue
            pred_mean = g_risk[in_bin].mean()
            obs_mean = observed[in_bin].mean()
            cal_errors.append(abs(pred_mean - obs_mean))

        mean_cal_error = np.mean(cal_errors) if cal_errors else 0.0

        results[name] = {
            "mean_predicted_risk": float(g_risk.mean()),
            "observed_event_rate": float(observed.mean()),
            "calibration_error": float(mean_cal_error),
            "n_patients": int(mask.sum()),
        }

    return results


def fairness_audit(
    predicted_risk: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    group_labels: np.ndarray,
    group_names: dict[int, str] | None = None,
    threshold_percentile: float = 75.0,
) -> FairnessReport:
    """Run a comprehensive fairness audit on survival predictions.

    Parameters
    ----------
    predicted_risk : (N,) risk scores
    time : (N,) survival times
    event : (N,) event indicators
    group_labels : (N,) demographic group codes
    group_names : dict mapping code -> name
    threshold_percentile : percentile for binary high-risk classification

    Returns
    -------
    FairnessReport with all metrics.
    """
    if group_names is None:
        group_names = RACE_NAMES

    report = FairnessReport()

    # Demographic parity
    dp_rates, dp_gap = demographic_parity(
        predicted_risk, group_labels, threshold_percentile, group_names
    )
    report.demographic_parity = dp_rates
    report.dp_gap = dp_gap

    # Equalized odds
    tpr_rates, fpr_rates, tpr_gap, fpr_gap = equalized_odds(
        predicted_risk, event, group_labels, threshold_percentile, group_names
    )
    report.equalized_odds_tpr = tpr_rates
    report.equalized_odds_fpr = fpr_rates
    report.eo_gap_tpr = tpr_gap
    report.eo_gap_fpr = fpr_gap

    # Calibration
    report.calibration = calibration_by_group(
        predicted_risk, time, event, group_labels,
        group_names=group_names,
    )

    # Overall fairness score (1 = perfectly fair)
    # Combines DP gap, EO TPR gap, EO FPR gap
    report.overall_fairness_score = max(
        0.0, 1.0 - (dp_gap + tpr_gap + fpr_gap) / 3.0
    )

    return report
