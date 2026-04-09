"""Stratified survival analysis by demographic subgroups.

Data-lineage:
  Model predictions + demographic/SES labels
    -> Kaplan-Meier estimates per subgroup
    -> Risk-stratified C-index per subgroup
    -> Intersectional analysis (race x income)

Provides the statistical evidence needed to identify and report
health disparities in TNBC survival predictions.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class KaplanMeierEstimate:
    """Kaplan-Meier survival curve for a subgroup."""
    times: np.ndarray
    survival_prob: np.ndarray
    n_at_risk: np.ndarray
    n_events: int
    n_censored: int
    median_survival: float | None


@dataclass
class SubgroupReport:
    """Container for subgroup analysis results."""
    km_curves: dict[str, KaplanMeierEstimate] = field(default_factory=dict)
    c_indices: dict[str, float] = field(default_factory=dict)
    median_risks: dict[str, float] = field(default_factory=dict)
    intersectional: dict[str, dict[str, float]] = field(default_factory=dict)


def kaplan_meier(
    time: np.ndarray,
    event: np.ndarray,
) -> KaplanMeierEstimate:
    """Compute Kaplan-Meier survival estimate.

    Parameters
    ----------
    time : (N,) observed survival times
    event : (N,) event indicators (1=event, 0=censored)

    Returns
    -------
    KaplanMeierEstimate with survival curve data.
    """
    n = len(time)
    order = np.argsort(time)
    sorted_time = time[order]
    sorted_event = event[order]

    unique_times = np.unique(sorted_time[sorted_event == 1])
    survival = np.ones(len(unique_times) + 1)
    km_times = np.zeros(len(unique_times) + 1)
    n_at_risk_arr = np.zeros(len(unique_times) + 1, dtype=int)

    km_times[0] = 0
    n_at_risk_arr[0] = n
    s = 1.0

    for i, t in enumerate(unique_times):
        n_at_risk = (sorted_time >= t).sum()
        n_events_t = ((sorted_time == t) & (sorted_event == 1)).sum()
        s *= (1.0 - n_events_t / n_at_risk) if n_at_risk > 0 else s
        km_times[i + 1] = t
        survival[i + 1] = s
        n_at_risk_arr[i + 1] = n_at_risk

    # Median survival
    below_50 = survival <= 0.5
    median_surv = float(km_times[below_50][0]) if below_50.any() else None

    return KaplanMeierEstimate(
        times=km_times,
        survival_prob=survival,
        n_at_risk=n_at_risk_arr,
        n_events=int(event.sum()),
        n_censored=int((event == 0).sum()),
        median_survival=median_surv,
    )


def subgroup_analysis(
    predicted_risk: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    race_labels: np.ndarray,
    income_labels: np.ndarray | None = None,
    race_names: dict[int, str] | None = None,
) -> SubgroupReport:
    """Run stratified survival analysis across demographic subgroups.

    Parameters
    ----------
    predicted_risk : (N,) model-predicted risk scores
    time : (N,) survival times
    event : (N,) event indicators
    race_labels : (N,) race codes
    income_labels : (N,) income quintile codes (0-4), optional
    race_names : dict mapping race code -> name

    Returns
    -------
    SubgroupReport with KM curves, C-indices, and intersectional analysis.
    """
    if race_names is None:
        race_names = {0: "White", 1: "Black", 2: "Asian",
                      3: "Hispanic", 4: "Other"}

    report = SubgroupReport()

    # Per-race analysis
    for code, name in race_names.items():
        mask = race_labels == code
        if mask.sum() < 10:
            continue

        # KM curve
        report.km_curves[name] = kaplan_meier(time[mask], event[mask])

        # Median predicted risk
        report.median_risks[name] = float(np.median(predicted_risk[mask]))

        # C-index within group
        if event[mask].sum() >= 2:
            import importlib
            _eval_mod = importlib.import_module("6_survival_modeling.evaluate")
            report.c_indices[name] = _eval_mod.concordance_index_fast(
                predicted_risk[mask], time[mask], event[mask]
            )

    # Intersectional analysis (race x income)
    if income_labels is not None:
        income_names = {0: "Q1_low", 1: "Q2", 2: "Q3", 3: "Q4", 4: "Q5_high"}
        for race_code, race_name in race_names.items():
            for inc_code, inc_name in income_names.items():
                mask = (race_labels == race_code) & (income_labels == inc_code)
                if mask.sum() < 5:
                    continue
                key = f"{race_name}_x_{inc_name}"
                report.intersectional[key] = {
                    "n": int(mask.sum()),
                    "event_rate": float(event[mask].mean()),
                    "median_risk": float(np.median(predicted_risk[mask])),
                    "median_survival": float(np.median(time[mask])),
                }

    return report


def print_subgroup_report(report: SubgroupReport) -> None:
    """Print subgroup analysis results."""
    print("=== Subgroup Survival Analysis ===\n")

    print("Kaplan-Meier Median Survival (months/days):")
    for name, km in sorted(report.km_curves.items()):
        med = f"{km.median_survival:.0f}" if km.median_survival else "NR"
        print(f"  {name:15s}: median={med}, events={km.n_events}, "
              f"censored={km.n_censored}")

    if report.c_indices:
        print("\nC-index by subgroup:")
        for name, ci in sorted(report.c_indices.items()):
            print(f"  {name:15s}: {ci:.4f}")

    print("\nMedian predicted risk by subgroup:")
    for name, risk in sorted(report.median_risks.items()):
        print(f"  {name:15s}: {risk:.4f}")

    if report.intersectional:
        print("\nIntersectional analysis (race x income):")
        for key, vals in sorted(report.intersectional.items()):
            print(f"  {key:25s}: n={vals['n']:4d}, "
                  f"event_rate={vals['event_rate']:.3f}, "
                  f"median_risk={vals['median_risk']:.3f}")
