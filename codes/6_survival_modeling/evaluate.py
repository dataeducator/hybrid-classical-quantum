"""Evaluation metrics for survival models.

Data-lineage:
  Model predictions + true survival labels
    -> Concordance index (C-index)
    -> Brier score (at specific time horizons)
    -> Calibration metrics per subgroup

Reference:
  Harrell et al. (1996) Concordance index
  Graf et al. (1999) Brier score for survival
"""

from __future__ import annotations

import numpy as np


def concordance_index(
    predicted_risk: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
) -> float:
    """Compute Harrell's concordance index (C-index).

    The C-index measures the probability that, for a random pair of
    patients, the patient with higher predicted risk experiences the
    event sooner. C=0.5 is random, C=1.0 is perfect.

    Parameters
    ----------
    predicted_risk : (N,) higher = higher risk (e.g., log-hazard)
    time : (N,) survival times
    event : (N,) 1=event occurred, 0=censored

    Returns
    -------
    C-index in [0, 1].
    """
    n = len(predicted_risk)
    concordant = 0
    discordant = 0
    tied_risk = 0

    for i in range(n):
        for j in range(i + 1, n):
            # Only count pairs where we can determine ordering
            if event[i] == 0 and event[j] == 0:
                continue
            if event[i] == 1 and event[j] == 1:
                # Both events: shorter time = higher risk expected
                if time[i] < time[j]:
                    if predicted_risk[i] > predicted_risk[j]:
                        concordant += 1
                    elif predicted_risk[i] < predicted_risk[j]:
                        discordant += 1
                    else:
                        tied_risk += 0.5
                elif time[i] > time[j]:
                    if predicted_risk[i] < predicted_risk[j]:
                        concordant += 1
                    elif predicted_risk[i] > predicted_risk[j]:
                        discordant += 1
                    else:
                        tied_risk += 0.5
            elif event[i] == 1 and time[i] < time[j]:
                # i had event before j was censored
                if predicted_risk[i] > predicted_risk[j]:
                    concordant += 1
                elif predicted_risk[i] < predicted_risk[j]:
                    discordant += 1
                else:
                    tied_risk += 0.5
            elif event[j] == 1 and time[j] < time[i]:
                if predicted_risk[j] > predicted_risk[i]:
                    concordant += 1
                elif predicted_risk[j] < predicted_risk[i]:
                    discordant += 1
                else:
                    tied_risk += 0.5

    total = concordant + discordant + tied_risk
    if total == 0:
        return 0.5
    return (concordant + tied_risk) / total


def concordance_index_fast(
    predicted_risk: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
) -> float:
    """Vectorized C-index computation (much faster for large N).

    Uses the same definition as concordance_index but avoids the O(N^2)
    Python loop by using numpy broadcasting.
    """
    n = len(predicted_risk)
    if n < 2:
        return 0.5

    # For large datasets, use sampling to keep computation tractable
    if n > 5000:
        rng = np.random.default_rng(0)
        idx = rng.choice(n, size=5000, replace=False)
        predicted_risk = predicted_risk[idx]
        time = time[idx]
        event = event[idx]
        n = 5000

    # Build pairwise comparison matrices
    risk_i = predicted_risk[:, None]
    risk_j = predicted_risk[None, :]
    time_i = time[:, None]
    time_j = time[None, :]
    event_i = event[:, None]
    event_j = event[None, :]

    # Valid pairs: i had event and time_i < time_j (or both events with different times)
    valid = np.zeros((n, n), dtype=bool)
    valid |= (event_i == 1) & (time_i < time_j)
    valid |= (event_j == 1) & (time_j < time_i)
    valid |= (event_i == 1) & (event_j == 1) & (time_i != time_j)

    # Remove double-counting and diagonal
    valid = np.triu(valid, k=1)

    if valid.sum() == 0:
        return 0.5

    # For valid pairs where time_i < time_j, concordant if risk_i > risk_j
    shorter_i = time_i < time_j
    shorter_j = time_j < time_i

    concordant = (
        (valid & shorter_i & (risk_i > risk_j)).sum()
        + (valid & shorter_j & (risk_j > risk_i)).sum()
    )
    discordant = (
        (valid & shorter_i & (risk_i < risk_j)).sum()
        + (valid & shorter_j & (risk_j < risk_i)).sum()
    )
    tied = (valid & (risk_i == risk_j)).sum() * 0.5

    total = concordant + discordant + tied
    if total == 0:
        return 0.5
    return float((concordant + tied) / total)


def brier_score(
    predicted_survival_prob: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    eval_time: float,
) -> float:
    """Compute Brier score at a specific time horizon.

    Parameters
    ----------
    predicted_survival_prob : (N,) predicted P(T > eval_time)
    time : (N,) observed survival times
    event : (N,) event indicators
    eval_time : float
        Time point at which to evaluate.

    Returns
    -------
    Brier score (lower is better). Range [0, 1].
    """
    n = len(time)
    scores = np.zeros(n)

    for i in range(n):
        if time[i] <= eval_time and event[i] == 1:
            # Event before eval_time: should have predicted low survival
            scores[i] = predicted_survival_prob[i] ** 2
        elif time[i] > eval_time:
            # Survived past eval_time: should have predicted high survival
            scores[i] = (1.0 - predicted_survival_prob[i]) ** 2
        # else: censored before eval_time -> excluded (IPCW would weight this)

    return float(scores.mean())


def subgroup_c_index(
    predicted_risk: np.ndarray,
    time: np.ndarray,
    event: np.ndarray,
    group_labels: np.ndarray,
    group_names: dict[int, str] | None = None,
) -> dict[str, float]:
    """Compute C-index for each subgroup.

    Returns
    -------
    Dict mapping group name -> C-index.
    """
    if group_names is None:
        unique_groups = np.unique(group_labels)
        group_names = {int(g): f"group_{g}" for g in unique_groups}

    results: dict[str, float] = {}
    for code, name in group_names.items():
        mask = group_labels == code
        if mask.sum() < 10:
            results[name] = float("nan")
            continue
        results[name] = concordance_index_fast(
            predicted_risk[mask], time[mask], event[mask]
        )

    return results
