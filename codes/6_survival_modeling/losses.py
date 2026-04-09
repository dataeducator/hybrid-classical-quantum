"""Survival loss functions for deep survival modeling.

Data-lineage:
  Model predictions (log-hazard ratios) + survival labels
    -> Cox partial likelihood loss
    -> Fairness regularization penalty

Reference:
  Katzman et al. (2018) "DeepSurv: personalized treatment recommender
  system using a Cox proportional hazards deep neural network"
"""

from __future__ import annotations

import torch
import torch.nn as nn


def cox_ph_loss(log_hazard: torch.Tensor, time: torch.Tensor,
                event: torch.Tensor) -> torch.Tensor:
    """Negative log partial likelihood for the Cox proportional hazards model.

    Parameters
    ----------
    log_hazard : (N,) predicted log-hazard ratios
    time : (N,) observed survival times
    event : (N,) event indicators (1 = event, 0 = censored)

    Returns
    -------
    Scalar loss (lower is better).
    """
    # Sort by descending time
    sorted_idx = torch.argsort(time, descending=True)
    log_hazard = log_hazard[sorted_idx]
    event = event[sorted_idx]

    # Log-cumulative-sum-exp of hazards (risk set)
    log_cumsum_h = torch.logcumsumexp(log_hazard, dim=0)

    # Partial likelihood: sum over events of (log_h_i - log_cumsum)
    uncensored = event.bool()
    if uncensored.sum() == 0:
        return torch.tensor(0.0, device=log_hazard.device, requires_grad=True)

    loss = -(log_hazard[uncensored] - log_cumsum_h[uncensored]).mean()
    return loss


class CoxPHLoss(nn.Module):
    """Module wrapper for Cox partial likelihood loss."""

    def forward(self, log_hazard: torch.Tensor, time: torch.Tensor,
                event: torch.Tensor) -> torch.Tensor:
        return cox_ph_loss(log_hazard, time, event)


def fairness_regularizer(
    log_hazard: torch.Tensor,
    group_labels: torch.Tensor,
    n_groups: int = 5,
    metric: str = "demographic_parity",
) -> torch.Tensor:
    """Compute fairness penalty based on group-level prediction disparity.

    Parameters
    ----------
    log_hazard : (N,) predicted log-hazard ratios
    group_labels : (N,) integer group labels (e.g., race codes 0-4)
    n_groups : int
        Number of distinct groups.
    metric : str
        "demographic_parity" penalizes variance in mean predicted risk
        across groups. "equalized_odds" penalizes variance in high-risk
        classification rates (requires thresholding).

    Returns
    -------
    Scalar regularization penalty.
    """
    # Compute mean hazard per group
    group_means = []
    for g in range(n_groups):
        mask = group_labels == g
        if mask.sum() > 0:
            group_means.append(log_hazard[mask].mean())

    if len(group_means) < 2:
        return torch.tensor(0.0, device=log_hazard.device, requires_grad=True)

    means = torch.stack(group_means)

    if metric == "demographic_parity":
        # Penalize variance of group-level mean predictions
        return means.var()
    elif metric == "equalized_odds":
        # Penalize variance + penalize spread of max-min
        spread = means.max() - means.min()
        return means.var() + 0.5 * spread ** 2
    else:
        return means.var()


class FairnessRegularizer(nn.Module):
    """Module wrapper for fairness regularization."""

    def __init__(self, n_groups: int = 5, metric: str = "demographic_parity",
                 weight: float = 0.1) -> None:
        super().__init__()
        self.n_groups = n_groups
        self.metric = metric
        self.weight = weight

    def forward(self, log_hazard: torch.Tensor,
                group_labels: torch.Tensor) -> torch.Tensor:
        return self.weight * fairness_regularizer(
            log_hazard, group_labels, self.n_groups, self.metric
        )
