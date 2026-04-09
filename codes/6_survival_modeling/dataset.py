"""PyTorch datasets for SEER and MIMIC survival modeling.

Data-lineage:
  Feature tensors (from 2_feature_engineering)
    -> SeerSurvivalDataset  (tabular + survival labels)
    -> MimicSurvivalDataset (tabular + temporal + text + survival labels)

Both datasets return dicts of tensors so the model can flexibly
accept whichever modalities are available.
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

# Deferred import to avoid circular dependency at module level
if sys.version_info >= (3, 11):
    from typing import Self
else:
    Self = Any


class SeerSurvivalDataset(Dataset):
    """PyTorch dataset for SEER tabular survival data.

    Each item returns a dict with keys:
      x_seer:  (d_features,)  float32  tabular features
      time:    ()              float32  survival months
      event:   ()              int32    event indicator
      race:    ()              int32    race code for fairness
    """

    def __init__(self, X: np.ndarray, time: np.ndarray,
                 event: np.ndarray, race: np.ndarray) -> None:
        self.X = torch.from_numpy(X).float()
        self.time = torch.from_numpy(time).float()
        self.event = torch.from_numpy(event).int()
        self.race = torch.from_numpy(race).int()

    def __len__(self) -> int:
        return len(self.X)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "x_seer": self.X[idx],
            "time": self.time[idx],
            "event": self.event[idx],
            "race": self.race[idx],
        }


class MimicSurvivalDataset(Dataset):
    """PyTorch dataset for MIMIC multi-modal survival data.

    Each item returns a dict with keys:
      x_mimic_tab: (d_tab,)    float32  tabular features
      x_seq:       (T, 7)      float32  lab sequences
      seq_mask:    (T,)        bool     valid time-step mask
      x_text:      (768,)      float32  ClinicalBERT embedding
      time:        ()          float32  survival days
      event:       ()          int32    event indicator
      race:        ()          int32    race code for fairness
    """

    def __init__(
        self,
        X_tab: np.ndarray,
        X_seq: np.ndarray,
        seq_mask: np.ndarray,
        X_text: np.ndarray,
        time: np.ndarray,
        event: np.ndarray,
        race: np.ndarray,
    ) -> None:
        self.X_tab = torch.from_numpy(X_tab).float()
        self.X_seq = torch.from_numpy(X_seq).float()
        self.seq_mask = torch.from_numpy(seq_mask).bool()
        self.X_text = torch.from_numpy(X_text).float()
        self.time = torch.from_numpy(time).float()
        self.event = torch.from_numpy(event).int()
        self.race = torch.from_numpy(race).int()

    def __len__(self) -> int:
        return len(self.X_tab)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "x_mimic_tab": self.X_tab[idx],
            "x_seq": self.X_seq[idx],
            "seq_mask": self.seq_mask[idx],
            "x_text": self.X_text[idx],
            "time": self.time[idx],
            "event": self.event[idx],
            "race": self.race[idx],
        }


def train_test_split_temporal(
    dataset: SeerSurvivalDataset | MimicSurvivalDataset,
    test_frac: float = 0.2,
    seed: int = 42,
) -> tuple[Dataset, Dataset]:
    """Split a survival dataset into train/test, preserving event ratio.

    Uses stratified sampling on the event indicator to maintain
    censoring balance across splits.
    """
    rng = np.random.default_rng(seed)
    n = len(dataset)
    events = dataset.event.numpy()

    # Stratify by event indicator
    idx_event = np.where(events == 1)[0]
    idx_censor = np.where(events == 0)[0]

    rng.shuffle(idx_event)
    rng.shuffle(idx_censor)

    n_test_event = max(1, int(len(idx_event) * test_frac))
    n_test_censor = max(1, int(len(idx_censor) * test_frac))

    test_idx = np.concatenate([
        idx_event[:n_test_event],
        idx_censor[:n_test_censor],
    ])
    train_idx = np.concatenate([
        idx_event[n_test_event:],
        idx_censor[n_test_censor:],
    ])

    return (
        torch.utils.data.Subset(dataset, train_idx.tolist()),
        torch.utils.data.Subset(dataset, test_idx.tolist()),
    )
