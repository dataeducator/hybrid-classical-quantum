"""Training loop for multi-source survival models with fairness constraints.

Data-lineage:
  Datasets (SEER / MIMIC) + Model
    -> Training with Cox PH loss + fairness regularization
    -> Checkpointed model + training history

Training stages:
  Stage 1: Pre-train SEER tower on population data (Cox loss)
  Stage 2: Train MIMIC model (temporal + text + tab)
  Stage 3: Transfer SEER tower weights -> fusion model fine-tuning
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .evaluate import concordance_index_fast
from .fusion import MultiSourceSurvivalModel
from .losses import CoxPHLoss, FairnessRegularizer


@dataclass
class TrainingConfig:
    """Hyperparameters for survival model training."""
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    epochs: int = 100
    patience: int = 15
    fairness_weight: float = 0.1
    fairness_metric: str = "demographic_parity"
    n_race_groups: int = 5
    checkpoint_dir: str = "checkpoints"
    device: str = "auto"


@dataclass
class TrainingHistory:
    """Record of training metrics per epoch."""
    train_loss: list[float] = field(default_factory=list)
    val_loss: list[float] = field(default_factory=list)
    val_c_index: list[float] = field(default_factory=list)
    fairness_penalty: list[float] = field(default_factory=list)
    best_epoch: int = 0
    best_c_index: float = 0.0


def _get_device(config: TrainingConfig) -> torch.device:
    if config.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(config.device)


def _collate_seer(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Collate SEER dataset items into batched tensors."""
    return {
        "x_seer": torch.stack([b["x_seer"] for b in batch]),
        "time": torch.stack([b["time"] for b in batch]),
        "event": torch.stack([b["event"] for b in batch]),
        "race": torch.stack([b["race"] for b in batch]),
    }


def _collate_mimic(batch: list[dict]) -> dict[str, torch.Tensor]:
    """Collate MIMIC dataset items into batched tensors."""
    return {
        "x_mimic_tab": torch.stack([b["x_mimic_tab"] for b in batch]),
        "x_seq": torch.stack([b["x_seq"] for b in batch]),
        "seq_mask": torch.stack([b["seq_mask"] for b in batch]),
        "x_text": torch.stack([b["x_text"] for b in batch]),
        "time": torch.stack([b["time"] for b in batch]),
        "event": torch.stack([b["event"] for b in batch]),
        "race": torch.stack([b["race"] for b in batch]),
    }


def train_seer_model(
    model: MultiSourceSurvivalModel,
    train_dataset: torch.utils.data.Dataset,
    val_dataset: torch.utils.data.Dataset,
    config: TrainingConfig | None = None,
) -> TrainingHistory:
    """Train a SEER-only survival model (Stage 1: pre-training).

    Parameters
    ----------
    model : MultiSourceSurvivalModel
        Should be built with ``build_seer_only_model()``.
    train_dataset, val_dataset : Dataset
        Output of ``train_test_split_temporal(SeerSurvivalDataset(...))``.
    config : TrainingConfig
        Hyperparameters.

    Returns
    -------
    TrainingHistory with per-epoch metrics.
    """
    if config is None:
        config = TrainingConfig()

    device = _get_device(config)
    model = model.to(device)

    cox_loss = CoxPHLoss()
    fairness_reg = FairnessRegularizer(
        n_groups=config.n_race_groups,
        metric=config.fairness_metric,
        weight=config.fairness_weight,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        collate_fn=_collate_seer, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        collate_fn=_collate_seer,
    )

    history = TrainingHistory()
    best_c = 0.0
    patience_counter = 0
    ckpt_dir = Path(config.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(config.epochs):
        # ---- Training ----
        model.train()
        epoch_loss = 0.0
        epoch_fair = 0.0
        n_batches = 0

        for batch in train_loader:
            x = batch["x_seer"].to(device)
            t = batch["time"].to(device)
            e = batch["event"].to(device)
            r = batch["race"].to(device)

            log_h = model(x_seer=x)
            loss = cox_loss(log_h, t, e)
            fair_pen = fairness_reg(log_h, r)
            total_loss = loss + fair_pen

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_fair += fair_pen.item()
            n_batches += 1

        scheduler.step()

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_fair = epoch_fair / max(n_batches, 1)
        history.train_loss.append(avg_loss)
        history.fairness_penalty.append(avg_fair)

        # ---- Validation ----
        model.eval()
        val_risks, val_times, val_events = [], [], []

        with torch.no_grad():
            val_loss_sum = 0.0
            val_n = 0
            for batch in val_loader:
                x = batch["x_seer"].to(device)
                t = batch["time"].to(device)
                e = batch["event"].to(device)

                log_h = model(x_seer=x)
                vl = cox_loss(log_h, t, e)
                val_loss_sum += vl.item()
                val_n += 1

                val_risks.append(log_h.cpu().numpy())
                val_times.append(t.cpu().numpy())
                val_events.append(e.cpu().numpy())

        history.val_loss.append(val_loss_sum / max(val_n, 1))

        all_risks = np.concatenate(val_risks)
        all_times = np.concatenate(val_times)
        all_events = np.concatenate(val_events)
        c_index = concordance_index_fast(all_risks, all_times, all_events)
        history.val_c_index.append(c_index)

        # Early stopping on C-index
        if c_index > best_c:
            best_c = c_index
            history.best_c_index = c_index
            history.best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_dir / "best_seer_model.pt")
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"  Epoch {epoch+1:3d}/{config.epochs} | "
                f"Loss {avg_loss:.4f} | Fair {avg_fair:.4f} | "
                f"Val C-index {c_index:.4f} (best {best_c:.4f})"
            )

        if patience_counter >= config.patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # Load best model
    model.load_state_dict(torch.load(ckpt_dir / "best_seer_model.pt", weights_only=True))
    return history


def train_mimic_model(
    model: MultiSourceSurvivalModel,
    train_dataset: torch.utils.data.Dataset,
    val_dataset: torch.utils.data.Dataset,
    config: TrainingConfig | None = None,
) -> TrainingHistory:
    """Train a MIMIC multi-modal survival model (Stage 2).

    Parameters
    ----------
    model : MultiSourceSurvivalModel
        Should be built with ``build_mimic_model()`` or ``build_fusion_model()``.
    train_dataset, val_dataset : Dataset
        Output of ``train_test_split_temporal(MimicSurvivalDataset(...))``.
    config : TrainingConfig
        Hyperparameters.

    Returns
    -------
    TrainingHistory with per-epoch metrics.
    """
    if config is None:
        config = TrainingConfig(batch_size=64, lr=5e-4, epochs=80)

    device = _get_device(config)
    model = model.to(device)

    cox_loss = CoxPHLoss()
    fairness_reg = FairnessRegularizer(
        n_groups=config.n_race_groups,
        metric=config.fairness_metric,
        weight=config.fairness_weight,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.epochs
    )

    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True,
        collate_fn=_collate_mimic, drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=config.batch_size, shuffle=False,
        collate_fn=_collate_mimic,
    )

    history = TrainingHistory()
    best_c = 0.0
    patience_counter = 0
    ckpt_dir = Path(config.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(config.epochs):
        model.train()
        epoch_loss = 0.0
        epoch_fair = 0.0
        n_batches = 0

        for batch in train_loader:
            x_tab = batch["x_mimic_tab"].to(device)
            x_seq = batch["x_seq"].to(device)
            mask = batch["seq_mask"].to(device)
            x_text = batch["x_text"].to(device)
            t = batch["time"].to(device)
            e = batch["event"].to(device)
            r = batch["race"].to(device)

            log_h = model(
                x_mimic_tab=x_tab, x_seq=x_seq,
                seq_mask=mask, x_text=x_text,
            )
            loss = cox_loss(log_h, t, e)
            fair_pen = fairness_reg(log_h, r)
            total_loss = loss + fair_pen

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            epoch_fair += fair_pen.item()
            n_batches += 1

        scheduler.step()

        avg_loss = epoch_loss / max(n_batches, 1)
        avg_fair = epoch_fair / max(n_batches, 1)
        history.train_loss.append(avg_loss)
        history.fairness_penalty.append(avg_fair)

        # Validation
        model.eval()
        val_risks, val_times, val_events = [], [], []

        with torch.no_grad():
            val_loss_sum = 0.0
            val_n = 0
            for batch in val_loader:
                x_tab = batch["x_mimic_tab"].to(device)
                x_seq = batch["x_seq"].to(device)
                mask = batch["seq_mask"].to(device)
                x_text = batch["x_text"].to(device)
                t = batch["time"].to(device)
                e = batch["event"].to(device)

                log_h = model(
                    x_mimic_tab=x_tab, x_seq=x_seq,
                    seq_mask=mask, x_text=x_text,
                )
                vl = cox_loss(log_h, t, e)
                val_loss_sum += vl.item()
                val_n += 1

                val_risks.append(log_h.cpu().numpy())
                val_times.append(t.cpu().numpy())
                val_events.append(e.cpu().numpy())

        history.val_loss.append(val_loss_sum / max(val_n, 1))

        all_risks = np.concatenate(val_risks)
        all_times = np.concatenate(val_times)
        all_events = np.concatenate(val_events)
        c_index = concordance_index_fast(all_risks, all_times, all_events)
        history.val_c_index.append(c_index)

        if c_index > best_c:
            best_c = c_index
            history.best_c_index = c_index
            history.best_epoch = epoch
            patience_counter = 0
            torch.save(model.state_dict(), ckpt_dir / "best_mimic_model.pt")
        else:
            patience_counter += 1

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"  Epoch {epoch+1:3d}/{config.epochs} | "
                f"Loss {avg_loss:.4f} | Fair {avg_fair:.4f} | "
                f"Val C-index {c_index:.4f} (best {best_c:.4f})"
            )

        if patience_counter >= config.patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(torch.load(ckpt_dir / "best_mimic_model.pt", weights_only=True))
    return history


def transfer_seer_weights(
    source_model: MultiSourceSurvivalModel,
    target_model: MultiSourceSurvivalModel,
    freeze_seer: bool = False,
) -> None:
    """Transfer SEER tower weights from pre-trained model to fusion model.

    Parameters
    ----------
    source_model : MultiSourceSurvivalModel
        Pre-trained SEER-only model.
    target_model : MultiSourceSurvivalModel
        Fusion model to receive weights.
    freeze_seer : bool
        If True, freeze the transferred SEER tower weights.
    """
    if not source_model.use_seer or not target_model.use_seer:
        raise ValueError("Both models must have a SEER tower for transfer.")

    target_model.seer_tower.load_state_dict(
        source_model.seer_tower.state_dict()
    )

    if freeze_seer:
        for param in target_model.seer_tower.parameters():
            param.requires_grad = False

    print("  Transferred SEER tower weights"
          + (" (frozen)" if freeze_seer else " (unfrozen)"))
