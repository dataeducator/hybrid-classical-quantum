"""Multi-source fusion model with DeepSurv survival head.

Data-lineage:
  Tower embeddings (SEER tab, MIMIC temporal, MIMIC text)
    -> Fusion MLP (concat or cross-attention)
    -> DeepSurv head -> log-hazard ratio (scalar per patient)

The model supports three operating modes:
  1. SEER-only:  tabular tower only (pre-training on 76K patients)
  2. MIMIC-only: temporal + text towers (+ optional tabular)
  3. Fusion:     all towers active (for patients with multi-source data,
                 or via transfer learning)

Reference:
  Katzman et al. (2018) DeepSurv
  Cheerla & Gevaert (2019) Deep learning with multimodal representation
    for pancancer prognosis prediction
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .towers import SEERTower, TemporalTower, TextTower


class SurvivalHead(nn.Module):
    """DeepSurv survival prediction head.

    Maps a fused embedding to a scalar log-hazard ratio.
    """

    def __init__(self, d_in: int = 64, d_hidden: int = 32,
                 dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(batch, d_in) -> (batch,) log-hazard."""
        return self.net(x).squeeze(-1)


class MultiSourceSurvivalModel(nn.Module):
    """Multi-tower survival model with optional fairness-aware training.

    Combines up to three modality-specific towers via concatenation
    and a fusion MLP, followed by a DeepSurv survival head.

    Parameters
    ----------
    seer_d_in : int
        Dimensionality of SEER tabular features.
    mimic_tab_d_in : int
        Dimensionality of MIMIC tabular features.
    n_labs : int
        Number of lab channels in temporal sequences.
    embed_dim : int
        Output dimension for each tower.
    max_seq_len : int
        Max time steps for temporal tower.
    dropout : float
        Dropout rate across all towers and heads.
    use_seer_tower : bool
        Whether to include the SEER tabular tower.
    use_temporal_tower : bool
        Whether to include the MIMIC temporal tower.
    use_text_tower : bool
        Whether to include the MIMIC text tower.
    use_mimic_tab : bool
        Whether to include a separate MIMIC tabular tower.
    """

    def __init__(
        self,
        seer_d_in: int = 15,
        mimic_tab_d_in: int = 8,
        n_labs: int = 7,
        embed_dim: int = 64,
        max_seq_len: int = 50,
        dropout: float = 0.3,
        use_seer_tower: bool = True,
        use_temporal_tower: bool = True,
        use_text_tower: bool = True,
        use_mimic_tab: bool = False,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.use_seer = use_seer_tower
        self.use_temporal = use_temporal_tower
        self.use_text = use_text_tower
        self.use_mimic_tab = use_mimic_tab

        # Build towers
        if use_seer_tower:
            self.seer_tower = SEERTower(
                d_in=seer_d_in, embed_dim=embed_dim, dropout=dropout
            )

        if use_mimic_tab:
            self.mimic_tab_tower = SEERTower(
                d_in=mimic_tab_d_in, embed_dim=embed_dim, dropout=dropout
            )

        if use_temporal_tower:
            self.temporal_tower = TemporalTower(
                n_labs=n_labs, embed_dim=embed_dim, dropout=dropout,
                max_seq_len=max_seq_len,
            )

        if use_text_tower:
            self.text_tower = TextTower(embed_dim=embed_dim, dropout=dropout)

        # Count active towers for fusion dimension
        n_towers = sum([use_seer_tower, use_temporal_tower,
                        use_text_tower, use_mimic_tab])
        fusion_dim = embed_dim * n_towers

        # Fusion MLP
        self.fusion = nn.Sequential(
            nn.Linear(fusion_dim, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Survival head
        self.survival_head = SurvivalHead(
            d_in=embed_dim, dropout=dropout
        )

    def forward(
        self,
        x_seer: torch.Tensor | None = None,
        x_mimic_tab: torch.Tensor | None = None,
        x_seq: torch.Tensor | None = None,
        seq_mask: torch.Tensor | None = None,
        x_text: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass through active towers and fusion.

        Parameters
        ----------
        x_seer : (batch, seer_d_in) or None
        x_mimic_tab : (batch, mimic_tab_d_in) or None
        x_seq : (batch, T, n_labs) or None
        seq_mask : (batch, T) or None
        x_text : (batch, 768) or None

        Returns
        -------
        (batch,) log-hazard ratios
        """
        embeddings: list[torch.Tensor] = []

        if self.use_seer and x_seer is not None:
            embeddings.append(self.seer_tower(x_seer))

        if self.use_mimic_tab and x_mimic_tab is not None:
            embeddings.append(self.mimic_tab_tower(x_mimic_tab))

        if self.use_temporal and x_seq is not None:
            embeddings.append(self.temporal_tower(x_seq, seq_mask))

        if self.use_text and x_text is not None:
            embeddings.append(self.text_tower(x_text))

        if not embeddings:
            raise ValueError("No tower inputs provided. At least one modality required.")

        fused = torch.cat(embeddings, dim=-1)
        h = self.fusion(fused)
        return self.survival_head(h)

    def get_embeddings(
        self,
        x_seer: torch.Tensor | None = None,
        x_mimic_tab: torch.Tensor | None = None,
        x_seq: torch.Tensor | None = None,
        seq_mask: torch.Tensor | None = None,
        x_text: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return fused embedding (before survival head) for downstream use."""
        embeddings: list[torch.Tensor] = []

        if self.use_seer and x_seer is not None:
            embeddings.append(self.seer_tower(x_seer))
        if self.use_mimic_tab and x_mimic_tab is not None:
            embeddings.append(self.mimic_tab_tower(x_mimic_tab))
        if self.use_temporal and x_seq is not None:
            embeddings.append(self.temporal_tower(x_seq, seq_mask))
        if self.use_text and x_text is not None:
            embeddings.append(self.text_tower(x_text))

        if not embeddings:
            raise ValueError("No tower inputs provided.")

        fused = torch.cat(embeddings, dim=-1)
        return self.fusion(fused)


def build_seer_only_model(d_in: int = 15, embed_dim: int = 64,
                          dropout: float = 0.3) -> MultiSourceSurvivalModel:
    """Factory: SEER-only model for pre-training on population data."""
    return MultiSourceSurvivalModel(
        seer_d_in=d_in,
        embed_dim=embed_dim,
        dropout=dropout,
        use_seer_tower=True,
        use_temporal_tower=False,
        use_text_tower=False,
        use_mimic_tab=False,
    )


def build_mimic_model(
    mimic_tab_d_in: int = 8,
    n_labs: int = 7,
    embed_dim: int = 64,
    max_seq_len: int = 50,
    dropout: float = 0.3,
) -> MultiSourceSurvivalModel:
    """Factory: MIMIC-only model (tabular + temporal + text)."""
    return MultiSourceSurvivalModel(
        mimic_tab_d_in=mimic_tab_d_in,
        n_labs=n_labs,
        embed_dim=embed_dim,
        max_seq_len=max_seq_len,
        dropout=dropout,
        use_seer_tower=False,
        use_temporal_tower=True,
        use_text_tower=True,
        use_mimic_tab=True,
    )


def build_fusion_model(
    seer_d_in: int = 15,
    n_labs: int = 7,
    embed_dim: int = 64,
    max_seq_len: int = 50,
    dropout: float = 0.3,
) -> MultiSourceSurvivalModel:
    """Factory: Full fusion model (SEER + temporal + text towers)."""
    return MultiSourceSurvivalModel(
        seer_d_in=seer_d_in,
        n_labs=n_labs,
        embed_dim=embed_dim,
        max_seq_len=max_seq_len,
        dropout=dropout,
        use_seer_tower=True,
        use_temporal_tower=True,
        use_text_tower=True,
        use_mimic_tab=False,
    )
