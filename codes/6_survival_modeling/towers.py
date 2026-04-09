"""Neural network tower modules for multi-source survival modeling.

Data-lineage:
  Feature tensors -> tower embeddings (64-dim each)
    SEERTower:     (N, d_features) -> (N, embed_dim)
    TemporalTower: (N, T, 7)      -> (N, embed_dim)
    TextTower:     (N, 768)        -> (N, embed_dim)

Architecture choices:
  - SEERTower: 2-layer MLP with residual connection
  - TemporalTower: Transformer encoder with learnable CLS token
  - TextTower: Linear projection from frozen ClinicalBERT space
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SEERTower(nn.Module):
    """Tabular feature encoder for SEER population-level data.

    Two-layer MLP with batch normalization and dropout.
    """

    def __init__(self, d_in: int = 15, d_hidden: int = 128,
                 embed_dim: int = 64, dropout: float = 0.3) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, d_hidden),
            nn.BatchNorm1d(d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_hidden),
            nn.BatchNorm1d(d_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, d_in) tabular features

        Returns
        -------
        (batch, embed_dim) embedding
        """
        return self.net(x)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for temporal sequences."""

    def __init__(self, d_model: int, max_len: int = 200) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TemporalTower(nn.Module):
    """Transformer encoder for MIMIC lab trajectories.

    Input: (batch, T, n_labs) lab values over T time steps.
    Output: (batch, embed_dim) temporal embedding via learnable CLS token.
    """

    def __init__(
        self,
        n_labs: int = 7,
        d_model: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        embed_dim: int = 64,
        dropout: float = 0.1,
        max_seq_len: int = 50,
    ) -> None:
        super().__init__()
        self.d_model = d_model

        # Project lab values to d_model
        self.input_proj = nn.Linear(n_labs, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len=max_seq_len + 1)

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        self.output_proj = nn.Linear(d_model, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)

    def forward(self, x: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, T, n_labs) lab sequences
        mask : (batch, T) bool mask, True = valid time step

        Returns
        -------
        (batch, embed_dim) temporal embedding
        """
        batch_size = x.size(0)

        # Project input and add positional encoding
        h = self.input_proj(x)  # (batch, T, d_model)

        # Prepend CLS token
        cls = self.cls_token.expand(batch_size, -1, -1)
        h = torch.cat([cls, h], dim=1)  # (batch, T+1, d_model)
        h = self.pos_enc(h)

        # Build attention mask: CLS always valid, then use provided mask
        if mask is not None:
            cls_mask = torch.ones(batch_size, 1, dtype=torch.bool, device=x.device)
            full_mask = torch.cat([cls_mask, mask], dim=1)
            # TransformerEncoder uses src_key_padding_mask where True = IGNORE
            padding_mask = ~full_mask
        else:
            padding_mask = None

        h = self.transformer(h, src_key_padding_mask=padding_mask)

        # CLS token output
        cls_out = h[:, 0]  # (batch, d_model)
        out = self.layer_norm(self.output_proj(cls_out))
        return out


class TextTower(nn.Module):
    """Projects frozen ClinicalBERT embeddings to survival embedding space.

    Input: (batch, 768) pre-computed ClinicalBERT [CLS] vectors.
    Output: (batch, embed_dim) text embedding.
    """

    def __init__(self, d_bert: int = 768, embed_dim: int = 64,
                 dropout: float = 0.2) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_bert, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : (batch, d_bert) pre-computed text embeddings

        Returns
        -------
        (batch, embed_dim) projected text embedding
        """
        return self.net(x)
