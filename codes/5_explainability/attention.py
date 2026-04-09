"""Attention weight extraction from the temporal transformer tower.

Data-lineage:
  Trained TemporalTower + lab sequence input
    -> per-layer, per-head attention matrices
    -> aggregated time-step importance scores
    -> lab-channel importance at each time step

Uses forward hooks to capture attention weights from PyTorch's
MultiheadAttention layers without modifying the tower code.

Reference:
  Vaswani et al. (2017) "Attention Is All You Need"
  Abnar & Zuidema (2020) "Quantifying Attention Flow in Transformers"
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn


@dataclass
class AttentionMaps:
    """Container for extracted attention weights."""
    # Per-layer attention: list of (n_heads, T+1, T+1) arrays
    layer_attentions: list[np.ndarray] = field(default_factory=list)
    # Aggregated CLS-to-sequence attention: (T,) importance per time step
    cls_attention: np.ndarray = field(default_factory=lambda: np.array([]))
    # Mean attention across heads and layers: (T+1, T+1)
    mean_attention: np.ndarray = field(default_factory=lambda: np.array([]))
    n_layers: int = 0
    n_heads: int = 0
    seq_len: int = 0


class AttentionExtractor:
    """Extract attention weights from a TemporalTower via forward hooks.

    Usage::

        extractor = AttentionExtractor(model.temporal_tower)
        output = model(x_seq=seq, seq_mask=mask)
        maps = extractor.get_attention_maps()
        extractor.remove_hooks()
    """

    def __init__(self, temporal_tower: nn.Module) -> None:
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self._attention_weights: list[torch.Tensor] = []
        self._register_hooks(temporal_tower)

    def _register_hooks(self, tower: nn.Module) -> None:
        """Register forward hooks on all MultiheadAttention layers."""
        for module in tower.modules():
            if isinstance(module, nn.MultiheadAttention):
                hook = module.register_forward_hook(self._capture_attention)
                self._hooks.append(hook)

    def _capture_attention(
        self,
        module: nn.Module,
        args: tuple,
        output: tuple,
    ) -> None:
        """Hook callback: capture attention weights from MHA output."""
        # nn.MultiheadAttention returns (attn_output, attn_weights)
        # attn_weights is None unless need_weights=True
        if len(output) >= 2 and output[1] is not None:
            self._attention_weights.append(output[1].detach())

    def clear(self) -> None:
        """Clear captured attention weights."""
        self._attention_weights.clear()

    def remove_hooks(self) -> None:
        """Remove all registered hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def get_attention_maps(self) -> AttentionMaps:
        """Return captured attention maps.

        Returns
        -------
        AttentionMaps with per-layer attention and CLS aggregation.
        """
        if not self._attention_weights:
            return AttentionMaps()

        maps = AttentionMaps()
        maps.n_layers = len(self._attention_weights)

        for attn in self._attention_weights:
            # attn shape: (batch, T+1, T+1) for averaged heads
            # or (batch * n_heads, T+1, T+1) for per-head
            arr = attn[0].cpu().numpy()  # take first batch item
            maps.layer_attentions.append(arr)

        # Mean attention across all layers
        all_attn = np.stack([a for a in maps.layer_attentions])
        # If 3D (n_heads, T+1, T+1), average over heads first
        if all_attn.ndim == 4:
            all_attn = all_attn.mean(axis=1)
        maps.mean_attention = all_attn.mean(axis=0)

        # CLS attention: row 0 = how CLS attends to each time step
        # Skip position 0 (CLS attending to itself)
        seq_len = maps.mean_attention.shape[0] - 1
        maps.seq_len = seq_len
        maps.cls_attention = maps.mean_attention[0, 1:]  # (T,)

        return maps


def enable_attention_weights(temporal_tower: nn.Module) -> None:
    """Patch MultiheadAttention modules to return attention weights.

    PyTorch's TransformerEncoderLayer calls self_attn with
    need_weights=False by default. This patches it to True.
    """
    for module in temporal_tower.modules():
        if isinstance(module, nn.TransformerEncoderLayer):
            # Monkey-patch the forward to request attention weights
            original_forward = module.forward

            def make_patched_forward(orig_fwd, layer):
                def patched_forward(src, src_mask=None, src_key_padding_mask=None,
                                    is_causal=False, **kwargs):
                    # Call self-attention with need_weights=True
                    x = src
                    if hasattr(layer, 'norm_first') and layer.norm_first:
                        x2 = layer.norm1(x)
                        x2, attn_w = layer.self_attn(
                            x2, x2, x2,
                            attn_mask=src_mask,
                            key_padding_mask=src_key_padding_mask,
                            need_weights=True,
                            average_attn_weights=True,
                        )
                        x = x + x2
                        x = x + layer._ff_block(layer.norm2(x))
                    else:
                        x2, attn_w = layer.self_attn(
                            x, x, x,
                            attn_mask=src_mask,
                            key_padding_mask=src_key_padding_mask,
                            need_weights=True,
                            average_attn_weights=True,
                        )
                        x = layer.norm1(x + x2)
                        x = layer.norm2(x + layer._ff_block(x))
                    return x
                return patched_forward

            module.forward = make_patched_forward(original_forward, module)


def extract_temporal_importance(
    model: nn.Module,
    x_seq: torch.Tensor,
    seq_mask: torch.Tensor | None = None,
    lab_names: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """Extract time-step and lab importance from temporal tower attention.

    Parameters
    ----------
    model : MultiSourceSurvivalModel or TemporalTower
    x_seq : (1, T, n_labs) or (T, n_labs) single sequence
    seq_mask : (1, T) or (T,) mask
    lab_names : names for each lab channel

    Returns
    -------
    Dict with keys:
      "time_importance": (T,) importance per time step from CLS attention
      "lab_importance": (n_labs,) importance per lab (from IG)
      "attention_maps": AttentionMaps object
    """
    if lab_names is None:
        lab_names = [f"lab_{i}" for i in range(x_seq.shape[-1])]

    # Ensure batch dimension
    if x_seq.ndim == 2:
        x_seq = x_seq.unsqueeze(0)
    if seq_mask is not None and seq_mask.ndim == 1:
        seq_mask = seq_mask.unsqueeze(0)

    # Find temporal tower
    tower = None
    if hasattr(model, 'temporal_tower'):
        tower = model.temporal_tower
    elif isinstance(model, type) and 'TemporalTower' in type(model).__name__:
        tower = model

    if tower is None:
        # Fall back: use gradient-based importance
        return _gradient_temporal_importance(model, x_seq, seq_mask, lab_names)

    # Enable attention weight output and set up extractor
    enable_attention_weights(tower)
    extractor = AttentionExtractor(tower)

    model.eval()
    with torch.no_grad():
        if hasattr(model, 'temporal_tower'):
            model(x_seq=x_seq, seq_mask=seq_mask)
        else:
            model(x_seq, seq_mask)

    maps = extractor.get_attention_maps()
    extractor.remove_hooks()

    # Time-step importance from CLS attention
    time_importance = maps.cls_attention if len(maps.cls_attention) > 0 else np.zeros(x_seq.shape[1])

    # Lab importance: sum of absolute input values weighted by time importance
    x_np = x_seq[0].cpu().numpy()  # (T, n_labs)
    if len(time_importance) > 0 and len(time_importance) == x_np.shape[0]:
        lab_importance = np.abs(x_np * time_importance[:, None]).sum(axis=0)
    else:
        lab_importance = np.abs(x_np).mean(axis=0)

    return {
        "time_importance": time_importance,
        "lab_importance": lab_importance,
        "attention_maps": maps,
    }


def _gradient_temporal_importance(
    model: nn.Module,
    x_seq: torch.Tensor,
    seq_mask: torch.Tensor | None,
    lab_names: list[str],
    forward_kwargs: dict | None = None,
) -> dict[str, np.ndarray]:
    """Fallback: gradient-based temporal importance."""
    if forward_kwargs is None:
        forward_kwargs = {}
    model.eval()
    x = x_seq.clone().requires_grad_(True)

    output = model(x_seq=x, seq_mask=seq_mask, **forward_kwargs)
    output.sum().backward()

    grad = x.grad[0].cpu().numpy()  # (T, n_labs)
    time_importance = np.abs(grad).sum(axis=1)  # (T,)
    lab_importance = np.abs(grad).sum(axis=0)    # (n_labs,)

    # Normalize
    ti_sum = time_importance.sum()
    if ti_sum > 0:
        time_importance = time_importance / ti_sum
    li_sum = lab_importance.sum()
    if li_sum > 0:
        lab_importance = lab_importance / li_sum

    return {
        "time_importance": time_importance,
        "lab_importance": lab_importance,
        "attention_maps": AttentionMaps(),
    }
