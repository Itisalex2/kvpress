# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
from dataclasses import dataclass, field

import torch
from torch import nn

from kvpress.presses.base_press import BasePress

logger = logging.getLogger(__name__)


@dataclass
class ScorerPress(BasePress):
    """
    Base class for score-based KV cache compression methods.

    This class assigns scores to key-value pairs and prune those with the lowest scores.
    Subclasses then implement the `score` method to define how importance is calculated.

    Parameters
    ----------
    compression_ratio : float, default=0.0
        Fraction of key-value pairs to remove during compression.
    get_attention_mse : bool, default=False
        Whether to compute and store the mean squared error (MSE) of attention outputs
        after compression, compared to the original attention outputs. This is useful for analyzing
        the impact of compression on attention quality.
    """

    compression_ratio: float = 0.0
    get_attention_mse: bool = False
    position_scores_by_layer: dict[int, torch.Tensor] = field(
        default_factory=dict, init=False
    )
    kept_indices_by_layer: dict[int, torch.Tensor] = field(
        default_factory=dict, init=False
    )
    context_mse_by_layer: dict[int, torch.Tensor] = field(
        default_factory=dict, init=False
    )

    force_keep_global: torch.Tensor | None = None

    def clear_analysis(self):
        self.position_scores_by_layer.clear()
        self.kept_indices_by_layer.clear()
        self.context_mse_by_layer.clear()

    def __post_init__(self):
        assert 0 <= self.compression_ratio < 1, (
            "Compression ratio must be between 0 and 1"
        )

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:
        """
        Compute importance scores for each key-value pair.

        This method must be implemented by subclasses to define how the importance
        of each token position is calculated. Higher scores indicate more important
        tokens that should be kept during compression.

        Parameters
        ----------
        module : nn.Module
            The transformer attention layer where scoring is applied.
        hidden_states : torch.Tensor
            Input embeddings with shape (batch_size, seq_len, hidden_dim).
        keys : torch.Tensor
            Key tensors with shape (batch_size, num_kv_heads, seq_len, head_dim).
        values : torch.Tensor
            Value tensors with shape (batch_size, num_kv_heads, seq_len, head_dim).
        attentions : torch.Tensor
            Attention weights with shape (batch_size, num_heads, seq_len, seq_len).
            May be None if not computed or needed by the scoring method.
        kwargs : dict
            Additional arguments from the forward pass, including cache and position info.

        Returns
        -------
        torch.Tensor
            Importance scores with shape (batch_size, num_kv_heads, seq_len).
            Higher scores indicate more important tokens. The tokens with the
            lowest scores will be pruned during compression.
        """
        raise NotImplementedError

    def compress(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs: dict,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.compression_ratio == 0:
            return keys, values

        # Compute scores
        scores = self.score(
            module, hidden_states, keys, values, attentions, kwargs
        )  # (B, KV_H, L)

        # Get indices of KV pairs with the lowest scores
        q_len = hidden_states.shape[1]
        n_kept = int(q_len * (1 - self.compression_ratio))
        indices = scores.topk(n_kept, dim=-1).indices  # (B, KV_H, n_kept)

        # Force keeping certain indices, if specified
        if self.force_keep_global is not None and self.force_keep_global.numel() > 0:
            forced = self.force_keep_global.to(
                scores.device, dtype=torch.long, non_blocking=True
            ).unique()
            if not (forced.ge(0).all() and forced.lt(q_len).all()):
                raise ValueError(
                    f"Forced indices {forced.tolist()} out of range [0, {q_len})"
                )
            forced_cnt = int(forced.numel())

            if forced_cnt > n_kept:
                raise ValueError(
                    f"Cannot force keep {forced_cnt} positions when only {n_kept} can be kept"
                )

            k_rest = n_kept - forced_cnt
            # Exclude forced positions from the candidate pool
            masked_scores = scores.clone()
            masked_scores[:, :, forced] = float("-inf")
            rest = masked_scores.topk(k_rest, dim=-1).indices  # (B, H, k_rest)
            forced_expanded = forced.view(1, 1, -1).expand(
                scores.size(0), scores.size(1), -1
            )  # (B, H, forced_cnt)
            indices = torch.cat([forced_expanded, rest], dim=-1)  # (B, H, n_kept)

        if self.get_attention_mse:
            assert attentions is not None, (
                "Attentions are required to compute attention MSE"
            )
            B, H, seq_len, _ = attentions.shape
            KV_H = keys.shape[1]

            groups = H // KV_H
            idx_full = indices.unsqueeze(2).repeat(
                1, 1, groups, 1
            )  # (B, KV_H, groups, n_kept)
            idx_full = idx_full.view(B, H, n_kept)
            keep_mask = torch.zeros(B, H, 1, seq_len, device=attentions.device)
            keep_mask.scatter_(
                dim=-1,
                index=idx_full.unsqueeze(2),
                src=torch.ones_like(idx_full.unsqueeze(2), dtype=attentions.dtype),
            )

            # Get ground truth attention
            head_dim = values.size(-1)
            V_full = values.unsqueeze(2).repeat(
                1, 1, groups, 1, 1
            )  # (B, KV_H, groups, seq_len, head_dim)
            V_full = V_full.view(B, H, seq_len, head_dim)
            ground_truth_attn = torch.matmul(
                attentions, V_full
            )  # (B, H, seq_len, head_dim)

            attention_pruned = attentions * keep_mask  # (B, H, seq_len, seq_len)
            attention_pruned = attention_pruned / (
                attention_pruned.sum(dim=-1, keepdim=True).clamp_min(1e-12)
            )  # Re-normalize

            pruned_attn = torch.matmul(
                attention_pruned, V_full
            )  # (B, H, seq_len, head_dim)

            # Last-query MSE per head (generation-relevant)
            mse = (
                (ground_truth_attn[:, :, -1, :] - pruned_attn[:, :, -1, :]) ** 2
            ).mean(dim=-1)  # (B, H)
            try:
                self.context_mse_by_layer[module.layer_idx] = mse.detach().cpu()
            except Exception:
                pass

        # Save raw per-position scores and kept indices for analysis
        try:
            self.position_scores_by_layer[module.layer_idx] = (
                scores.detach().cpu()
            )  # (B, H, L)
            self.kept_indices_by_layer[module.layer_idx] = (
                indices.detach().cpu()
            )  # (B, H, L') where L' is the pruned length
        except Exception:
            pass

        indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)

        # Prune keys and values
        keys = keys.gather(2, indices).contiguous()
        values = values.gather(2, indices).contiguous()

        return keys, values
