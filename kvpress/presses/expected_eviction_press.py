# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
from dataclasses import dataclass

import torch
from torch import nn

from kvpress.presses.scorer_press import ScorerPress

logger = logging.getLogger(__name__)


@dataclass
class ExpectedEvictionPress(ScorerPress):
    """
    ExpectedEvictionPress: Attention-based KV cache compression by sampling from attention via Gumbel noise.

    Uses attention weights to sample key-value pairs for retention, favoring those with higher attention scores.
    Mimick sampling via the Gumbel-Max trick.

    Parameters
    ----------
    compression_ratio : float, default=0.0
        Fraction of key-value pairs to remove during compression.
    output_attentions : bool, default=False
        Whether to return attention weights in model output.
        Controls whether attention weights are included in output after compression.
        Attention weights are always needed internally for scoring but can be removed
        from output to save memory.
    same_noise_all_heads : bool, optional
        If True, the same Gumbel noise is applied across all attention heads. If False, different noise is applied per head.
        Default is True.
    """

    compression_ratio: float = 0.0
    output_attentions: bool = False
    same_noise_all_heads: bool = True

    def __post_init__(self):
        if not self.output_attentions:
            logger.warning(
                "Model will not return attentions in its output to save memory. "
                "Set output_attentions=True if attentions are needed in the output."
            )
        super().__post_init__()

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

        assert attentions is not None, (
            'Set output_attentions=True and attn_implementation="eager" to use this hook'
        )
        bsz, num_key_value_heads, q_len, _ = keys.shape
        num_heads = attentions.shape[1]

        assert num_heads % num_key_value_heads == 0, (
            "Number of attention heads must be divisible by number of key-value heads."
        )
        num_key_value_groups: int = num_heads // num_key_value_heads

        attn_weights = attentions[..., -1:, :]  # (bsz, num_heads, 1, seq_len)
        attn_weights = attn_weights.squeeze(2)  # (bsz, num_heads, seq_len)
        logged_attn_weights = torch.log(attn_weights)

        if self.same_noise_all_heads:
            uniform_noise = torch.rand(bsz, 1, q_len, device=keys.device)
        else:
            uniform_noise = torch.rand(bsz, num_heads, q_len, device=keys.device)
        gumbel_noise = -torch.log(-torch.log(uniform_noise))

        scores = logged_attn_weights + gumbel_noise  # (bsz, num_heads, seq_len)

        # Average per group (https://github.com/FasterDecoding/SnapKV/issues/22)
        scores = scores.view(bsz, num_key_value_heads, num_key_value_groups, q_len)
        scores = scores.mean(2)  # (bsz, num_kv_heads, seq_len)

        return scores

    def forward_hook(
        self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list
    ):
        output = super().forward_hook(module, input, kwargs, output)
        # attentions are needed as input for the hook, but unless the user wants to return them in the output,
        # we can remove them to save memory
        if not self.output_attentions:
            output = (output[0], None)  # type: ignore

        return output
