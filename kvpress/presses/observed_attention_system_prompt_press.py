# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


import logging
from dataclasses import dataclass, field

import torch
from torch import nn

from kvpress.presses.scorer_press import ScorerPress

logger = logging.getLogger(__name__)


@dataclass
class ObservedAttentionSystemPromptPress(ScorerPress):
    """
    System prompt-aware KV cache compression.
    Computes importance scores based on actual attention weights observed during
    forward pass. Score for each key-value pair is the average attention weight
    it receives from all query tokens.
    Requires: output_attentions=True and attn_implementation="eager".
    Related to H2O (https://arxiv.org/abs/2306.14048).
    Modification:
    Ensures that the system prompt and defense are proportionally retained
    during compression, regardless of their computed importance scores.
    This can help with balance prompt leakage and instruction following.

    Requires a defense span to be specified.
    (start, end) indices. The span is inclusive of start and exclusive of end.
    """

    compression_ratio: float = 0.0
    output_attentions: bool = False
    defense_span: tuple[int, int] = (0, 0)
    position_scores_by_layer: dict[int, torch.Tensor] = field(
        default_factory=dict, init=False
    )
    kept_indices_by_layer: dict[int, torch.Tensor] = field(
        default_factory=dict, init=False
    )

    def clear_analysis(self):
        self.position_scores_by_layer.clear()
        self.kept_indices_by_layer.clear()

    def __post_init__(self):
        if not self.output_attentions:
            logger.warning(
                "Model will not return attentions in its output to save memory. "
                "Set output_attentions=True if attentions are needed in the output."
            )
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
        assert attentions is not None, (
            'Set output_attentions=True and attn_implementation="eager" to use this hook'
        )
        scores = attentions.sum(2)
        bsz, num_key_value_heads, n_tokens, _ = keys.shape
        n_tokens_in_sum = torch.arange(n_tokens, 0, -1).to(
            attentions.device, attentions.dtype
        )
        scores = scores / n_tokens_in_sum
        scores = scores.view(bsz, num_key_value_heads, -1, n_tokens).mean(2)
        return scores

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
        scores = self.score(module, hidden_states, keys, values, attentions, kwargs)

        # Get indices of KV pairs with the lowest scores
        q_len = hidden_states.shape[1]
        n_kept = int(q_len * (1 - self.compression_ratio))
        defense_span_start, defense_span_end = self.defense_span
        assert 0 <= defense_span_start <= defense_span_end <= q_len, (
            f"Invalid defense span. Got {self.defense_span} for sequence length {q_len}."
        )
        defense_len = defense_span_end - defense_span_start
        defense_kept = int(defense_len / q_len * n_kept)
        rest_len = q_len - defense_len
        rest_kept = n_kept - defense_kept

        defense_scores = scores[:, :, defense_span_start:defense_span_end]
        if defense_kept > 0:
            defense_indices = defense_scores.topk(defense_kept, dim=-1).indices
        else:
            defense_indices = torch.empty(
                (scores.shape[0], scores.shape[1], 0),
                dtype=torch.long,
                device=scores.device,
            )
        defense_indices += defense_span_start

        rest_left_scores = scores[:, :, :defense_span_start]
        rest_left_kept = int(rest_kept * defense_span_start / rest_len)
        rest_right_scores = scores[:, :, defense_span_end:]
        rest_right_kept = rest_kept - rest_left_kept
        if rest_left_kept > 0:
            rest_left_indices = rest_left_scores.topk(rest_left_kept, dim=-1).indices
        else:
            rest_left_indices = torch.empty(
                (scores.shape[0], scores.shape[1], 0),
                dtype=torch.long,
                device=scores.device,
            )
        if rest_right_kept > 0:
            rest_right_indices = rest_right_scores.topk(rest_right_kept, dim=-1).indices
        else:
            rest_right_indices = torch.empty(
                (scores.shape[0], scores.shape[1], 0),
                dtype=torch.long,
                device=scores.device,
            )
        rest_right_indices += defense_span_end

        indices = torch.cat(
            [defense_indices, rest_left_indices, rest_right_indices], dim=-1
        )

        # Save raw per-position scores and kept indices for analysis
        try:
            self.position_scores_by_layer[module.layer_idx] = (  # type: ignore
                scores.detach().cpu()
            )  # (B, H, L)
            self.kept_indices_by_layer[module.layer_idx] = (  # type: ignore
                indices.detach().cpu()
            )  # (B, H, L') where L' is the pruned length
        except Exception:
            pass

        indices = indices.unsqueeze(-1).expand(-1, -1, -1, module.head_dim)  # type: ignore

        # Prune keys and values
        keys = keys.gather(2, indices).contiguous()
        values = values.gather(2, indices).contiguous()

        return keys, values

    def forward_hook(
        self, module: nn.Module, input: list[torch.Tensor], kwargs: dict, output: list
    ):
        output = super().forward_hook(module, input, kwargs, output)
        # attentions are needed as input for the hook, but unless the user wants to return them in the output,
        # we can remove them to save memory
        if not self.output_attentions:
            output = (output[0], None)  # type: ignore

        return output
