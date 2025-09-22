# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from dataclasses import dataclass

import torch
from torch import nn

from kvpress.presses.snapkv_press import SnapKVPress


@dataclass
class SnapKVSystemPromptPress(SnapKVPress):
    defense_span: tuple[int, int] = (0, 0)

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
