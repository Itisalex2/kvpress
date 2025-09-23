# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


from dataclasses import dataclass

import torch
from torch import nn

from kvpress.presses.streaming_llm_press import StreamingLLMPress


@dataclass
class StreamingLLMFairEvictionPress(StreamingLLMPress):
    defense_span: tuple[int, int] | None = None  # [start, end)
    sys_instr_span: tuple[int, int] | None = None  # [start, end)

    def score(
        self,
        module: nn.Module,
        hidden_states: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        attentions: torch.Tensor,
        kwargs,
    ) -> torch.Tensor:
        q_len = hidden_states.shape[1]
        assert q_len > self.n_sink, (
            f"Input should contain more tokens than n_sink={self.n_sink}"
        )
        assert self.defense_span is not None, (
            "defense_span must be set for StreamingLLMSystemPromptPress"
        )
        assert self.sys_instr_span is not None, (
            "sys_instr_span must be set for StreamingLLMSystemPromptPress"
        )

        scores = torch.zeros_like(keys[..., 0])

        # Set ramp scores for defense span
        defense_span_start, defense_span_end = self.defense_span
        assert 0 <= defense_span_start <= defense_span_end <= q_len, (
            f"Invalid defense span. Got {self.defense_span} for sequence length {q_len}."
        )
        defense_ramp = torch.linspace(
            0.1, 0.9, defense_span_end - defense_span_start
        ).to(scores.device)
        scores[:, :, defense_span_start:defense_span_end] = defense_ramp.view(
            1, 1, defense_span_end - defense_span_start
        )

        # Set ramp scores for system instruction span
        sys_instr_span_start, sys_instr_span_end = self.sys_instr_span
        assert 0 <= sys_instr_span_start <= sys_instr_span_end <= q_len, (
            f"Invalid system instruction span. Got {self.sys_instr_span} for sequence length {q_len}."
        )
        sys_instr_ramp = torch.linspace(
            0.1, 0.9, sys_instr_span_end - sys_instr_span_start
        ).to(scores.device)
        scores[:, :, sys_instr_span_start:sys_instr_span_end] = sys_instr_ramp.view(
            1, 1, sys_instr_span_end - sys_instr_span_start
        )

        # Set ramp scores for tail tokens that aren't in defense or sys instr spans
        tail_kept_index_start = max(self.n_sink, defense_span_end, sys_instr_span_end)
        if tail_kept_index_start < q_len:
            ramp = torch.linspace(0.91, 0.99, q_len - tail_kept_index_start).to(
                scores.device
            )
            scores[:, :, tail_kept_index_start:] = ramp.view(
                1, 1, q_len - tail_kept_index_start
            )

        # Set max scores for sink tokens
        scores[:, :, : self.n_sink] = 1.0

        return scores
