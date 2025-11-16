# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
import torch
import math
from torch import nn

from kvpress.presses.streaming_llm_press import StreamingLLMPress


@dataclass
class StreamingLLMInterpolatePress(StreamingLLMPress):
    """ """

    defense_span: tuple[int, int] | None = None  # [start, end)
    sys_instr_span: tuple[int, int] | None = None  # [start, end)
    interpolation_lambda: float = 0.5  # 0.0 = default, 1.0 = fair

    def get_spans(self, q_len) -> dict[str, int]:
        # Unpack spans and validate bounds
        assert self.defense_span is not None, "defense_span must be set"
        assert self.sys_instr_span is not None, "sys_instr_span must be set"
        defense_span_start, defense_span_end = self.defense_span
        sys_instr_span_start, sys_instr_span_end = self.sys_instr_span

        for s, e, name in [
            (defense_span_start, defense_span_end, "defense_span"),
            (sys_instr_span_start, sys_instr_span_end, "sys_instr_span"),
        ]:
            assert 0 <= s <= e <= q_len, f"Invalid {name} {s, e} for q_len={q_len}"

        # Enforce adjacency & determine order (no overlap, exactly touching)
        # Accept either defense first or system-instruction first.
        if defense_span_end == sys_instr_span_start:
            defense_first = True
        elif sys_instr_span_end == defense_span_start:
            defense_first = False
        else:
            raise AssertionError(
                f"Spans must be adjacent and non-overlapping. "
                f"Got defense={self.defense_span}, sys_instr={self.sys_instr_span}"
            )

        # Expanded ownership:
        # - Earlier span owns [0, earlier_end)
        # - Later span owns [later_start, q_len)
        if defense_first:
            _, earlier_span_end = defense_span_start, defense_span_end
            later_span_start, _ = sys_instr_span_start, sys_instr_span_end
        else:
            _, earlier_span_end = (
                sys_instr_span_start,
                sys_instr_span_end,
            )
            later_span_start, _ = defense_span_start, defense_span_end

        return {
            "earlier_span_start": self.n_sink,
            "earlier_span_end": earlier_span_end,
            "later_span_start": later_span_start,
            "later_span_end": q_len,
        }

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

        q_len = hidden_states.shape[1]
        assert q_len > self.n_sink, (
            f"Input should contain more tokens than n_sink={self.n_sink}"
        )
        n_kept = int(q_len * (1 - self.compression_ratio))
        assert 0.0 <= self.interpolation_lambda <= 1.0

        spans = self.get_spans(q_len)
        earlier_span_start = spans["earlier_span_start"]
        earlier_span_end = spans["earlier_span_end"]
        later_span_start = spans["later_span_start"]
        later_span_end = spans["later_span_end"]

        scores = torch.zeros_like(keys[..., 0])  # (batch, heads, q_len)
        scores[:, :, : self.n_sink] = 1.0

        len_earlier = earlier_span_end - earlier_span_start
        len_later = later_span_end - later_span_start

        n_kept_available = n_kept - self.n_sink  # do not consider sink tokens

        # Fair allocation of kept tokens
        fair_kept_earlier = ((n_kept_available) * len_earlier) // (
            len_earlier + len_later
        )

        default_kept_later = min(len_later, n_kept_available)
        default_kept_earlier = n_kept_available - default_kept_later

        # Interpolation between fair and default kept tokens
        interpolate_kept_earlier = int(
            round(
                self.interpolation_lambda * fair_kept_earlier
                + (1 - self.interpolation_lambda) * default_kept_earlier
            )
        )
        interpolate_kept_later = n_kept_available - interpolate_kept_earlier

        scores[:, :, earlier_span_end - interpolate_kept_earlier : earlier_span_end] = (
            1.0
        )
        scores[:, :, later_span_end - interpolate_kept_later : later_span_end] = 1.0

        indices = scores.topk(n_kept, dim=-1).indices  # (B, KV_H, n_kept)

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
