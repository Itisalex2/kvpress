# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
import torch
from torch import nn

from kvpress.presses.snapkv_press import SnapKVPress


@dataclass
class SnapKVLLMInterpolatePress(SnapKVPress):
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
            "earlier_span_start": 0,
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

        bsz, num_kv_heads, q_len, _ = keys.shape
        n_kept = int(q_len * (1 - self.compression_ratio))

        assert self.defense_span is not None, (
            "defense_span must be set for SnapKVLLMInterpolatePress"
        )
        defense_span_start, defense_span_end = self.defense_span
        assert 0 <= defense_span_start <= defense_span_end <= q_len, (
            f"Invalid defense span. Got {self.defense_span} for sequence length {q_len}."
        )

        spans = self.get_spans(q_len)
        earlier_span_start = spans["earlier_span_start"]
        earlier_span_end = spans["earlier_span_end"]
        later_span_start = spans["later_span_start"]
        later_span_end = spans["later_span_end"]

        scores = self.score(
            module=module,
            hidden_states=hidden_states,
            keys=keys,
            values=values,
            attentions=attentions,
            kwargs=kwargs,
        )  # (B, KV_H, L)

        default_indices = scores.topk(n_kept, dim=-1).indices  # (B, KV_H, n_kept)

        # Segment lengths
        len_earlier = earlier_span_end - earlier_span_start
        len_later = later_span_end - later_span_start
        assert len_earlier + len_later == q_len, (
            "Span partition must cover full sequence"
        )

        # Default kept counts per segment (B, KV_H)
        default_earlier_kept = (
            (default_indices >= earlier_span_start)
            & (default_indices < earlier_span_end)
        ).sum(dim=-1)
        default_later_kept = (
            (default_indices >= later_span_start) & (default_indices < later_span_end)
        ).sum(dim=-1)

        # "Fair" allocation: proportional to segment lengths
        fair_earlier = int(n_kept * len_earlier / (len_earlier + len_later))

        # Interpolated counts per (batch, head)
        interp_earlier = torch.empty_like(default_earlier_kept, dtype=torch.long)
        interp_later = torch.empty_like(default_later_kept, dtype=torch.long)

        for b in range(bsz):
            for h in range(num_kv_heads):
                default_earlier = int(default_earlier_kept[b, h].item())

                # Interpolate towards fair allocation
                earlier_raw = (
                    1.0 - self.interpolation_lambda
                ) * default_earlier + self.interpolation_lambda * fair_earlier
                earlier = int(round(earlier_raw))

                # Later is whatever remains
                later = n_kept - earlier

                interp_earlier[b, h] = earlier
                interp_later[b, h] = later

        # Select per (batch, head) using SnapKV scores
        final_indices = torch.empty(
            (bsz, num_kv_heads, n_kept),
            dtype=torch.long,
            device=scores.device,
        )

        for b in range(bsz):
            for h in range(num_kv_heads):
                keep_e = int(interp_earlier[b, h].item())
                keep_l = int(interp_later[b, h].item())

                # Earlier segment [earlier_span_start, earlier_span_end)
                if keep_e > 0 and len_earlier > 0:
                    earlier_scores = scores[
                        b, h, earlier_span_start:earlier_span_end
                    ]  # (len_earlier,)
                    topk_e = earlier_scores.topk(keep_e, dim=-1).indices
                    topk_e = topk_e + earlier_span_start
                    final_indices[b, h, 0:keep_e] = topk_e

                # Later segment [later_span_start, later_span_end)
                if keep_l > 0 and len_later > 0:
                    later_scores = scores[
                        b, h, later_span_start:later_span_end
                    ]  # (len_later,)
                    topk_l = later_scores.topk(keep_l, dim=-1).indices
                    topk_l = topk_l + later_span_start
                    final_indices[b, h, keep_e : keep_e + keep_l] = topk_l

        # Save raw per-position scores and kept indices for analysis
        try:
            self.position_scores_by_layer[module.layer_idx] = (  # type: ignore
                scores.detach().cpu()
            )  # (B, H, L)
            self.kept_indices_by_layer[module.layer_idx] = (  # type: ignore
                final_indices.detach().cpu()
            )  # (B, H, L') where L' is the pruned length
        except Exception:
            pass

        final_indices_expanded = final_indices.unsqueeze(-1).expand(
            -1,
            -1,
            -1,
            module.head_dim,  # type: ignore[attr-defined]
        )
        keys = keys.gather(2, final_indices_expanded).contiguous()
        values = values.gather(2, final_indices_expanded).contiguous()

        return keys, values
