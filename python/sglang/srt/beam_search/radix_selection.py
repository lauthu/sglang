# Copyright 2026 SGLang Team
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Opt-in trie beam selection with the radix-select kernels
(SGLANG_ENABLE_BEAM_TRIE_RADIX_SELECTION).

Every covered group of a step that shares a RadixStep runs in the same launches,
whatever its beam width; a decode step's launches also apply the beam KV remap.
Groups outside the coverage keep the torch path.
"""

from __future__ import annotations

from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import torch

from sglang.kernels.ops.sampling.beam_radix_select import radix_select_trie_beams
from sglang.srt.beam_search.beam_group import BeamGroup
from sglang.srt.beam_search.joint_select import FinalSelect, SelectResult
from sglang.srt.beam_search.trie_constraint import BeamTrieConstraint

# The kernels hold one frontier row's codes in one program.
_MAX_CODEBOOK_SIZE = 4096


class RadixStep(NamedTuple):
    """Groups with equal keys share one launch set."""

    depth: int
    final: bool
    kv_len: int  # decode slots per beam to remap; 0 on first and final steps


class RadixGroupResult(NamedTuple):
    selection: Union[SelectResult, FinalSelect]
    # [beam_width, depth + 1]; None on a final step.
    prefix_codes: Optional[torch.Tensor]
    # (old, new) decode slots for the deferred orphan search; None without a remap.
    orphans: Optional[Tuple[torch.Tensor, torch.Tensor]]


class RadixBeamSelector:
    def __init__(self, trie: BeamTrieConstraint) -> None:
        self._trie = trie
        size = trie.config.codebook_size
        # Dense [codebook**depth, codebook] child masks per depth. A codebook too large for
        # a dense level 2 keeps that step on the torch path.
        self._tables = [
            trie.root_mask.view(1, size),
            trie.level1_masks,
            trie.level2_masks,
        ]
        # Per beam width: the SelectResult fields of a step in which no beam finishes.
        self._constants: Dict[int, Tuple[torch.Tensor, ...]] = {}

    def step_of(self, group: BeamGroup, *, num_rows: int) -> Optional[RadixStep]:
        """The launch key when the kernels cover the group's next selection, else None."""
        depth = group.num_generated
        if (
            self._trie.config.codebook_size > _MAX_CODEBOOK_SIZE
            or self._tables[depth] is None
            or num_rows != (1 if depth == 0 else group.beam_width)
        ):
            return None
        final = group.next_step_is_final()
        kv_len = 0
        if not final and depth > 0:
            kv_len = group.leader.kv.kv_committed_len - group.prompt_len
        return RadixStep(depth, final, kv_len)

    def first_step_inputs(
        self, piece: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The leader's code logits and normalizer from its full-vocabulary prefill row,
        scored as the torch path does: logit minus the row's log-sum-exp."""
        config = self._trie.config
        x = piece.float()
        # Per row, not per batch: logsumexp on CUDA is not batch-invariant.
        lse = torch.logsumexp(x, dim=-1, keepdim=True)
        start = config.token_start_for_depth(0)
        codes = x[:, start : start + config.codebook_size]
        # The kernel subtracts the row max, then the log-sum-exp after it: zero keeps the logit.
        normalizer = torch.nn.functional.pad(lse, (1, 0))
        return codes, normalizer

    def select(
        self,
        step: RadixStep,
        groups: Sequence[BeamGroup],
        *,
        leader_logits: torch.Tensor,
        leader_normalizer: torch.Tensor,
        leader_rows: Sequence[int],
        tail_logits: Optional[torch.Tensor] = None,
        tail_normalizer: Optional[torch.Tensor] = None,
        tail_starts: Optional[Sequence[int]] = None,
        req_to_token: Optional[torch.Tensor] = None,
        rows: Optional[torch.Tensor] = None,
    ) -> Tuple[List[RadixGroupResult], torch.Tensor]:
        """Each group's selection, and every group's next tokens back to back.
        `rows` are the groups' req_to_token rows back to back, for a step that remaps."""
        depth, final, kv_len = step
        if len(groups) == 1:
            (group,) = groups
            cum, prefix = group.frontier_cum_logprobs, group.trie_prefix_codes
        else:
            cum = torch.cat([g.frontier_cum_logprobs for g in groups])
            prefix = torch.cat([g.trie_prefix_codes for g in groups])
        config = self._trie.config
        out = radix_select_trie_beams(
            leader_logits=leader_logits,
            leader_normalizer=leader_normalizer,
            leader_rows=leader_rows,
            tail_logits=tail_logits,
            tail_normalizer=tail_normalizer,
            tail_starts=tail_starts,
            beam_widths=[g.beam_width for g in groups],
            cum_logprobs=cum,
            prefix_codes=prefix,
            child_masks=self._tables[depth],
            codebook_size=config.codebook_size,
            token_start=config.token_start_for_depth(depth),
            final=final,
            req_to_token=req_to_token if kv_len else None,
            kv_rows=rows if kv_len else None,
            kv_starts=[g.prompt_len for g in groups] if kv_len else None,
            kv_len=kv_len,
        )
        results = []
        for i, group in enumerate(groups):
            beams = slice(out.offsets[i], out.offsets[i + 1])
            tokens, parents = out.tokens[beams], out.parent_idx[beams]
            scores = out.cum_logprobs[beams]
            if final:
                selection = FinalSelect(
                    tokens=tokens, parent_idx=parents, cum_logprobs=scores
                )
            else:
                selection = self._select_result(
                    tokens, parents, scores, group.beam_width
                )
            orphans = None
            if out.old_kv is not None:
                frontier = slice(out.row_offsets[i], out.row_offsets[i + 1])
                orphans = (out.old_kv[frontier], out.new_kv[beams])
            results.append(
                RadixGroupResult(
                    selection=selection,
                    prefix_codes=None if final else out.prefix_codes[beams],
                    orphans=orphans,
                )
            )
        return results, out.tokens

    def _select_result(self, tokens, parents, scores, width: int) -> SelectResult:
        # What joint_select returns for a trie group: its stop tokens lie outside the SID
        # codebooks, so every selected beam survives and none finishes.
        if width not in self._constants:
            device = tokens.device
            self._constants[width] = (
                torch.full((), width, dtype=torch.int64, device=device),
                torch.zeros(width, dtype=torch.int64, device=device),
                torch.zeros(width, dtype=torch.float32, device=device),
                torch.zeros((), dtype=torch.int64, device=device),
            )
        survivors, fin_ids, fin_scores, no_finished = self._constants[width]
        return SelectResult(
            next_tokens=tokens,
            parent_idx=parents,
            new_cum_logprobs=scores,
            num_survivors=survivors,
            fin_tokens=fin_ids,
            fin_parent_idx=fin_ids,
            fin_cum_logprobs=fin_scores,
            num_finished=no_finished,
        )
