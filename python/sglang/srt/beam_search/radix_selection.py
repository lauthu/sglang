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
"""Trie beam selection with the fused radix-select kernels
(SGLANG_ENABLE_BEAM_TRIE_RADIX_SELECTION).

The beams of one request are a group. The default path selects each group's next
beams with its own kernels. RadixBeamSelector selects for several groups at once:

1. The coordinator asks can_select for every group of a step. A group the kernels
   do not cover keeps the default path, in the same step.
2. It hands select the covered groups whose next selection is the same RadixStep.
   They run in one set of kernel launches, whatever their beam widths.
3. On a decode step that is not a group's last, the same launches remap the
   beams' KV slots.
"""

from __future__ import annotations

import logging
from typing import Dict, List, NamedTuple, Optional, Sequence, Tuple, Union

import torch

from sglang.kernels.ops.sampling.beam_radix_select import (
    MAX_CODEBOOK_SIZE,
    radix_select_trie_beams,
)
from sglang.srt.beam_search.beam_group import BeamGroup
from sglang.srt.beam_search.joint_select import FinalSelect, SelectResult
from sglang.srt.beam_search.trie_constraint import BeamTrieConstraint

logger = logging.getLogger(__name__)


class RadixStep(NamedTuple):
    """What the kernels compile and launch for. Groups whose next selections are
    equal RadixSteps are selected in the same launches."""

    depth: int  # codes every beam has so far; 0 for the first selection
    final: bool  # the group's last selection
    kv_len: int  # decode KV slots per beam to remap; 0 on first and final steps


class RadixGroupResult(NamedTuple):
    selection: Union[SelectResult, FinalSelect]
    # [beam_width, depth + 1]; None on a final step.
    prefix_codes: Optional[torch.Tensor]
    # (old, new) decode slots for the deferred orphan search; None without a remap.
    orphans: Optional[Tuple[torch.Tensor, torch.Tensor]]


class RadixBeamSelector:
    """Selects the next beams of trie-constrained requests with the radix-select
    kernels.

    Covered:

    - The first selection, from the prefill's full-vocabulary logits.
    - The selections of decode steps whose forward returns the compact trie
      head's logits with their normalizers.
    - Any beam width. Requests of different widths share the launches.
    - Requests with stop tokens. Admission keeps stop tokens outside the SID
      codebooks, so no selected beam finishes on one.

    Not covered, the default path selects instead:

    - A codebook above MAX_CODEBOOK_SIZE codes: every selection.
    - A trie level stored sparse: the selection that reads it.
    - A decode step whose logits come from the dense LM head, which carry no
      normalizer. The coordinator checks this before it asks.
    """

    def __init__(self, trie: BeamTrieConstraint) -> None:
        self._trie = trie
        codebook_size = trie.config.codebook_size
        self._codebook_fits = codebook_size <= MAX_CODEBOOK_SIZE
        # Per depth: [codebook_size**depth, codebook_size] bool valid children of
        # each prefix; None for a level the trie stores sparse.
        self._child_masks = [
            trie.root_mask.view(1, codebook_size),
            trie.level1_masks,
            trie.level2_masks,
        ]
        # Per beam width: the SelectResult fields of a step in which no beam finishes.
        self._no_finish_fields: Dict[int, Tuple[torch.Tensor, ...]] = {}
        self._warn_if_not_covered()

    def _warn_if_not_covered(self) -> None:
        codebook_size = self._trie.config.codebook_size
        sparse_depths = [
            depth for depth, masks in enumerate(self._child_masks) if masks is None
        ]
        if not self._codebook_fits:
            logger.warning(
                "SGLANG_ENABLE_BEAM_TRIE_RADIX_SELECTION has no effect: the SID "
                "codebook has %d codes and the radix-select kernels take at most %d. "
                "Every beam selection keeps the default path.",
                codebook_size,
                MAX_CODEBOOK_SIZE,
            )
        elif sparse_depths:
            logger.warning(
                "SGLANG_ENABLE_BEAM_TRIE_RADIX_SELECTION: with %d codes per codebook "
                "the trie stores level %s sparse, which the radix-select kernels do "
                "not read. A request's selection at that level keeps the default path.",
                codebook_size,
                ", ".join(str(depth) for depth in sparse_depths),
            )

    def can_select(self, group: BeamGroup) -> bool:
        """Whether the kernels cover the group's next selection."""
        return (
            self._codebook_fits and self._child_masks[group.num_generated] is not None
        )

    def next_step(self, group: BeamGroup) -> RadixStep:
        depth = group.num_generated
        final = group.next_step_is_final()
        kv_len = 0
        if depth > 0 and not final:
            kv_len = group.leader.kv.kv_committed_len - group.prompt_len
        return RadixStep(depth=depth, final=final, kv_len=kv_len)

    def prepare_prefill_inputs(
        self, leader_logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """The kernels' logits and normalizer for a first selection, from the
        leader's full-vocabulary prefill row [1, vocab]. They score a code as the
        default path does: its logit minus the row's log-sum-exp."""
        config = self._trie.config
        logits = leader_logits.float()
        # Per row, not per batch: logsumexp on CUDA is not batch-invariant.
        lse = torch.logsumexp(logits, dim=-1, keepdim=True)
        start = config.token_start_for_depth(0)
        code_logits = logits[:, start : start + config.codebook_size]
        # The kernels subtract the row max, then the log-sum-exp after it: a row
        # max of zero keeps the logit.
        normalizer = torch.nn.functional.pad(lse, (1, 0))
        return code_logits, normalizer

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
        all_rows: Optional[torch.Tensor] = None,
    ) -> Tuple[List[RadixGroupResult], torch.Tensor]:
        """Selects for `groups`, whose next selection is `step`.

        Returns each group's result, and every group's next tokens back to back.
        A step that remaps KV slots also needs `req_to_token` and `all_rows`, the
        groups' req_to_token rows back to back.
        """
        if len(groups) == 1:
            (group,) = groups
            cum_logprobs = group.frontier_cum_logprobs
            prefix_codes = group.trie_prefix_codes
        else:
            cum_logprobs = torch.cat([g.frontier_cum_logprobs for g in groups])
            prefix_codes = torch.cat([g.trie_prefix_codes for g in groups])
        config = self._trie.config
        remaps = step.kv_len > 0
        selected = radix_select_trie_beams(
            leader_logits=leader_logits,
            leader_normalizer=leader_normalizer,
            leader_rows=leader_rows,
            tail_logits=tail_logits,
            tail_normalizer=tail_normalizer,
            tail_starts=tail_starts,
            beam_widths=[g.beam_width for g in groups],
            cum_logprobs=cum_logprobs,
            prefix_codes=prefix_codes,
            child_masks=self._child_masks[step.depth],
            codebook_size=config.codebook_size,
            token_start=config.token_start_for_depth(step.depth),
            final=step.final,
            req_to_token=req_to_token if remaps else None,
            kv_rows=all_rows if remaps else None,
            kv_starts=[g.prompt_len for g in groups] if remaps else None,
            kv_len=step.kv_len,
        )
        results = []
        for i, group in enumerate(groups):
            beams = slice(selected.offsets[i], selected.offsets[i + 1])
            tokens = selected.tokens[beams]
            parent_idx = selected.parent_idx[beams]
            scores = selected.cum_logprobs[beams]
            if step.final:
                selection = FinalSelect(
                    tokens=tokens, parent_idx=parent_idx, cum_logprobs=scores
                )
            else:
                selection = self._select_result(
                    tokens=tokens,
                    parent_idx=parent_idx,
                    cum_logprobs=scores,
                    beam_width=group.beam_width,
                )
            orphans = None
            if selected.old_kv is not None:
                frontier_rows = slice(
                    selected.row_offsets[i], selected.row_offsets[i + 1]
                )
                orphans = (selected.old_kv[frontier_rows], selected.new_kv[beams])
            results.append(
                RadixGroupResult(
                    selection=selection,
                    prefix_codes=None if step.final else selected.prefix_codes[beams],
                    orphans=orphans,
                )
            )
        return results, selected.tokens

    def _select_result(
        self,
        *,
        tokens: torch.Tensor,
        parent_idx: torch.Tensor,
        cum_logprobs: torch.Tensor,
        beam_width: int,
    ) -> SelectResult:
        # What joint_select returns for a trie group: its stop tokens lie outside the SID
        # codebooks, so every selected beam survives and none finishes.
        if beam_width not in self._no_finish_fields:
            device = tokens.device
            self._no_finish_fields[beam_width] = (
                torch.full((), beam_width, dtype=torch.int64, device=device),
                torch.zeros(beam_width, dtype=torch.int64, device=device),
                torch.zeros(beam_width, dtype=torch.float32, device=device),
                torch.zeros((), dtype=torch.int64, device=device),
            )
        num_survivors, fin_ids, fin_scores, num_finished = self._no_finish_fields[
            beam_width
        ]
        return SelectResult(
            next_tokens=tokens,
            parent_idx=parent_idx,
            new_cum_logprobs=cum_logprobs,
            num_survivors=num_survivors,
            fin_tokens=fin_ids,
            fin_parent_idx=fin_ids,
            fin_cum_logprobs=fin_scores,
            num_finished=num_finished,
        )
