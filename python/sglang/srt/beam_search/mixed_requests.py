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
"""Ordinary requests in beam decode steps (SGLANG_BEAM_DECODE_GRAPH_LEVEL=3).

An ordinary request that decodes next to beam requests joins the beam step as a
one-row group: shared-context attention reads its whole KV as the group's
prompt, and the trie output head keeps its full-vocabulary logits for the
regular sampler. In exact mode the head computes every row's full-vocabulary
logits anyway, for its normalizer, so keeping them costs stores, not GEMMs.

Ordinary requests are base rows, one per request, so the head keeps rows
[0, requests). The sampler also draws a token from each beam leader's kept row,
as in any step with ordinary requests, and beam selection overwrites it.

A step with more requests than SGLANG_BEAM_DECODE_GRAPH_MAX_MIXED_REQUESTS runs
the ordinary decode path, as at lower levels, and is counted in the log.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import msgspec
import torch

from sglang.srt.beam_search.shared_context_ragged import decode_graph_level
from sglang.srt.environ import BeamDecodeGraphLevel, envs

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch

logger = logging.getLogger(__name__)


class BeamDenseRows(msgspec.Struct):
    """Rows [0, num_rows) of a decode step whose full-vocabulary logits the trie
    head keeps. A step's own value carries only num_rows; a captured graph's also
    holds the buffers it writes, and num_rows is their capacity."""

    num_rows: int
    # int32 [1]: rows to keep in this forward; 0 keeps none.
    count: Optional[torch.Tensor] = None
    # fp32 [num_rows, vocab].
    logits: Optional[torch.Tensor] = None

    def with_buffers(self, vocab_size: int, device: torch.device) -> BeamDenseRows:
        """The buffers the head writes: this value when it holds a captured graph's,
        else one with new buffers that keep this step's rows."""
        if self.logits is not None:
            if self.logits.shape[1] != vocab_size:
                raise ValueError(
                    f"Kept logits have {self.logits.shape[1]} columns, the "
                    f"vocabulary {vocab_size}"
                )
            return self
        return BeamDenseRows(
            num_rows=self.num_rows,
            count=torch.full((1,), self.num_rows, dtype=torch.int32, device=device),
            logits=torch.empty(
                (self.num_rows, vocab_size), dtype=torch.float32, device=device
            ),
        )


def ordinary_requests_enabled() -> bool:
    return decode_graph_level() >= BeamDecodeGraphLevel.MIXED_REQUESTS


def max_mixed_requests() -> int:
    field = envs.SGLANG_BEAM_DECODE_GRAPH_MAX_MIXED_REQUESTS
    value = field.get()
    if value <= 0:
        raise ValueError(f"{field.name} must be positive, got {value}")
    return value


def has_ordinary_request(batch: ScheduleBatch) -> bool:
    return any(req.beam_group is None for req in batch.reqs)


def admits(batch: ScheduleBatch) -> bool:
    """Whether every request of a decode step may run the beam path: beam requests
    always; ordinary ones at level 3, in a step of at most
    SGLANG_BEAM_DECODE_GRAPH_MAX_MIXED_REQUESTS requests."""
    if not has_ordinary_request(batch):
        return True
    return ordinary_requests_enabled() and len(batch.reqs) <= max_mixed_requests()


_overflow_steps = 0


def note_overflow(num_requests: int) -> None:
    """Count a step that held too many requests to run as a beam step."""
    global _overflow_steps
    _overflow_steps += 1
    # Powers of two keep the log short under sustained overflow.
    if _overflow_steps & (_overflow_steps - 1) == 0:
        logger.warning(
            "%d decode steps with beam and ordinary requests ran the ordinary decode "
            "path: they held more than SGLANG_BEAM_DECODE_GRAPH_MAX_MIXED_REQUESTS=%d "
            "requests (latest: %d)",
            _overflow_steps,
            max_mixed_requests(),
            num_requests,
        )


def overflow_steps() -> int:
    return _overflow_steps


def build_dense_rows(batch: ScheduleBatch) -> Optional[BeamDenseRows]:
    """The rows a beam decode step keeps: every request's, when one is ordinary."""
    if not has_ordinary_request(batch) or not ordinary_requests_enabled():
        return None
    return BeamDenseRows(num_rows=len(batch.reqs))


def allocate_capture_dense_rows(
    max_rows: int, vocab_size: int, device
) -> BeamDenseRows:
    """One buffer for every captured row count: only one graph replays at a time."""
    num_rows = min(max_rows, max_mixed_requests())
    return BeamDenseRows(
        num_rows=num_rows,
        count=torch.zeros(1, dtype=torch.int32, device=device),
        logits=torch.empty((num_rows, vocab_size), dtype=torch.float32, device=device),
    )


def stage_dense_rows(target: BeamDenseRows, step: Optional[BeamDenseRows]) -> None:
    """Set a captured graph's kept-row count to this step's, 0 for a beam-only step."""
    num_rows = 0 if step is None else step.num_rows
    if num_rows > target.num_rows:
        raise ValueError(
            f"{num_rows} kept rows exceed the captured buffer's {target.num_rows}"
        )
    target.count.fill_(num_rows)
