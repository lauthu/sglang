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
"""Decode batches that mix beam and regular requests, on the beam path
(SGLANG_ENABLE_BEAM_DECODE_GRAPH_MIXED_BATCH).

A regular request that decodes next to beam requests joins the beam batch as a
one-row group:

- Shared-context attention reads its whole KV as the group's prompt.
- The trie output head keeps its full-vocabulary logits for the regular sampler.
  The head computes every row's full-vocabulary logits anyway, for its
  normalizer, so keeping them costs stores, not GEMMs.

Regular requests are base rows, one per request, so the head keeps rows
[0, requests). The sampler also draws a token from each beam leader's kept row,
as in any batch with regular requests, and beam selection overwrites it.

A batch with more requests than SGLANG_BEAM_DECODE_GRAPH_MIXED_BATCH_MAX_REQUESTS
runs the ordinary decode path, as it does with the switch off, and is counted in
the log.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import msgspec
import torch

from sglang.srt.beam_search.decode_graph_settings import (
    max_mixed_batch_requests,
    mixed_batch_enabled,
)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import ScheduleBatch

logger = logging.getLogger(__name__)


class BeamDenseRows(msgspec.Struct):
    """Rows [0, num_rows) of a decode batch whose full-vocabulary logits the trie
    head keeps; a captured graph's value also holds the buffers, of that capacity."""

    num_rows: int
    # int32 [1]: rows to keep in this forward; 0 keeps none.
    count: Optional[torch.Tensor] = None
    # fp32 [num_rows, vocab].
    logits: Optional[torch.Tensor] = None

    def with_buffers(self, vocab_size: int, device: torch.device) -> BeamDenseRows:
        """The buffers the head writes: this value when it holds a captured graph's,
        else one with new buffers that keep this batch's rows."""
        if self.logits is not None:
            return self
        return BeamDenseRows(
            num_rows=self.num_rows,
            count=torch.full((1,), self.num_rows, dtype=torch.int32, device=device),
            logits=torch.empty(
                (self.num_rows, vocab_size), dtype=torch.float32, device=device
            ),
        )


def has_regular_request(batch: ScheduleBatch) -> bool:
    return any(req.beam_group is None for req in batch.reqs)


def runs_beam_path(batch: ScheduleBatch) -> bool:
    """Whether every request of a decode batch may run the beam path: a regular
    one only with the switch on, in a batch of at most the request cap."""
    if not has_regular_request(batch):
        return True
    return mixed_batch_enabled() and len(batch.reqs) <= max_mixed_batch_requests()


_num_oversized_batches = 0


def count_oversized_batch(num_requests: int) -> None:
    """Count a mixed batch that held too many requests to run the beam path."""
    global _num_oversized_batches
    _num_oversized_batches += 1
    # Powers of two keep the log short when oversized batches keep coming.
    if _num_oversized_batches & (_num_oversized_batches - 1) == 0:
        logger.warning(
            "%d decode batches with beam and regular requests ran the ordinary "
            "decode path: they held more than "
            "SGLANG_BEAM_DECODE_GRAPH_MIXED_BATCH_MAX_REQUESTS=%d requests "
            "(latest: %d)",
            _num_oversized_batches,
            max_mixed_batch_requests(),
            num_requests,
        )


def num_oversized_batches() -> int:
    return _num_oversized_batches


def build_dense_rows(batch: ScheduleBatch) -> Optional[BeamDenseRows]:
    """The rows a beam decode batch keeps: every request's, when one is regular."""
    if not has_regular_request(batch) or not mixed_batch_enabled():
        return None
    return BeamDenseRows(num_rows=len(batch.reqs))


def allocate_capture_dense_rows(
    max_rows: int, vocab_size: int, device
) -> BeamDenseRows:
    """One buffer for every captured row count: only one graph replays at a time."""
    num_rows = min(max_rows, max_mixed_batch_requests())
    return BeamDenseRows(
        num_rows=num_rows,
        count=torch.zeros(1, dtype=torch.int32, device=device),
        logits=torch.empty((num_rows, vocab_size), dtype=torch.float32, device=device),
    )
