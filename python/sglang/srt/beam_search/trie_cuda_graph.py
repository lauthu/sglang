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
"""Decode CUDA graphs for all-beam trie batches.

A trie batch runs the compact trie head, and with shared-context attention its own
attention path, so it cannot replay an ordinary decode graph. The runner here
captures a second graph per size for it and replays that graph with the batch's
trie levels and shared-context metadata staged into the buffers the graph reads.

SGLANG_BEAM_DECODE_GRAPH_LEVEL sets which batches a shared-context trie graph
takes: at 0 the one configured beam width at exactly a captured size, at 2 any
mix of widths padded to a captured size (shared_context_ragged), at 3 also
batches with ordinary requests, whose full-vocabulary logits the graph keeps
(mixed_requests).
"""

from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, Optional

import msgspec
import torch

from sglang.srt.beam_search import mixed_requests, shared_context_attention
from sglang.srt.beam_search import shared_context_ragged as ragged
from sglang.srt.beam_search.decode_graph_level import (
    decode_graph_level,
    stream_k_selected,
)
from sglang.srt.beam_search.forward_inputs import BeamForwardInputs
from sglang.srt.beam_search.mixed_requests import BeamDenseRows
from sglang.srt.beam_search.shared_context_attention import (
    BeamAttentionBucket,
    BeamAttentionMetadata,
)
from sglang.srt.beam_search.shared_context_ragged import RaggedBeamAttentionMetadata
from sglang.srt.beam_search.trie_config import get_beam_trie_config
from sglang.srt.environ import BeamDecodeGraphLevel, envs
from sglang.srt.model_executor.runner.decode_cuda_graph_runner import (
    DecodeCudaGraphRunner,
)

if TYPE_CHECKING:
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)

# Graph key label of the trie graphs. The compact trie head is off under LoRA, so
# the label never meets a LoRA one.
TRIE_GRAPH_VARIANT = "beam_trie"


class SharedContextCapacity(msgspec.Struct, frozen=True):
    """What the shared-context trie graphs hold per beam group."""

    # The one beam width the graphs serve.
    beam_width: int
    # Prompt tokens: the model's context length, so every prompt fits.
    max_context: int
    # Decode steps a step attends: all generated tokens but the trie's last.
    max_decode: int


def _shared_context_capacity(model_runner: ModelRunner) -> SharedContextCapacity:
    model_config = model_runner.model_config
    return SharedContextCapacity(
        beam_width=envs.SGLANG_BEAM_SHARED_CONTEXT_WIDTH.get(),
        max_context=model_config.context_len,
        max_decode=get_beam_trie_config(model_config).num_codebooks - 1,
    )


def _trie_graph_inputs(
    model_runner: ModelRunner, *, max_rows: int, device
) -> TrieGraphInputs:
    """The static inputs of this runner's trie graphs, for its decode graph level."""
    level = decode_graph_level()
    shared = (
        shared_context_attention.enabled()
        and shared_context_attention.supports_shared_context(model_runner)
    )
    width_free = shared and level >= BeamDecodeGraphLevel.MIXED_WIDTHS
    keeps_dense_rows = level >= BeamDecodeGraphLevel.MIXED_REQUESTS
    return TrieGraphInputs(
        max_rows=max_rows,
        device=device,
        shared_context=(
            _shared_context_capacity(model_runner)
            if shared and not width_free
            else None
        ),
        ragged_layout=(
            ragged.allocate_capture_layout(
                max_rows=max_rows,
                device=device,
                req_to_token=model_runner.req_to_token_pool.req_to_token,
                one_row_groups=keeps_dense_rows,
                stream_k=stream_k_selected(),
            )
            if width_free
            else None
        ),
        dense_rows=(
            mixed_requests.allocate_capture_dense_rows(
                max_rows=max_rows,
                vocab_size=model_runner.model_config.vocab_size,
                device=device,
            )
            if keeps_dense_rows
            else None
        ),
    )


class TrieGraphInputs:
    """The buffers every trie graph reads, sized for the largest one; a graph of
    fewer rows reads their leading part. Built anew for every capture, so the
    graphs record the buffers' initial values."""

    def __init__(
        self,
        *,
        max_rows: int,
        device,
        shared_context: Optional[SharedContextCapacity],
        ragged_layout: Optional[RaggedBeamAttentionMetadata] = None,
        dense_rows: Optional[BeamDenseRows] = None,
    ):
        # [max_rows] int64 SID codebook depth per row.
        self.levels = torch.zeros(max_rows, dtype=torch.int64, device=device)
        # Shared-context attention: this one-width capacity, or the width-free
        # layout below; neither with ordinary attention.
        self.shared_context = shared_context
        self.ragged_layout = ragged_layout
        # Full-vocabulary rows every graph keeps for a batch's ordinary requests.
        self.dense_rows = dense_rows
        # Rows of the graphs captured so far.
        self.captured_rows: set[int] = set()
        self._bucket: Optional[BeamAttentionBucket] = None
        if shared_context is not None:
            width = shared_context.beam_width
            groups = max_rows // width
            int32 = {"dtype": torch.int32, "device": device}
            # Captured with one prompt token and max_decode steps per group, on KV
            # slot 0; every replay stages the batch's values first.
            self._bucket = BeamAttentionBucket(
                row_indices=torch.arange(groups * width, **int32).view(groups, width),
                context_slots=torch.zeros(groups * shared_context.max_context, **int32),
                context_cu_seqlens=torch.arange(groups + 1, **int32),
                decode_slots=torch.zeros(
                    (groups, width * shared_context.max_decode), **int32
                ),
                beam_width=width,
                decode_lens=torch.full((groups,), shared_context.max_decode, **int32),
            )

    def supports_rows(self, rows: int) -> bool:
        """A graph of the one-width capacity holds whole groups of that width."""
        return self.shared_context is None or rows % self.shared_context.beam_width == 0

    def for_capture(self, rows: int) -> BeamForwardInputs:
        self.captured_rows.add(rows)
        return BeamForwardInputs(
            trie_levels=self.levels[:rows],
            attention_metadata=self._attention_for_capture(rows),
            dense_rows=self.dense_rows,
        )

    def _attention_for_capture(
        self, rows: int
    ) -> Optional[BeamAttentionMetadata | RaggedBeamAttentionMetadata]:
        if self.ragged_layout is not None:
            return self.ragged_layout
        if self._bucket is None:
            return None
        groups = rows // self._bucket.beam_width
        bucket = BeamAttentionBucket(
            row_indices=self._bucket.row_indices[:groups],
            context_slots=self._bucket.context_slots,
            context_cu_seqlens=self._bucket.context_cu_seqlens[: groups + 1],
            decode_slots=self._bucket.decode_slots[:groups],
            beam_width=self._bucket.beam_width,
            decode_lens=self._bucket.decode_lens[:groups],
        )
        return BeamAttentionMetadata(buckets=[bucket], num_rows=rows)

    def can_replay(self, beam: BeamForwardInputs, *, captured_rows: int) -> bool:
        if beam.trie_levels is None or captured_rows not in self.captured_rows:
            return False
        metadata, capacity = beam.attention_metadata, self.shared_context
        if self.ragged_layout is not None:
            # Any split pads like any decode batch; None is ordinary attention.
            return isinstance(metadata, RaggedBeamAttentionMetadata)
        if metadata is None or capacity is None:
            # Ordinary attention pads like any decode batch.
            return metadata is None and capacity is None
        return (
            metadata.num_rows == captured_rows
            and len(metadata.buckets) == 1
            and metadata.buckets[0].beam_width == capacity.beam_width
            and metadata.buckets[0].decode_slots.shape[1]
            <= capacity.beam_width * capacity.max_decode
        )

    def stage(self, beam: BeamForwardInputs) -> None:
        """Copy a batch that can_replay accepted. Slots past the staged lengths
        or counts and padding rows keep an earlier batch's values: the kernels
        read no slot past them, and a stale level is a valid depth."""
        self.levels[: beam.trie_levels.shape[0]].copy_(beam.trie_levels)
        if self.dense_rows is not None:
            mixed_requests.stage_dense_rows(
                target=self.dense_rows, step=beam.dense_rows
            )
        if beam.attention_metadata is None:
            return
        if self.ragged_layout is not None:
            ragged.stage_graph_metadata(
                target=self.ragged_layout, source=beam.attention_metadata
            )
            return
        source, static = beam.attention_metadata.buckets[0], self._bucket
        groups = source.row_indices.shape[0]
        static.row_indices[:groups].copy_(source.row_indices)
        static.context_slots[: source.context_slots.numel()].copy_(source.context_slots)
        static.context_cu_seqlens[: groups + 1].copy_(source.context_cu_seqlens)
        static.decode_slots[:groups, : source.decode_slots.shape[1]].copy_(
            source.decode_slots
        )
        static.decode_lens[:groups].copy_(source.decode_lens)


class BeamTrieDecodeCudaGraphRunner(DecodeCudaGraphRunner):
    """Decode graph runner of a compact-trie-head model: every size gets an
    ordinary graph and, where the size holds whole beam groups, a trie graph."""

    def __init__(self, model_runner: ModelRunner, **kwargs):
        self._capturing_trie_graph = False
        super().__init__(model_runner, **kwargs)

    def capture(self) -> None:
        self._trie_inputs = _trie_graph_inputs(
            self.model_runner, max_rows=self.max_num_token, device=self.device
        )
        super().capture()
        if self._trie_inputs.captured_rows:
            logger.info(
                "Beam trie decode graphs captured for %s rows.",
                sorted(self._trie_inputs.captured_rows),
            )
        else:
            logger.warning(
                "SGLANG_ENABLE_BEAM_TRIE_DECODE_GRAPH is set, but no decode graph "
                "size in %s is a multiple of SGLANG_BEAM_SHARED_CONTEXT_WIDTH=%d; "
                "trie batches run eagerly.",
                self.capture_bs,
                self._trie_inputs.shared_context.beam_width,
            )

    def capture_one_shape(
        self, size, forward, variant_label=None, attention_variant=None
    ):
        super().capture_one_shape(size, forward, variant_label, attention_variant)
        if not self._trie_inputs.supports_rows(size):
            return
        self._capturing_trie_graph = True
        try:
            super().capture_one_shape(
                size, forward, TRIE_GRAPH_VARIANT, attention_variant
            )
        finally:
            self._capturing_trie_graph = False

    def capture_prepare(self, size, num_tokens=None):
        forward_batch, attn_backend, pp_proxy_tensors = super().capture_prepare(
            size, num_tokens=num_tokens
        )
        if self._capturing_trie_graph:
            forward_batch.beam = self._trie_inputs.for_capture(size)
        return forward_batch, attn_backend, pp_proxy_tensors

    def _resolve_lora_variant(self, forward_batch: ForwardBatch):
        if forward_batch.beam is not None:
            return TRIE_GRAPH_VARIANT
        return super()._resolve_lora_variant(forward_batch)

    def _can_replay_beam(self, forward_batch: ForwardBatch) -> bool:
        rows = forward_batch.batch_size
        if rows > self.max_bs:
            return False
        captured_rows = (
            rows if self.disable_padding else self._pad_to_bucket(rows, self.capture_bs)
        )
        return self._trie_inputs.can_replay(
            forward_batch.beam, captured_rows=captured_rows
        )

    def load_batch(self, forward_batch: ForwardBatch, pp_proxy_tensors=None):
        if forward_batch.beam is not None:
            self._trie_inputs.stage(forward_batch.beam)
        super().load_batch(forward_batch, pp_proxy_tensors)

    def _init_replay_attn_metadata(self, forward_batch: ForwardBatch, **kwargs):
        # A shared-context trie graph reads the staged beam metadata and no
        # FlashInfer plan.
        if (
            forward_batch.beam is not None
            and forward_batch.beam.attention_metadata is not None
        ):
            return
        super()._init_replay_attn_metadata(forward_batch, **kwargs)

    def _process_output_after_replay(self, output, forward_batch: ForwardBatch):
        if output.beam_normalizer is None:
            return output
        return dataclasses.replace(
            output, beam_normalizer=output.beam_normalizer[: self.raw_num_token]
        )
