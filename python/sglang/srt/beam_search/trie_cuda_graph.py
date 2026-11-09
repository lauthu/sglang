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
"""Trie-only graph state, separate from ordinary and speculative graph inputs."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sglang.srt.beam_search import mixed_requests
from sglang.srt.beam_search import shared_context_attention as shared
from sglang.srt.beam_search import shared_context_ragged as ragged
from sglang.srt.environ import BeamDecodeGraphLevel, envs
from sglang.srt.model_executor.runner.shape_key import ShapeKey


@dataclass(frozen=True)
class BeamTrieShapeKey(ShapeKey):
    """A distinct key type keeps compact outputs out of ordinary graph replay."""


class TrieDecodeGraphState:
    def __init__(self, model_runner, max_tokens: int, device):
        self.shared_supported = shared.supports_shared_context(model_runner)
        self.signature = self.current_signature()
        self.use_shared = shared.enabled() and self.shared_supported
        self.levels = torch.zeros(max_tokens, dtype=torch.int64, device=device)
        self.metadata = {}
        # One layout for every row count, when any split of the rows can replay.
        self.ragged_template = None
        level = ragged.decode_graph_level()
        if self.use_shared and level >= BeamDecodeGraphLevel.MIXED_WIDTHS:
            ragged.warn_unused_fixed_width_settings()
            self.ragged_template = ragged.allocate_capture_layout(
                max_tokens,
                device,
                model_runner.req_to_token_pool.req_to_token,
                one_row_groups=level >= BeamDecodeGraphLevel.MIXED_REQUESTS,
            )
        # Full-vocabulary rows every beam graph may keep, for ordinary requests.
        self.dense_rows = None
        if level >= BeamDecodeGraphLevel.MIXED_REQUESTS:
            self.dense_rows = mixed_requests.allocate_capture_dense_rows(
                max_tokens, model_runner.model_config.vocab_size, device
            )

    def current_signature(self):
        attention = shared.enabled()
        level = ragged.decode_graph_level()
        return (
            attention,
            envs.SGLANG_OPT_BEAM_TRIE_TOPK_LOGPROB.get(),
            level,
            (
                mixed_requests.max_mixed_requests()
                if level >= BeamDecodeGraphLevel.MIXED_REQUESTS
                else None
            ),
            (
                shared.graph_beam_width(),
                shared.graph_max_context(),
                shared.graph_max_decode(),
            )
            if attention
            and self.shared_supported
            and level == BeamDecodeGraphLevel.FIXED_WIDTH
            else None,
        )

    def supports_rows(self, rows: int) -> bool:
        return (
            not self.use_shared
            or self.ragged_template is not None
            or (
                rows >= shared.graph_beam_width()
                and rows % shared.graph_beam_width() == 0
            )
        )

    def _capture_metadata(self, rows: int):
        if not self.use_shared:
            return None
        if self.ragged_template is not None:
            return ragged.capture_metadata(self.ragged_template, rows)
        return shared.build_graph_capture_metadata(rows, self.levels.device)

    def prepare_capture(self, forward_batch, rows: int) -> None:
        forward_batch.beam_trie_levels = self.levels[:rows]
        if rows not in self.metadata:
            self.metadata[rows] = self._capture_metadata(rows)
        if self.ragged_template is not None:
            # A recapture must not replay a staged step's tables while recording.
            self.ragged_template.counts().zero_()
        forward_batch.beam_attention_metadata = self.metadata[rows]
        if self.dense_rows is not None:
            self.dense_rows.count.zero_()
            forward_batch.beam_dense_rows = self.dense_rows

    def can_replay(self, forward_batch, rows: int, captured_rows: int) -> bool:
        if (
            self.signature != self.current_signature()
            or captured_rows not in self.metadata
            or forward_batch.beam_trie_levels.shape != (rows,)
            or forward_batch.beam_trie_levels.dtype != torch.int64
        ):
            return False
        dense_rows = forward_batch.beam_dense_rows
        if dense_rows is not None and (
            self.dense_rows is None or dense_rows.num_rows > self.dense_rows.num_rows
        ):
            return False
        if self.ragged_template is not None:
            return ragged.graph_compatible(
                forward_batch.beam_attention_metadata,
                rows,
                self.metadata[captured_rows],
            )
        if self.use_shared:
            return shared.graph_compatible(
                forward_batch.beam_attention_metadata, rows, captured_rows
            )
        return forward_batch.beam_attention_metadata is None

    def stage(self, forward_batch, rows: int, captured_rows: int, graph_size: int):
        if not self.can_replay(forward_batch, rows, captured_rows):
            raise ValueError(
                "Trie graph inputs do not match the captured configuration"
            )
        self.levels[:rows].copy_(forward_batch.beam_trie_levels)
        self.levels[rows:captured_rows].zero_()
        metadata = self.metadata[graph_size]
        if self.ragged_template is not None:
            ragged.stage_graph_metadata(metadata, forward_batch.beam_attention_metadata)
        elif metadata is not None:
            shared.stage_graph_metadata(metadata, forward_batch.beam_attention_metadata)
        if self.dense_rows is not None:
            mixed_requests.stage_dense_rows(
                self.dense_rows, forward_batch.beam_dense_rows
            )
        return metadata
