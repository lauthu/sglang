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
"""Shared-context decode attention for wide beam batches.

This module decides whether a server runs it and which of its two layouts a
beam decode batch gets; each layout has a module of its own:
- fixed-width (shared_context_fixed_width): buckets of requests that share a
  beam width, one kernel launch per bucket.
- variable-width (shared_context_variable_width): one layout for any beam
  widths, on a server whose beam decode graphs take any width.
"""

from __future__ import annotations

import logging
import weakref
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.beam_search import shared_context_fixed_width as fixed_width
from sglang.srt.beam_search import shared_context_variable_width as variable_width
from sglang.srt.environ import envs

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.mem_cache.kv.router import KVLayerRouter
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner

logger = logging.getLogger(__name__)


def enabled() -> bool:
    return envs.SGLANG_ENABLE_BEAM_SHARED_CONTEXT_ATTENTION.get()


def variable_width_graphs_enabled() -> bool:
    """SGLANG_ENABLE_BEAM_DECODE_GRAPH is on with no fixed width. With
    shared-context attention on, which the callers check, the beam decode graphs
    then take any beam width."""
    return (
        envs.SGLANG_ENABLE_BEAM_DECODE_GRAPH.get()
        and envs.SGLANG_BEAM_DECODE_GRAPH_FIXED_WIDTH.get() is None
    )


def _unsupported_reason(model_runner: ModelRunner) -> Optional[str]:
    from sglang.srt.configs.model_config import AttentionArch
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.runtime_context import get_exec

    if model_runner.decode_attention_backend_str != "flashinfer":
        return "the decode attention backend is not FlashInfer"
    if model_runner.sliding_window_size is not None:
        return "sliding-window attention is not supported"
    if model_runner.model_config.is_encoder_decoder:
        return "encoder-decoder models are not supported"
    if model_runner.model_config.attention_arch != AttentionArch.MHA:
        return "only MHA attention is supported"
    if model_runner.kv_cache_dtype not in (torch.float16, torch.bfloat16):
        return "the KV cache dtype must be float16 or bfloat16"
    if get_exec().overlap.enable_two_batch_overlap:
        return "two-batch overlap is enabled"
    for module in model_runner.model.modules():
        if isinstance(module, RadixAttention):
            if module.logit_cap > 0:
                return "attention logit caps are not supported"
            if module.v_head_dim != module.head_dim:
                return "v_head_dim differs from head_dim"
    return None


_support: weakref.WeakKeyDictionary[ModelRunner, bool] = weakref.WeakKeyDictionary()


def supports_shared_context(model_runner: ModelRunner) -> bool:
    """Evaluated once per runner; warns once when the env flag is on but the
    setup falls back to ordinary attention."""
    supported = _support.get(model_runner)
    if supported is None:
        reason = _unsupported_reason(model_runner)
        if reason is not None:
            logger.warning(
                "SGLANG_ENABLE_BEAM_SHARED_CONTEXT_ATTENTION is set but "
                "shared-context beam attention is unsupported (%s); "
                "using ordinary attention.",
                reason,
            )
        supported = _support[model_runner] = reason is None
    return supported


def build_metadata(
    batch: ScheduleBatch,
    model_runner: ModelRunner,
) -> Optional[
    fixed_width.BeamAttentionMetadata
    | variable_width.VariableWidthBeamAttentionMetadata
]:
    if batch.beam_tail is None or not batch.forward_mode.is_decode() or not enabled():
        return None
    if not supports_shared_context(model_runner):
        return None
    entries = batch.beam_tail.entries
    if not entries or batch.seq_lens_cpu is None:
        return None
    # A variable-width graph replays only the variable-width layout, so on such a
    # server every beam decode batch gets it, replayed or not.
    if variable_width_graphs_enabled():
        return variable_width.build_variable_width_metadata(
            batch=batch,
            req_to_token=model_runner.req_to_token_pool.req_to_token,
            stream_k_grid=variable_width.get_stream_k_grid(model_runner),
        )
    return fixed_width.build_fixed_width_metadata(
        batch=batch, model_runner=model_runner
    )


def forward(
    q: torch.Tensor,
    layer: RadixAttention,
    forward_batch: ForwardBatch,
    kv_layers: KVLayerRouter,
) -> Optional[torch.Tensor]:
    if forward_batch.beam is None:
        return None
    metadata = forward_batch.beam.attention_metadata
    if metadata is None:
        return None
    assert layer.logit_cap == 0

    k_buffer, v_buffer = kv_layers.read(layer.layer_id).kv()
    if k_buffer.ndim != 3 or v_buffer.ndim != 3:
        raise RuntimeError("Shared-context attention requires 3D MHA K/V cache buffers")
    # Each layout has its own kernels.
    if isinstance(metadata, variable_width.VariableWidthBeamAttentionMetadata):
        return variable_width.forward(
            q=q,
            layer=layer,
            forward_batch=forward_batch,
            metadata=metadata,
            k_buffer=k_buffer,
            v_buffer=v_buffer,
        )
    return fixed_width.forward(
        q=q, layer=layer, metadata=metadata, k_buffer=k_buffer, v_buffer=v_buffer
    )
