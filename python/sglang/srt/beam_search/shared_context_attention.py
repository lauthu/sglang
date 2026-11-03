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
"""Shared-context decode attention for wide beam batches."""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import TYPE_CHECKING, Optional

import torch

from sglang.srt.beam_search.beam_kernels import (
    group_beam_rows,
    ungroup_beam_rows,
)
from sglang.srt.environ import envs
from sglang.srt.utils.common import is_pin_memory_available

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner


@dataclass
class BeamAttentionMetadata:
    num_groups: int
    context_slots: torch.Tensor
    context_cu_seqlens: torch.Tensor
    decode_slots: torch.Tensor
    beam_width: int
    decode_len: int
    decode_lens: torch.Tensor


def enabled() -> bool:
    return envs.SGLANG_BEAM_SHARED_CONTEXT_ATTENTION.get()


def _positive_int_env(field) -> int:
    value = field.get()
    if value <= 0:
        raise ValueError(f"{field.name} must be positive, got {value}")
    return value


def graph_beam_width() -> int:
    return _positive_int_env(envs.SGLANG_BEAM_SHARED_CONTEXT_WIDTH)


def graph_max_context() -> int:
    return _positive_int_env(envs.SGLANG_BEAM_SHARED_CONTEXT_GRAPH_MAX_CONTEXT)


def graph_max_decode() -> int:
    return _positive_int_env(envs.SGLANG_BEAM_SHARED_CONTEXT_GRAPH_MAX_DECODE)


def graph_compatible(
    metadata: Optional[BeamAttentionMetadata],
    num_rows: int,
    captured_rows: int,
) -> bool:
    if metadata is None or num_rows != captured_rows:
        return False
    if metadata.beam_width != graph_beam_width():
        return False
    return (
        metadata.num_groups * metadata.beam_width == num_rows
        and metadata.context_slots.numel() <= metadata.num_groups * graph_max_context()
        and metadata.decode_len <= graph_max_decode()
    )


def build_graph_capture_metadata(
    num_rows: int,
    device: torch.device,
) -> Optional[BeamAttentionMetadata]:
    if not enabled():
        return None
    beam_width = graph_beam_width()
    if num_rows < beam_width or num_rows % beam_width:
        return None

    num_groups = num_rows // beam_width
    max_context = graph_max_context()
    max_decode = graph_max_decode()
    return BeamAttentionMetadata(
        num_groups=num_groups,
        context_slots=torch.zeros(
            num_groups * max_context,
            dtype=torch.int32,
            device=device,
        ),
        context_cu_seqlens=torch.arange(
            num_groups + 1,
            dtype=torch.int32,
            device=device,
        ),
        decode_slots=torch.zeros(
            (num_groups, max_decode * beam_width),
            dtype=torch.int32,
            device=device,
        ),
        beam_width=beam_width,
        decode_len=max_decode,
        decode_lens=torch.full(
            (num_groups,),
            max_decode,
            dtype=torch.int32,
            device=device,
        ),
    )


def stage_graph_metadata(
    target: BeamAttentionMetadata,
    source: BeamAttentionMetadata,
) -> None:
    if not graph_compatible(
        source,
        source.num_groups * source.beam_width,
        target.num_groups * target.beam_width,
    ):
        raise ValueError("Beam attention metadata does not fit the captured graph")

    target.context_slots[: source.context_slots.numel()].copy_(source.context_slots)
    target.context_cu_seqlens.copy_(source.context_cu_seqlens)
    target.decode_slots[:, : source.decode_slots.shape[1]].copy_(source.decode_slots)
    target.decode_lens.fill_(source.decode_len)


def build_metadata(
    batch: ScheduleBatch,
    model_runner: ModelRunner,
) -> Optional[BeamAttentionMetadata]:
    if not enabled() or batch.beam_tail is None:
        return None
    entries = batch.beam_tail.entries
    if not entries or batch.seq_lens_cpu is None:
        return None

    widths = [entry.end - entry.start + 1 for entry in entries]
    if len(set(widths)) != 1:
        return None
    beam_width = widths[0]
    if beam_width <= 1:
        return None
    if len(entries) * beam_width != len(batch.seq_lens):
        return None

    # Uploads are pinned and non-blocking, and no index reads the device back,
    # so building this metadata never waits on the running forward.
    pin_memory = is_pin_memory_available(batch.device)
    num_base_rows = batch.beam_tail.num_base_rows
    group_rows_cpu = torch.tensor(
        [
            [entry.leader_idx]
            + list(
                range(
                    num_base_rows + entry.start,
                    num_base_rows + entry.end,
                )
            )
            for entry in entries
        ],
        dtype=torch.int64,
        pin_memory=pin_memory,
    )
    group_rows = group_rows_cpu.to(batch.device, non_blocking=True)
    req_rows = batch.req_pool_indices[group_rows]
    req_to_token = model_runner.req_to_token_pool.req_to_token

    context_parts = []
    decode_parts = []
    prompt_lens = []
    decode_lens = []
    for group_index, entry in enumerate(entries):
        prompt_len = int(entry.group.prompt_len)
        seq_len = int(batch.seq_lens_cpu[entry.leader_idx])
        decode_len = seq_len - prompt_len
        if decode_len <= 0:
            return None
        group_req_rows = req_rows[group_index]
        # A slice, not group_req_rows[0]: a 0-d CUDA index is read back to host.
        context_parts.append(req_to_token[group_req_rows[:1], :prompt_len].flatten())
        decode_parts.append(
            req_to_token[
                group_req_rows,
                prompt_len:seq_len,
            ]
            .transpose(0, 1)
            .reshape(-1)
        )
        prompt_lens.append(prompt_len)
        decode_lens.append(decode_len)

    if len(set(decode_lens)) != 1:
        return None
    decode_len = decode_lens[0]

    context_cu_seqlens = torch.tensor(
        [0, *itertools.accumulate(prompt_lens)],
        dtype=torch.int32,
        pin_memory=pin_memory,
    ).to(batch.device, non_blocking=True)
    return BeamAttentionMetadata(
        num_groups=len(entries),
        context_slots=torch.cat(context_parts),
        context_cu_seqlens=context_cu_seqlens,
        decode_slots=torch.stack(decode_parts),
        beam_width=beam_width,
        decode_len=decode_len,
        decode_lens=torch.full(
            (len(entries),),
            decode_len,
            dtype=torch.int32,
            device=batch.device,
        ),
    )


def forward(
    q: torch.Tensor,
    layer: RadixAttention,
    forward_batch: ForwardBatch,
    token_to_kv_pool: MHATokenToKVPool,
) -> Optional[torch.Tensor]:
    metadata = forward_batch.beam_attention_metadata
    if metadata is None:
        return None

    k_buffer, v_buffer = token_to_kv_pool.get_kv_buffer(layer.layer_id)
    if k_buffer.ndim != 3 or v_buffer.ndim != 3:
        raise RuntimeError("Shared-context attention requires 3D MHA K/V cache buffers")

    num_groups = metadata.num_groups
    beam_width = metadata.beam_width
    q_heads = layer.tp_q_head_num
    head_dim = layer.head_dim
    grouped_q = group_beam_rows(
        q,
        num_groups,
        beam_width,
    ).view(num_groups, beam_width, q_heads, head_dim)
    from sglang.srt.beam_search.shared_context_attention_kernel import (
        shared_context_attention,
    )

    output = shared_context_attention(
        grouped_q,
        k_buffer,
        v_buffer,
        metadata.context_slots,
        metadata.context_cu_seqlens,
        metadata.decode_slots,
        metadata.decode_lens,
        softmax_scale=layer.scaling,
    )
    flat_output = ungroup_beam_rows(
        output,
        num_groups,
        beam_width,
    ).view(-1, q_heads, head_dim)
    return flat_output.view(-1, q_heads * head_dim)
