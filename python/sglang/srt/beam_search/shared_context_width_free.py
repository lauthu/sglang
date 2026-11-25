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
"""Width-free shared-context attention metadata, for the beam decode graphs that
take any beam width (decode_graph_settings).

A decode batch describes its beam groups in one small int32 layout instead of
per-width buckets of gathered KV slots. One kernel launch then serves every
width in the batch, and a beam decode graph captured for a row count replays for
any split of at most that many rows into groups. For the stream-K kernel the
layout ends with its schedule: each tile's work units.
"""

from __future__ import annotations

import itertools
from typing import TYPE_CHECKING, Optional

import msgspec
import torch

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

# Beams per tile, the kernels' M dimension. Arbitrary: the fixed-width kernel's
# tile for small group counts.
WIDTH_FREE_BLOCK_M = 128


class WidthFreeBeamAttentionMetadata(msgspec.Struct):
    # int32 layout of kernels/ops/attention/beam_shared_context_width_free.py: a batch's
    # own, or a captured buffer sized for the most a graph holds.
    layout: torch.Tensor
    req_to_token: torch.Tensor
    # Run the stream-K kernel, which reads the schedule at the layout's end.
    stream_k: bool = False


def _layout_tensor(values: list[int], device) -> torch.Tensor:
    if torch.device(device).type == "cuda":
        return torch.tensor(values, dtype=torch.int32, pin_memory=True).to(
            device, non_blocking=True
        )
    return torch.tensor(values, dtype=torch.int32, device=device)


def build_width_free_metadata(
    batch: ScheduleBatch,
    req_to_token: torch.Tensor,
    *,
    stream_k: bool = False,
) -> Optional[WidthFreeBeamAttentionMetadata]:
    """The layout of a beam decode batch, built on the host from the beam tail; None
    leaves the batch to ordinary attention. Every row must belong to a beam group
    of at least two rows: a batch with a regular request gets no layout."""
    num_rows = len(batch.seq_lens)
    num_base_rows = batch.beam_tail.num_base_rows
    # Leaders are base rows.
    base_lens = batch.seq_lens_cpu[:num_base_rows].tolist()
    # Per group: (leader row, first member row, members, prompt length).
    groups: list[tuple[int, int, int, int]] = []
    for entry in batch.beam_tail.entries:
        num_members = entry.end - entry.start
        prompt_len = int(entry.group.prompt_len)
        if num_members < 1 or base_lens[entry.leader_idx] <= prompt_len:
            return None
        groups.append(
            (entry.leader_idx, num_base_rows + entry.start, num_members, prompt_len)
        )
    covered_rows = sum(num_members + 1 for _, _, num_members, _ in groups)
    if covered_rows != num_rows:
        return None

    # A tile is WIDTH_FREE_BLOCK_M beams of one group, the leader first. It carries
    # its group's fields, so a kernel reads a tile without a second lookup.
    tiles = [
        (leader, first_member, num_members, prompt_len, first_beam)
        for leader, first_member, num_members, prompt_len in groups
        for first_beam in range(0, num_members + 1, WIDTH_FREE_BLOCK_M)
    ]
    schedule = _stream_k_schedule(tiles=tiles, base_lens=base_lens) if stream_k else []
    return WidthFreeBeamAttentionMetadata(
        layout=_layout_tensor(
            values=[
                len(tiles),
                num_rows,
                *itertools.chain.from_iterable(tiles),
                *schedule,
            ],
            device=batch.device,
        ),
        req_to_token=req_to_token,
        stream_k=stream_k,
    )


def _stream_k_schedule(
    tiles: list[tuple[int, int, int, int, int]], base_lens: list[int]
) -> list[int]:
    """The stream-K kernel's schedule: each tile's work units for one query head, from
    its group's prompt length and decode depth."""
    from sglang.kernels.ops.attention.beam_shared_context_width_free_stream_k import (
        build_schedule,
        tile_units,
    )

    return build_schedule(
        [
            tile_units(prompt_len=prompt_len, decode_len=base_lens[leader] - prompt_len)
            for leader, _, _, prompt_len, _ in tiles
        ]
    )


def allocate_capture_layout(
    max_rows: int,
    device,
    req_to_token: torch.Tensor,
    *,
    stream_k: bool = False,
) -> WidthFreeBeamAttentionMetadata:
    """One layout buffer for every captured row count, sized for any batch of
    max_rows rows. All zeros, it records a launch with its fixed grid and no work."""
    from sglang.kernels.ops.attention.beam_shared_context_width_free import (
        layout_tables_size,
    )
    from sglang.kernels.ops.attention.beam_shared_context_width_free_stream_k import (
        schedule_size,
    )

    # Every beam group has at least two rows.
    max_groups = max(1, max_rows // 2)
    # Each group adds at most one partial tile to the rows' full tiles.
    max_tiles = -(-max_rows // WIDTH_FREE_BLOCK_M) + max_groups
    size = layout_tables_size(num_tiles=max_tiles)
    if stream_k:
        size += schedule_size(max_tiles)
    return WidthFreeBeamAttentionMetadata(
        layout=torch.zeros(size, dtype=torch.int32, device=device),
        req_to_token=req_to_token,
        stream_k=stream_k,
    )


def forward(
    q: torch.Tensor,
    layer: RadixAttention,
    forward_batch: ForwardBatch,
    metadata: WidthFreeBeamAttentionMetadata,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
) -> torch.Tensor:
    q_heads = layer.tp_q_head_num
    head_dim = layer.head_dim
    # Two kernels of one contract, chosen by SGLANG_BEAM_SHARED_CONTEXT_KERNEL: see
    # the stream-K module for where each is faster.
    if metadata.stream_k:
        from sglang.kernels.ops.attention.beam_shared_context_width_free_stream_k import (
            shared_context_attention_width_free_stream_k as attention,
        )
    else:
        from sglang.kernels.ops.attention.beam_shared_context_width_free import (
            shared_context_attention_width_free as attention,
        )
    output = attention(
        q=q.reshape(-1, q_heads, head_dim),
        k_buffer=k_buffer,
        v_buffer=v_buffer,
        req_to_token=metadata.req_to_token,
        req_pool_indices=forward_batch.req_pool_indices,
        seq_lens=forward_batch.seq_lens,
        layout=metadata.layout,
        block_m=WIDTH_FREE_BLOCK_M,
        softmax_scale=layer.scaling,
    )
    return output.view(-1, q_heads * head_dim)
