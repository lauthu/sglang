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
"""Width-free shared-context attention metadata (SGLANG_BEAM_DECODE_GRAPH_LEVEL >= 2).

A decode step describes its beam groups in one small int32 layout instead of
per-width buckets of gathered KV slots. One kernel launch then serves every
width in the step, and a beam decode graph captured for a row count replays for
any split of at most that many rows into groups. At level 3 with the compact trie
head, every other row of the step (an ordinary request, or a finished group's
leader) is a one-row group whose whole KV is its prompt. For the stream-K kernel
the layout ends with its schedule: each tile's work units.
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
RAGGED_BLOCK_M = 128


class RaggedBeamAttentionMetadata(msgspec.Struct):
    # int32 layout of kernels/ops/attention/beam_shared_context_ragged.py: a step's
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


def build_ragged_metadata(
    batch: ScheduleBatch,
    req_to_token: torch.Tensor,
    *,
    one_row_groups: bool = False,
    stream_k: bool = False,
) -> Optional[RaggedBeamAttentionMetadata]:
    """The layout of a beam decode step, built on the host from the beam tail; None
    leaves the step to ordinary attention. Every row must belong to a beam group,
    unless one_row_groups lets the other base rows run as one-row groups."""
    num_rows = len(batch.seq_lens)
    num_base_rows = batch.beam_tail.num_base_rows
    # Leaders and one-row groups are base rows.
    base_lens = batch.seq_lens_cpu[:num_base_rows].tolist()
    # Rows of the group table: (leader row, first member row, members, prompt length).
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
    if covered_rows < num_rows and one_row_groups:
        # The row's whole KV is its prompt, so the kernel's decode loop is empty.
        leaders = {leader for leader, _, _, _ in groups}
        groups += [
            (row, row, 0, seq_len)
            for row, seq_len in enumerate(base_lens)
            if row not in leaders
        ]
        covered_rows = sum(num_members + 1 for _, _, num_members, _ in groups)
    if covered_rows != num_rows:
        return None

    # A tile is RAGGED_BLOCK_M beams of one group, the leader first.
    tiles = [
        (group, first_beam)
        for group, (_, _, num_members, _) in enumerate(groups)
        for first_beam in range(0, num_members + 1, RAGGED_BLOCK_M)
    ]
    schedule = (
        _stream_k_schedule(groups=groups, tiles=tiles, base_lens=base_lens)
        if stream_k
        else []
    )
    return RaggedBeamAttentionMetadata(
        layout=_layout_tensor(
            values=[
                len(groups),
                len(tiles),
                num_rows,
                *itertools.chain.from_iterable(groups),
                *itertools.chain.from_iterable(tiles),
                *schedule,
            ],
            device=batch.device,
        ),
        req_to_token=req_to_token,
        stream_k=stream_k,
    )


def _stream_k_schedule(
    groups: list[tuple[int, int, int, int]],
    tiles: list[tuple[int, int]],
    base_lens: list[int],
) -> list[int]:
    """The stream-K kernel's schedule: each tile's work units for one query head, from
    its group's prompt length and decode depth (0 for a one-row group)."""
    from sglang.kernels.ops.attention.beam_shared_context_ragged_stream_k import (
        build_schedule,
        tile_units,
    )

    group_units = [
        tile_units(prompt_len=prompt_len, decode_len=base_lens[leader] - prompt_len)
        for leader, _, _, prompt_len in groups
    ]
    return build_schedule([group_units[group] for group, _ in tiles])


def allocate_capture_layout(
    max_rows: int,
    device,
    req_to_token: torch.Tensor,
    *,
    one_row_groups: bool = False,
    stream_k: bool = False,
) -> RaggedBeamAttentionMetadata:
    """One layout buffer for every captured row count, sized for any step of
    max_rows rows. All zeros, it records a launch with its fixed grid and no work."""
    from sglang.kernels.ops.attention.beam_shared_context_ragged import (
        layout_tables_size,
    )
    from sglang.kernels.ops.attention.beam_shared_context_ragged_stream_k import (
        schedule_size,
    )

    # Every beam group has at least two rows, unless one-row groups are allowed.
    max_groups = max_rows if one_row_groups else max(1, max_rows // 2)
    # Each group adds at most one partial tile to the rows' full tiles.
    max_tiles = -(-max_rows // RAGGED_BLOCK_M) + max_groups
    size = layout_tables_size(num_groups=max_groups, num_tiles=max_tiles)
    if stream_k:
        size += schedule_size(max_tiles)
    return RaggedBeamAttentionMetadata(
        layout=torch.zeros(size, dtype=torch.int32, device=device),
        req_to_token=req_to_token,
        stream_k=stream_k,
    )


def forward(
    q: torch.Tensor,
    layer: RadixAttention,
    forward_batch: ForwardBatch,
    metadata: RaggedBeamAttentionMetadata,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
) -> torch.Tensor:
    q_heads = layer.tp_q_head_num
    head_dim = layer.head_dim
    # Two kernels of one contract, chosen by SGLANG_BEAM_SHARED_CONTEXT_KERNEL: see
    # the stream-K module for where each is faster.
    if metadata.stream_k:
        from sglang.kernels.ops.attention.beam_shared_context_ragged_stream_k import (
            shared_context_attention_ragged_stream_k as attention,
        )
    else:
        from sglang.kernels.ops.attention.beam_shared_context_ragged import (
            shared_context_attention_ragged as attention,
        )
    output = attention(
        q=q.reshape(-1, q_heads, head_dim),
        k_buffer=k_buffer,
        v_buffer=v_buffer,
        req_to_token=metadata.req_to_token,
        req_pool_indices=forward_batch.req_pool_indices,
        seq_lens=forward_batch.seq_lens,
        layout=metadata.layout,
        block_m=RAGGED_BLOCK_M,
        softmax_scale=layer.scaling,
    )
    return output.view(-1, q_heads * head_dim)
