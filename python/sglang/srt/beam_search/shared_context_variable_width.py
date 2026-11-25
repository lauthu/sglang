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
"""Variable-width shared-context attention: the host side.

The beam decode graphs that take any beam width (decode_graph_settings) cannot
record a width at capture. So instead of per-width buckets of gathered KV slots,
a decode batch describes its beams in one small int32 layout that the kernels read
on the GPU. One kernel launch then serves every beam width in the batch, and a
graph captured for a row count replays any batch of at most that many rows.

Terms:
- group: the beams of one request. Its leader is the request's own batch row; its
  members are rows appended after all requests' rows.
- tile: up to BEAMS_PER_TILE consecutive beams of one group. The kernels work
  tile by tile.
- layout: one int32 tensor that lists a batch's tiles. Its format is in
  kernels/ops/attention/beam_shared_context_variable_width.py.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Optional

import msgspec
import torch

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

# Beams per tile, passed to the kernels as block_m. Not swept for these kernels:
# it is beam_shared_context.py's tile size for up to 5 groups.
BEAMS_PER_TILE = 128


class VariableWidthBeamAttentionMetadata(msgspec.Struct):
    # One batch's layout, or the graphs' shared buffer, which is long enough for
    # the largest graph's layout.
    layout: torch.Tensor
    # The request pool's KV slot table; the kernels look slots up in it on the GPU.
    req_to_token: torch.Tensor
    # Run the stream-K kernel, whose schedule follows the tiles in the layout.
    stream_k: bool = False


class _Tile(msgspec.Struct, frozen=True):
    """One tile's entry in the layout; the field order is the kernels'."""

    leader_row: int
    first_member_row: int
    num_members: int
    prompt_len: int
    # The beam of its group that the tile starts at; beam 0 is the leader.
    first_beam: int


def _layout_tensor(values: list[int], device) -> torch.Tensor:
    if torch.device(device).type == "cuda":
        return torch.tensor(values, dtype=torch.int32, pin_memory=True).to(
            device, non_blocking=True
        )
    return torch.tensor(values, dtype=torch.int32, device=device)


def build_variable_width_metadata(
    batch: ScheduleBatch,
    req_to_token: torch.Tensor,
    *,
    stream_k: bool = False,
) -> Optional[VariableWidthBeamAttentionMetadata]:
    """The layout of a beam decode batch, built on the host. Returns None for a
    batch the kernels do not take; it then runs ordinary attention."""
    num_rows = len(batch.seq_lens)
    # Batch rows: one row per request first (a group's leader), then every group's
    # member rows.
    num_leader_rows = batch.beam_tail.num_base_rows
    leader_seq_lens = batch.seq_lens_cpu[:num_leader_rows].tolist()

    tiles: list[_Tile] = []
    # Per tile: the generated tokens each of its beams attends.
    decode_lens: list[int] = []
    rows_in_groups = 0
    for entry in batch.beam_tail.entries:
        num_members = entry.end - entry.start
        prompt_len = int(entry.group.prompt_len)
        decode_len = leader_seq_lens[entry.leader_idx] - prompt_len
        # Ordinary attention takes a group without member rows or generated tokens.
        if num_members < 1 or decode_len < 1:
            return None
        beam_width = num_members + 1
        rows_in_groups += beam_width
        for first_beam in range(0, beam_width, BEAMS_PER_TILE):
            tiles.append(
                _Tile(
                    leader_row=entry.leader_idx,
                    first_member_row=num_leader_rows + entry.start,
                    num_members=num_members,
                    prompt_len=prompt_len,
                    first_beam=first_beam,
                )
            )
            decode_lens.append(decode_len)
    # A row outside every group: a regular request, or a finished group's leader.
    if rows_in_groups != num_rows:
        return None

    # Header [tile count, real rows], then each tile's fields, then the stream-K
    # schedule: the order the kernels read.
    values = [len(tiles), num_rows]
    for tile in tiles:
        values.extend(msgspec.structs.astuple(tile))
    if stream_k:
        values.extend(_stream_k_schedule(tiles=tiles, decode_lens=decode_lens))
    return VariableWidthBeamAttentionMetadata(
        layout=_layout_tensor(values=values, device=batch.device),
        req_to_token=req_to_token,
        stream_k=stream_k,
    )


def _stream_k_schedule(tiles: list[_Tile], decode_lens: list[int]) -> list[int]:
    """The stream-K kernel's schedule: how much work each tile is, from its group's
    prompt length and generated tokens."""
    # The kernel modules import Triton, so this module imports them where it calls
    # them.
    from sglang.kernels.ops.attention.beam_shared_context_variable_width_stream_k import (
        build_schedule,
        task_units,
    )

    return build_schedule(
        units_per_task=[
            task_units(prompt_len=tile.prompt_len, decode_len=decode_len)
            for tile, decode_len in zip(tiles, decode_lens, strict=True)
        ]
    )


def allocate_graph_metadata(
    max_rows: int,
    device,
    req_to_token: torch.Tensor,
    *,
    stream_k: bool = False,
) -> VariableWidthBeamAttentionMetadata:
    """The metadata every beam graph captures: one zeroed layout buffer, long
    enough for any batch of up to max_rows rows. A zeroed layout has no tiles and
    no real rows, so the captured launch attends nothing and writes every output
    row as zeros."""
    from sglang.kernels.ops.attention.beam_shared_context_variable_width import (
        header_and_tiles_len,
    )
    from sglang.kernels.ops.attention.beam_shared_context_variable_width_stream_k import (
        schedule_len,
    )

    # Upper bound: the rows' full tiles plus one partial tile per group, and a
    # group has at least two rows.
    max_groups = max(1, max_rows // 2)
    max_tiles = math.ceil(max_rows / BEAMS_PER_TILE) + max_groups
    layout_len = header_and_tiles_len(num_tiles=max_tiles)
    if stream_k:
        layout_len += schedule_len(num_tiles=max_tiles)
    return VariableWidthBeamAttentionMetadata(
        layout=torch.zeros(layout_len, dtype=torch.int32, device=device),
        req_to_token=req_to_token,
        stream_k=stream_k,
    )


def attention_kernel(stream_k: bool):
    """The kernel a layout runs. Two kernels of one contract, chosen by
    SGLANG_BEAM_SHARED_CONTEXT_KERNEL; the stream-K module says where each is
    faster."""
    if stream_k:
        from sglang.kernels.ops.attention.beam_shared_context_variable_width_stream_k import (
            shared_context_attention_variable_width_stream_k,
        )

        return shared_context_attention_variable_width_stream_k
    from sglang.kernels.ops.attention.beam_shared_context_variable_width import (
        shared_context_attention_variable_width,
    )

    return shared_context_attention_variable_width


def forward(
    q: torch.Tensor,
    layer: RadixAttention,
    forward_batch: ForwardBatch,
    metadata: VariableWidthBeamAttentionMetadata,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
) -> torch.Tensor:
    q_heads = layer.tp_q_head_num
    head_dim = layer.head_dim
    attention = attention_kernel(metadata.stream_k)
    output = attention(
        q=q.reshape(-1, q_heads, head_dim),
        k_buffer=k_buffer,
        v_buffer=v_buffer,
        req_to_token=metadata.req_to_token,
        req_pool_indices=forward_batch.req_pool_indices,
        seq_lens=forward_batch.seq_lens,
        layout=metadata.layout,
        block_m=BEAMS_PER_TILE,
        softmax_scale=layer.scaling,
    )
    return output.view(-1, q_heads * head_dim)
