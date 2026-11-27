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

The beam decode graphs that take any beam width
(SGLANG_BEAM_DECODE_GRAPH_FIXED_WIDTH unset) cannot record a width at capture. So
instead of per-width buckets of gathered KV slots, a decode batch describes its
beams in one small int32 layout that the kernels read on the GPU. One kernel
launch then serves every beam width in the batch, and a graph captured for a row
count replays any batch of at most that many rows.

Terms:
- group: the beams of one request. Its leader is the request's own batch row; its
  members are rows appended after all requests' rows.
- tile: up to BEAMS_PER_TILE consecutive beams of one group. The kernels work
  tile by tile.
- layout: one int32 tensor that lists a batch's tiles and, for the stream-K
  kernel, its plan: the share of the batch's work that each of the kernel's CTAs
  does. Its format is in
  kernels/ops/attention/beam_shared_context_variable_width.py.

The layout is built on the host once per decode step and uploaded once: into the
graph's buffer when a graph replays the batch, into a tensor of its own otherwise.
Every layer of the step reads that one upload.
"""

from __future__ import annotations

import weakref
from typing import TYPE_CHECKING, Optional

import msgspec
import numpy as np
import torch

from sglang.srt.environ import envs
from sglang.srt.utils.device import is_pin_memory_available

if TYPE_CHECKING:
    from sglang.kernels.ops.attention.beam_shared_context_variable_width import (
        LayoutParts,
    )
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch
    from sglang.srt.model_executor.model_runner import ModelRunner

# Beams per tile, passed to the kernels as block_m. Not swept for these kernels:
# it is beam_shared_context.py's tile size for up to 5 groups.
BEAMS_PER_TILE = 128
# The variable-width layout has no split-KV kernel.
_KERNELS = ("default", "stream_k")


class StreamKGrid(msgspec.Struct, frozen=True):
    """What a stream-K plan is built for; fixed for a server's model and GPU."""

    # Query heads of an attention layer on this tensor-parallel rank.
    q_heads: int
    # CTAs of the kernel's grid: one per SM of the GPU.
    num_ctas: int


class VariableWidthBeamAttentionMetadata(msgspec.Struct):
    """What the variable-width kernels read. It is one of two things:

    - A batch's metadata: its layout is on the host (host_layout) until the one
      upload of the decode step, into a graph's buffer or into `layout`.
    - The graphs' metadata: `layout` is the buffer that every beam graph captured;
      it has no host_layout.
    """

    # A batch's layout on the host. None for the graphs' metadata.
    host_layout: Optional[torch.Tensor]
    # The layout on the GPU, which the kernels read. None for a batch until a
    # forward outside a graph uploads it.
    layout: Optional[torch.Tensor]
    # The request pool's KV slot table; the kernels look slots up in it on the GPU.
    req_to_token: torch.Tensor
    # Set to run the stream-K kernel; its plan is in the layout, before the tiles.
    stream_k_grid: Optional[StreamKGrid]
    # The parts of `layout` as the kernels take them, views of it. Split once: by
    # the first forward on it.
    layout_parts: Optional[LayoutParts] = None


class _Tile(msgspec.Struct, frozen=True):
    """One tile's entry in the layout; the field order is the kernels'."""

    leader_row: int
    first_member_row: int
    num_members: int
    prompt_len: int
    # The beam of its group that the tile starts at; beam 0 is the leader.
    first_beam: int


def _selected_kernel() -> str:
    kernel = envs.SGLANG_BEAM_SHARED_CONTEXT_KERNEL.get()
    if kernel not in _KERNELS:
        raise ValueError(
            f"SGLANG_BEAM_SHARED_CONTEXT_KERNEL is {kernel!r}, not one of "
            f"{_KERNELS}: the beam decode graphs for any beam width "
            "(SGLANG_BEAM_DECODE_GRAPH_FIXED_WIDTH unset) have no other kernel."
        )
    return kernel


_stream_k_grids: weakref.WeakKeyDictionary[ModelRunner, StreamKGrid] = (
    weakref.WeakKeyDictionary()
)


def get_stream_k_grid(model_runner: ModelRunner) -> Optional[StreamKGrid]:
    """The stream-K kernel's grid on this server; None when the default kernel
    runs. The grid is computed once per runner."""
    if _selected_kernel() != "stream_k":
        return None
    grid = _stream_k_grids.get(model_runner)
    if grid is None:
        from sglang.kernels.ops.attention.beam_shared_context_variable_width_stream_k import (
            default_num_ctas,
        )
        from sglang.srt.layers.radix_attention import RadixAttention

        # Every attention layer of a model has the same number of query heads.
        q_heads = next(
            module.tp_q_head_num
            for module in model_runner.model.modules()
            if isinstance(module, RadixAttention)
        )
        device = model_runner.req_to_token_pool.req_to_token.device
        grid = StreamKGrid(q_heads=q_heads, num_ctas=default_num_ctas(device.index))
        _stream_k_grids[model_runner] = grid
    return grid


def build_variable_width_metadata(
    batch: ScheduleBatch,
    req_to_token: torch.Tensor,
    *,
    stream_k_grid: Optional[StreamKGrid],
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

    # Header [tile count, real rows], then the stream-K plan, then each tile's
    # fields: the order split_layout reads them in. The plan is before the tiles
    # because its length does not depend on the batch.
    parts = [np.array([len(tiles), num_rows], dtype=np.int32)]
    if stream_k_grid is not None:
        parts.append(
            _stream_k_plan(tiles=tiles, decode_lens=decode_lens, grid=stream_k_grid)
        )
    parts.append(
        np.array([msgspec.structs.astuple(tile) for tile in tiles], dtype=np.int32)
    )
    layout = torch.from_numpy(np.concatenate([part.reshape(-1) for part in parts]))
    # Pinned, so that the upload does not wait for the copy.
    if is_pin_memory_available(batch.device):
        layout = layout.pin_memory()
    return VariableWidthBeamAttentionMetadata(
        host_layout=layout,
        layout=None,
        req_to_token=req_to_token,
        stream_k_grid=stream_k_grid,
    )


def _stream_k_plan(
    tiles: list[_Tile], decode_lens: list[int], grid: StreamKGrid
) -> np.ndarray:
    """The stream-K kernel's plan: how the batch's work is shared over the kernel's
    CTAs, from each tile's prompt length and generated tokens."""
    # The kernel modules import Triton, so this module imports them where it calls
    # them.
    from sglang.kernels.ops.attention.beam_shared_context_variable_width_stream_k import (
        build_stream_k_plan,
        work_units,
    )

    return build_stream_k_plan(
        work_units_per_tile=[
            work_units(prompt_len=tile.prompt_len, decode_len=decode_len)
            for tile, decode_len in zip(tiles, decode_lens, strict=True)
        ],
        q_heads=grid.q_heads,
        num_ctas=grid.num_ctas,
    )


def allocate_graph_metadata(
    max_rows: int,
    device,
    req_to_token: torch.Tensor,
    *,
    stream_k_grid: Optional[StreamKGrid],
) -> VariableWidthBeamAttentionMetadata:
    """The metadata every beam graph captures: one zeroed layout buffer, long
    enough for any batch of up to max_rows rows. A zeroed layout has no tiles and
    no real rows, so the captured launch attends nothing and writes every output
    row as zeros."""
    from sglang.kernels.ops.attention.beam_shared_context_variable_width import (
        header_and_tiles_len,
    )
    from sglang.kernels.ops.attention.beam_shared_context_variable_width_stream_k import (
        plan_len,
    )

    # The most tiles: every group has two beams, the fewest, and so one tile.
    max_tiles = max_rows // 2
    layout_len = header_and_tiles_len(num_tiles=max_tiles)
    if stream_k_grid is not None:
        layout_len += plan_len(num_ctas=stream_k_grid.num_ctas)
    return VariableWidthBeamAttentionMetadata(
        host_layout=None,
        layout=torch.zeros(layout_len, dtype=torch.int32, device=device),
        req_to_token=req_to_token,
        stream_k_grid=stream_k_grid,
    )


def device_layout(metadata: VariableWidthBeamAttentionMetadata, device) -> torch.Tensor:
    """The layout on the GPU. The graphs' buffer is there already. A batch's is
    uploaded by the first layer that runs it outside a graph, and kept for the
    step's other layers."""
    if metadata.layout is None:
        metadata.layout = metadata.host_layout.to(device, non_blocking=True)
    return metadata.layout


def device_layout_parts(
    metadata: VariableWidthBeamAttentionMetadata, device
) -> LayoutParts:
    """The layout on the GPU as its parts, which the kernels take. A batch's layout
    is split once per decode step and the graphs' buffer once, so a layer's launch
    makes no views."""
    if metadata.layout_parts is None:
        from sglang.kernels.ops.attention.beam_shared_context_variable_width import (
            split_layout,
        )
        from sglang.kernels.ops.attention.beam_shared_context_variable_width_stream_k import (
            plan_len,
        )

        grid = metadata.stream_k_grid
        metadata.layout_parts = split_layout(
            device_layout(metadata, device=device),
            plan_len=0 if grid is None else plan_len(num_ctas=grid.num_ctas),
        )
    return metadata.layout_parts


def _attention_kernel(stream_k: bool):
    """The kernel a layout runs, chosen by SGLANG_BEAM_SHARED_CONTEXT_KERNEL. Both
    take the same inputs; the stream-K kernel also takes its plan and grid size.
    Its module says how the two differ."""
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
    grid = metadata.stream_k_grid
    attention = _attention_kernel(stream_k=grid is not None)
    layout = device_layout_parts(metadata, device=q.device)
    # The stream-K kernel also takes its plan and the grid the plan was built for.
    stream_k_inputs = (
        {} if grid is None else {"plan": layout.plan, "num_ctas": grid.num_ctas}
    )
    output = attention(
        q=q.reshape(-1, q_heads, head_dim),
        k_buffer=k_buffer,
        v_buffer=v_buffer,
        req_to_token=metadata.req_to_token,
        req_pool_indices=forward_batch.req_pool_indices,
        seq_lens=forward_batch.seq_lens,
        tile_count=layout.tile_count,
        row_count=layout.row_count,
        tiles=layout.tiles,
        block_m=BEAMS_PER_TILE,
        softmax_scale=layer.scaling,
        **stream_k_inputs,
    )
    return output.view(-1, q_heads * head_dim)
