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
"""Width-free shared-context attention metadata (SGLANG_BEAM_DECODE_GRAPH_LEVEL >= 1).

A decode step describes its beam groups in one small int32 layout instead of
per-width buckets of gathered KV slots. One kernel launch then serves every
width in the step, and a CUDA graph captured for a row count replays for any
split of those rows into groups.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import msgspec
import torch

from sglang.srt.environ import BeamDecodeGraphLevel, envs

if TYPE_CHECKING:
    from sglang.srt.layers.radix_attention import RadixAttention
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

logger = logging.getLogger(__name__)

# Beams per tile, the kernel's M dimension. Arbitrary: upstream's tile for small
# group counts; a layout carries it, so the kernel always matches its tiles.
RAGGED_BLOCK_M = 128
# Layout: counts (groups, tiles, real rows), the group table, then the tile table.
_NUM_COUNTS = 3
_GROUP_FIELDS = 4
_TILE_FIELDS = 2


class RaggedBeamAttentionMetadata(msgspec.Struct):
    # int32 [3 + 4 * group_capacity + 2 * tile_capacity]
    layout: torch.Tensor
    group_capacity: int
    tile_capacity: int
    block_m: int
    # Rows of Q the kernel reads: the step's rows, or a captured row count.
    num_rows: int
    req_to_token: torch.Tensor

    def counts(self) -> torch.Tensor:
        return self.layout[:_NUM_COUNTS]

    def groups(self) -> torch.Tensor:
        start = _NUM_COUNTS
        end = start + _GROUP_FIELDS * self.group_capacity
        return self.layout[start:end].view(self.group_capacity, _GROUP_FIELDS)

    def tiles(self) -> torch.Tensor:
        start = _NUM_COUNTS + _GROUP_FIELDS * self.group_capacity
        end = start + _TILE_FIELDS * self.tile_capacity
        return self.layout[start:end].view(self.tile_capacity, _TILE_FIELDS)


def decode_graph_level() -> BeamDecodeGraphLevel:
    level = BeamDecodeGraphLevel(envs.SGLANG_BEAM_DECODE_GRAPH_LEVEL.get())
    if level == BeamDecodeGraphLevel.MIXED_REQUESTS:
        raise NotImplementedError(
            "SGLANG_BEAM_DECODE_GRAPH_LEVEL=2 (beam and ordinary requests in one "
            "graph) is not implemented yet; use 0 or 1"
        )
    kernel = envs.SGLANG_BEAM_SHARED_CONTEXT_KERNEL.get()
    if level != BeamDecodeGraphLevel.FIXED_WIDTH and kernel != "default":
        raise ValueError(
            f"SGLANG_BEAM_SHARED_CONTEXT_KERNEL={kernel!r} supports only "
            "SGLANG_BEAM_DECODE_GRAPH_LEVEL=0; level 1 runs its own kernel"
        )
    return level


def ragged_enabled() -> bool:
    return decode_graph_level() >= BeamDecodeGraphLevel.MIXED_WIDTHS


def warn_unused_fixed_width_settings() -> None:
    unused = [
        field.name
        for field in (
            envs.SGLANG_BEAM_SHARED_CONTEXT_WIDTH,
            envs.SGLANG_BEAM_SHARED_CONTEXT_GRAPH_MAX_CONTEXT,
            envs.SGLANG_BEAM_SHARED_CONTEXT_GRAPH_MAX_DECODE,
        )
        if field.is_set()
    ]
    if unused:
        logger.warning(
            "%s have no effect with SGLANG_BEAM_DECODE_GRAPH_LEVEL>=1: beam decode "
            "graphs serve any width and read KV slots from req_to_token",
            ", ".join(unused),
        )


def ragged_tile_capacity(num_rows: int, group_capacity: int, block_m: int) -> int:
    """Bound on sum(ceil(width / block_m)) over at most group_capacity groups of
    num_rows rows: each group adds at most one partial tile."""
    return -(-num_rows // block_m) + group_capacity


def _layout_tensor(values: list[int], device) -> torch.Tensor:
    if torch.device(device).type == "cuda":
        return torch.tensor(values, dtype=torch.int32, pin_memory=True).to(
            device, non_blocking=True
        )
    return torch.tensor(values, dtype=torch.int32, device=device)


def build_ragged_metadata(
    batch: ScheduleBatch, req_to_token: torch.Tensor
) -> Optional[RaggedBeamAttentionMetadata]:
    """Group and tile tables of an all-beam decode step, built on the host from the
    beam tail; None leaves the step to ordinary attention."""
    num_rows = len(batch.seq_lens)
    num_base_rows = batch.beam_tail.num_base_rows
    groups: list[int] = []
    tiles: list[int] = []
    covered_rows = 0
    for group, entry in enumerate(batch.beam_tail.entries):
        num_members = entry.end - entry.start
        prompt_len = int(entry.group.prompt_len)
        if num_members < 1:
            return None
        if int(batch.seq_lens_cpu[entry.leader_idx]) <= prompt_len:
            return None
        groups.extend(
            (entry.leader_idx, num_base_rows + entry.start, num_members, prompt_len)
        )
        for first_beam in range(0, num_members + 1, RAGGED_BLOCK_M):
            tiles.extend((group, first_beam))
        covered_rows += num_members + 1
    if covered_rows != num_rows:
        return None

    num_groups = len(groups) // _GROUP_FIELDS
    num_tiles = len(tiles) // _TILE_FIELDS
    return RaggedBeamAttentionMetadata(
        layout=_layout_tensor(
            [num_groups, num_tiles, num_rows, *groups, *tiles], batch.device
        ),
        group_capacity=num_groups,
        tile_capacity=num_tiles,
        block_m=RAGGED_BLOCK_M,
        num_rows=num_rows,
        req_to_token=req_to_token,
    )


def allocate_capture_layout(
    max_rows: int, device, req_to_token: torch.Tensor
) -> RaggedBeamAttentionMetadata:
    """One layout for every captured row count: only one graph replays at a time."""
    # Every beam group has at least two rows.
    group_capacity = max(1, max_rows // 2)
    tile_capacity = ragged_tile_capacity(max_rows, group_capacity, RAGGED_BLOCK_M)
    size = _NUM_COUNTS + _GROUP_FIELDS * group_capacity + _TILE_FIELDS * tile_capacity
    return RaggedBeamAttentionMetadata(
        layout=torch.zeros(size, dtype=torch.int32, device=device),
        group_capacity=group_capacity,
        tile_capacity=tile_capacity,
        block_m=RAGGED_BLOCK_M,
        num_rows=max_rows,
        req_to_token=req_to_token,
    )


def capture_metadata(
    template: RaggedBeamAttentionMetadata, num_rows: int
) -> RaggedBeamAttentionMetadata:
    """The template's layout for a graph of num_rows rows; zero counts while
    capturing record the launch with its fixed grid and no work."""
    if num_rows > template.num_rows:
        raise ValueError(f"{num_rows} rows exceed the layout's {template.num_rows}")
    template.counts().zero_()
    return msgspec.structs.replace(template, num_rows=num_rows)


def graph_compatible(
    metadata, num_rows: int, target: RaggedBeamAttentionMetadata
) -> bool:
    return (
        isinstance(metadata, RaggedBeamAttentionMetadata)
        and metadata.num_rows == num_rows <= target.num_rows
        and metadata.block_m == target.block_m
        and metadata.group_capacity <= target.group_capacity
        and metadata.tile_capacity <= target.tile_capacity
    )


def stage_graph_metadata(
    target: RaggedBeamAttentionMetadata, source: RaggedBeamAttentionMetadata
) -> None:
    """Copy a step's tables into a captured layout; counts bound what the kernel reads."""
    if not graph_compatible(source, source.num_rows, target):
        raise ValueError("Ragged beam metadata does not fit the captured layout")
    target.counts().copy_(source.counts())
    target.groups()[: source.group_capacity].copy_(source.groups())
    target.tiles()[: source.tile_capacity].copy_(source.tiles())


def forward(
    q: torch.Tensor,
    layer: RadixAttention,
    forward_batch: ForwardBatch,
    metadata: RaggedBeamAttentionMetadata,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
) -> torch.Tensor:
    from sglang.kernels.ops.attention.beam_shared_context_ragged import (
        shared_context_attention_ragged,
    )

    q_heads = layer.tp_q_head_num
    head_dim = layer.head_dim
    rows_q = q.reshape(-1, q_heads, head_dim)
    if rows_q.shape[0] != metadata.num_rows:
        raise RuntimeError(
            f"Q has {rows_q.shape[0]} rows, the beam layout {metadata.num_rows}"
        )
    output = shared_context_attention_ragged(
        rows_q,
        k_buffer,
        v_buffer,
        metadata.req_to_token,
        forward_batch.req_pool_indices,
        forward_batch.seq_lens,
        metadata.counts(),
        metadata.groups(),
        metadata.tiles(),
        block_m=metadata.block_m,
        softmax_scale=layer.scaling,
    )
    return output.view(-1, q_heads * head_dim)
