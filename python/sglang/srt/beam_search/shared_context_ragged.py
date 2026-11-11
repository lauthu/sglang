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
width in the step, and a CUDA graph captured for a row count replays for any
split of those rows into groups. At level 3, every other row of the step (an
ordinary request, or a finished group's leader) is a one-row group whose whole
KV is its prompt. With SGLANG_BEAM_SHARED_CONTEXT_KERNEL=stream_k the layout ends
with the stream-K kernel's schedule: each tile's work units.
"""

from __future__ import annotations

import itertools
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
# SGLANG_BEAM_SHARED_CONTEXT_KERNEL values levels 2 and 3 run.
_RAGGED_KERNELS = ("default", "stream_k")


def _tables_size(group_capacity: int, tile_capacity: int) -> int:
    return _NUM_COUNTS + _GROUP_FIELDS * group_capacity + _TILE_FIELDS * tile_capacity


class RaggedBeamAttentionMetadata(msgspec.Struct):
    # int32 [3 + 4 * group_capacity + 2 * tile_capacity], then the schedule if stream_k
    layout: torch.Tensor
    # A step's own group and tile counts, or the most a captured layout holds.
    group_capacity: int
    tile_capacity: int
    block_m: int
    # Rows of Q the kernel reads: the step's rows, or a captured row count.
    num_rows: int
    req_to_token: torch.Tensor
    # Run the stream-K kernel, which reads the schedule at the layout's end.
    stream_k: bool = False

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

    def schedule(self) -> torch.Tensor:
        return self.layout[_tables_size(self.group_capacity, self.tile_capacity) :]


def decode_graph_level() -> BeamDecodeGraphLevel:
    value = envs.SGLANG_BEAM_DECODE_GRAPH_LEVEL.get()
    if value == 1:
        raise ValueError(
            "SGLANG_BEAM_DECODE_GRAPH_LEVEL=1 (one width per step) has no mode of its "
            "own: use 2, which also replays steps of one width"
        )
    level = BeamDecodeGraphLevel(value)
    if (
        level == BeamDecodeGraphLevel.MIXED_REQUESTS
        and envs.SGLANG_OPT_BEAM_TRIE_TOPK_LOGPROB.get()
    ):
        raise ValueError(
            "SGLANG_BEAM_DECODE_GRAPH_LEVEL=3 samples ordinary requests from the "
            "full-vocabulary logits of the trie head's normalizer pass, which "
            "SGLANG_OPT_BEAM_TRIE_TOPK_LOGPROB=true skips; use level 2"
        )
    kernel = envs.SGLANG_BEAM_SHARED_CONTEXT_KERNEL.get()
    if level != BeamDecodeGraphLevel.FIXED_WIDTH and kernel not in _RAGGED_KERNELS:
        raise ValueError(
            f"SGLANG_BEAM_SHARED_CONTEXT_KERNEL={kernel!r} supports only "
            "SGLANG_BEAM_DECODE_GRAPH_LEVEL=0; levels 2 and 3 run "
            f"{' or '.join(_RAGGED_KERNELS)}"
        )
    return level


def stream_k_selected() -> bool:
    """Levels 2 and 3 run the stream-K kernel instead of the tile-per-CTA one."""
    return envs.SGLANG_BEAM_SHARED_CONTEXT_KERNEL.get() == "stream_k"


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
            "%s have no effect with SGLANG_BEAM_DECODE_GRAPH_LEVEL>=2: beam decode "
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
    batch: ScheduleBatch,
    req_to_token: torch.Tensor,
    *,
    one_row_groups: bool = False,
    stream_k: bool = False,
) -> Optional[RaggedBeamAttentionMetadata]:
    """Group and tile tables of a beam decode step, built on the host from the beam
    tail; None leaves the step to ordinary attention. Every row must belong to a
    beam group, unless one_row_groups lets the other base rows (ordinary requests,
    finished groups' leaders) run as one-row groups. stream_k adds that kernel's
    schedule."""
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
    schedule = _stream_k_schedule(groups, tiles, base_lens) if stream_k else []
    return RaggedBeamAttentionMetadata(
        layout=_layout_tensor(
            [
                len(groups),
                len(tiles),
                num_rows,
                *itertools.chain.from_iterable(groups),
                *itertools.chain.from_iterable(tiles),
                *schedule,
            ],
            batch.device,
        ),
        group_capacity=len(groups),
        tile_capacity=len(tiles),
        block_m=RAGGED_BLOCK_M,
        num_rows=num_rows,
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
        tile_units(prompt_len, base_lens[leader] - prompt_len)
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
    """One layout for every captured row count: only one graph replays at a time."""
    # Every beam group has at least two rows, unless one-row groups are allowed.
    group_capacity = max_rows if one_row_groups else max(1, max_rows // 2)
    tile_capacity = ragged_tile_capacity(max_rows, group_capacity, RAGGED_BLOCK_M)
    size = _tables_size(group_capacity, tile_capacity)
    if stream_k:
        from sglang.kernels.ops.attention.beam_shared_context_ragged_stream_k import (
            schedule_size,
        )

        size += schedule_size(tile_capacity)
    return RaggedBeamAttentionMetadata(
        layout=torch.zeros(size, dtype=torch.int32, device=device),
        group_capacity=group_capacity,
        tile_capacity=tile_capacity,
        block_m=RAGGED_BLOCK_M,
        num_rows=max_rows,
        req_to_token=req_to_token,
        stream_k=stream_k,
    )


def capture_metadata(
    template: RaggedBeamAttentionMetadata, num_rows: int
) -> RaggedBeamAttentionMetadata:
    """The template's layout for a graph of num_rows rows. Capture it with zero
    counts: they record the launch with its fixed grid and no work."""
    if num_rows > template.num_rows:
        raise ValueError(f"{num_rows} rows exceed the layout's {template.num_rows}")
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
        and metadata.stream_k == target.stream_k
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
    if source.stream_k:
        schedule = source.schedule()
        target.schedule()[: len(schedule)].copy_(schedule)


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
    rows_q = q.reshape(-1, q_heads, head_dim)
    if rows_q.shape[0] != metadata.num_rows:
        raise RuntimeError(
            f"Q has {rows_q.shape[0]} rows, the beam layout {metadata.num_rows}"
        )
    inputs = (
        rows_q,
        k_buffer,
        v_buffer,
        metadata.req_to_token,
        forward_batch.req_pool_indices,
        forward_batch.seq_lens,
        metadata.counts(),
        metadata.groups(),
        metadata.tiles(),
    )
    if metadata.stream_k:
        from sglang.kernels.ops.attention.beam_shared_context_ragged_stream_k import (
            shared_context_attention_ragged_stream_k,
        )

        output = shared_context_attention_ragged_stream_k(
            *inputs,
            metadata.schedule(),
            block_m=metadata.block_m,
            softmax_scale=layer.scaling,
        )
    else:
        from sglang.kernels.ops.attention.beam_shared_context_ragged import (
            shared_context_attention_ragged,
        )

        output = shared_context_attention_ragged(
            *inputs, block_m=metadata.block_m, softmax_scale=layer.scaling
        )
    return output.view(-1, q_heads * head_dim)
