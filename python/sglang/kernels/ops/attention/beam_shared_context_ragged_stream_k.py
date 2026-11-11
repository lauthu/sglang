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
"""Stream-K shared-context attention for any split of a step's rows into groups.

Opt-in with SGLANG_BEAM_SHARED_CONTEXT_KERNEL=stream_k at
SGLANG_BEAM_DECODE_GRAPH_LEVEL 2 or 3. It reads the layout of
beam_shared_context_ragged.py plus a schedule, which build_schedule writes:
- schedule: [most units of a tile, units of all tiles, unit_starts[0], ...,
  unit_starts[tiles]], per query head; unit_starts[t] is tile t's first unit

A tile and query head has one setup unit, then one unit per BLOCK_N prompt tokens
and one per decode step of its group; units run tile by tile, head by head. The
grid is persistent, one CTA per SM, and each CTA takes an equal run of the step's
units, which may end inside a tile and head. When whole tiles and heads per CTA
cost no more (the tiles already fill the SMs), the runs end at their bounds
instead. A tile and head covered by one CTA is written directly. One split across
CTAs leaves a partial output and log-sum-exp per CTA, and the CTA holding its first
work unit lists it for the merge kernel. Only a run's first and last segments can
be partial, so the workspace holds two slots per CTA for any step.
"""

from __future__ import annotations

import itertools
from typing import Sequence

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.beam_shared_context import (
    attend_beam_token,
    attend_prompt_block,
    empty_softmax_state,
    num_sms,
)
from sglang.kernels.ops.attention.beam_shared_context_ragged import (
    check_ragged_inputs,
)

# Measured on RTX PRO 6000 (SM120), Qwen3-0.6B heads, 1-8 groups of 1-2,000 beams.
_BLOCK_N = 64
_NUM_WARPS = 8
_NUM_STAGES = 2
_SETUP_UNITS = 1
# Units a split adds to its share: a second Q load, partial stores and the merge.
# Measured: 2 picks the faster of the split and whole tiles at all 39 benchmarked
# steps of 1-8 requests with 1-2,000 beams (0-16 tried).
_SPLIT_COST_UNITS = 2
# Measured: 8-row merge programs are fastest at 1-4 groups (8-64 tried).
_MERGE_ROWS = 8
# Tiles read per step of the tile search; one step covers most decode steps.
_TILE_SEARCH = 128
# Schedule fields before unit_starts.
_SCHEDULE_HEADER = 2
# Merge list fields per CTA: tile (-1: none), query head, last CTA, first slot.
_MERGE_FIELDS = 4
# K/V offsets are 32-bit: 64-bit ones cost the registers of a third pipeline stage.
_MAX_OFFSET = 2**31


def tile_units(prompt_len: int, decode_len: int) -> int:
    """Work units of one tile and query head of a group."""
    return _SETUP_UNITS + -(-prompt_len // _BLOCK_N) + decode_len


def build_schedule(units_per_tile: Sequence[int]) -> list[int]:
    """The schedule of a step whose tiles, in order, have these `tile_units`."""
    unit_starts = [0, *itertools.accumulate(units_per_tile)]
    return [max(units_per_tile), unit_starts[-1], *unit_starts]


def schedule_size(num_tiles: int) -> int:
    return _SCHEDULE_HEADER + num_tiles + 1


@triton.jit
def _tile_head_start(unit_starts, tile_head, Q_HEADS: tl.constexpr):
    """First unit of tile and head number tile_head (tile-major); all units' end
    for the number past the last."""
    tile = tile_head // Q_HEADS
    head = tile_head % Q_HEADS
    start = tl.load(unit_starts + tile)
    units = tl.load(unit_starts + tile + 1, mask=head > 0, other=start) - start
    return start * Q_HEADS + head * units


@triton.jit
def _tile_head_of(
    unit_starts, num_tiles, unit, Q_HEADS: tl.constexpr, TILE_SEARCH: tl.constexpr
):
    """Tile-major number of the tile and head that holds unit."""
    tile = tl.maximum(
        _find_tile(unit_starts, num_tiles, unit // Q_HEADS, TILE_SEARCH), 0
    )
    start = tl.load(unit_starts + tile)
    units = tl.maximum(tl.load(unit_starts + tile + 1) - start, 1)
    return tile * Q_HEADS + (unit - start * Q_HEADS) // units


@triton.jit
def _cta_run(
    schedule,
    unit_starts,
    num_tiles,
    cta,
    num_ctas,
    Q_HEADS: tl.constexpr,
    SPLIT_COST_UNITS: tl.constexpr,
):
    """This CTA's units [run_start, run_end), the CTAs splitting them and all units.
    Equal runs over all CTAs, unless whole tiles and heads per CTA, in as few waves,
    are no slower; then each run ends at a tile and head bound."""
    max_units = tl.load(schedule)
    total = tl.load(schedule + 1) * Q_HEADS
    tile_heads = num_tiles * Q_HEADS
    waves = tl.maximum(tl.cdiv(tile_heads, num_ctas), 1)
    if waves * max_units <= tl.cdiv(total, num_ctas) + SPLIT_COST_UNITS:
        first = tl.minimum(cta * waves, tile_heads)
        run_start = _tile_head_start(unit_starts, first, Q_HEADS)
        run_end = _tile_head_start(
            unit_starts, tl.minimum(first + waves, tile_heads), Q_HEADS
        )
        active = tl.maximum(tl.cdiv(tile_heads, waves), 1)
    else:
        run_start = ((cta.to(tl.int64) * total) // num_ctas).to(tl.int32)
        run_end = (((cta + 1).to(tl.int64) * total) // num_ctas).to(tl.int32)
        active = num_ctas
    return run_start, run_end, active, total


@triton.jit
def _find_tile(unit_starts, num_tiles, key, TILE_SEARCH: tl.constexpr):
    """Last tile whose first unit (per head) is at most key; unit_starts increases."""
    found = -1
    for base in range(0, num_tiles, TILE_SEARCH):
        offsets = base + tl.arange(0, TILE_SEARCH)
        valid = offsets < num_tiles
        starts = tl.load(unit_starts + offsets, mask=valid, other=0)
        found += tl.sum(((starts <= key) & valid).to(tl.int32))
    return found


@triton.jit
def _tile_rows(groups, tiles, tile, lanes):
    """Rows of a tile's beams and their mask, then its group's leader row and prompt
    length, as beam_shared_context_ragged.py reads the tables."""
    group = tl.load(tiles + 2 * tile)
    first_beam = tl.load(tiles + 2 * tile + 1)
    leader = tl.load(groups + 4 * group)
    first_member = tl.load(groups + 4 * group + 1)
    num_members = tl.load(groups + 4 * group + 2)
    prompt_len = tl.load(groups + 4 * group + 3)
    beams = first_beam + lanes
    beam_mask = beams <= num_members
    rows = tl.where(beams == 0, leader, first_member + beams - 1)
    rows = tl.where(beam_mask, rows, leader)
    return rows, beam_mask, leader, prompt_len


@triton.jit
def _attend_tile_head(
    q,
    k_buffer,
    v_buffer,
    req_to_token,
    req_pool_indices,
    seq_lens,
    groups,
    tiles,
    tile,
    q_head,
    work_start,
    work_end,
    softmax_scale_log2,
    stride_q_row,
    stride_q_head,
    stride_k_slot,
    stride_k_head,
    stride_v_slot,
    stride_v_head,
    stride_req_to_token,
    Q_PER_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Softmax state of a tile and query head over its work units
    [work_start, work_end): prompt blocks first, then decode steps."""
    kv_head = q_head // Q_PER_KV
    lanes = tl.arange(0, BLOCK_M)
    context_offsets = tl.arange(0, BLOCK_N)
    dims = tl.arange(0, BLOCK_D)
    dim_mask = dims < HEAD_DIM
    negative_infinity = -float("inf")

    rows, beam_mask, leader, prompt_len = _tile_rows(groups, tiles, tile, lanes)
    row_mask = beam_mask[:, None] & dim_mask[None, :]
    q_values = tl.load(
        q + rows[:, None] * stride_q_row + q_head * stride_q_head + dims[None, :],
        mask=row_mask,
        other=0.0,
    )
    max_values, normalizers, accumulator = empty_softmax_state(
        beam_mask, BLOCK_M, BLOCK_D
    )

    num_blocks = tl.cdiv(prompt_len, BLOCK_N)
    leader_pool = tl.load(req_pool_indices + leader).to(tl.int64)
    prompt_slots = req_to_token + leader_pool * stride_req_to_token
    for block in tl.range(work_start, tl.minimum(work_end, num_blocks)):
        tokens = block * BLOCK_N + context_offsets
        token_mask = tokens < prompt_len
        slots = tl.load(prompt_slots + tokens, mask=token_mask, other=0)
        kv_mask = token_mask[:, None] & dim_mask[None, :]
        k_values = tl.load(
            k_buffer
            + slots[:, None] * stride_k_slot
            + kv_head * stride_k_head
            + dims[None, :],
            mask=kv_mask,
            other=0.0,
        )
        v_values = tl.load(
            v_buffer
            + slots[:, None] * stride_v_slot
            + kv_head * stride_v_head
            + dims[None, :],
            mask=kv_mask,
            other=0.0,
        )
        scores = tl.dot(q_values, tl.trans(k_values))
        scores = tl.where(
            beam_mask[:, None] & token_mask[None, :], scores, negative_infinity
        )
        max_values, normalizers, accumulator = attend_prompt_block(
            scores,
            v_values,
            max_values,
            normalizers,
            accumulator,
            softmax_scale_log2,
        )

    # Every beam of a group sits at the same depth: the leader's length.
    decode_len = tl.load(seq_lens + leader).to(tl.int32) - prompt_len
    row_pools = tl.load(req_pool_indices + rows, mask=beam_mask, other=0).to(tl.int64)
    decode_slots = req_to_token + row_pools * stride_req_to_token + prompt_len
    for step in range(
        tl.maximum(work_start - num_blocks, 0),
        tl.minimum(work_end - num_blocks, decode_len),
    ):
        slots = tl.load(decode_slots + step, mask=beam_mask, other=0)
        k_values = tl.load(
            k_buffer
            + slots[:, None] * stride_k_slot
            + kv_head * stride_k_head
            + dims[None, :],
            mask=row_mask,
            other=0.0,
        )
        v_values = tl.load(
            v_buffer
            + slots[:, None] * stride_v_slot
            + kv_head * stride_v_head
            + dims[None, :],
            mask=row_mask,
            other=0.0,
        )
        scores = tl.sum(q_values.to(tl.float32) * k_values.to(tl.float32), axis=1)
        scores = tl.where(beam_mask, scores, negative_infinity)
        max_values, normalizers, accumulator = attend_beam_token(
            scores,
            v_values,
            max_values,
            normalizers,
            accumulator,
            softmax_scale_log2,
        )
    return rows, beam_mask, max_values, normalizers, accumulator


@triton.jit
def _ragged_stream_k_kernel(
    q,
    k_buffer,
    v_buffer,
    req_to_token,
    req_pool_indices,
    seq_lens,
    counts,
    groups,
    tiles,
    schedule,
    output,
    partial_output,
    partial_lse,
    merge_list,
    softmax_scale,
    num_rows,
    num_ctas,
    stride_q_row,
    stride_q_head,
    stride_k_slot,
    stride_k_head,
    stride_v_slot,
    stride_v_head,
    stride_req_to_token,
    stride_out_row,
    stride_out_head,
    Q_HEADS: tl.constexpr,
    Q_PER_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    SETUP_UNITS: tl.constexpr,
    SPLIT_COST_UNITS: tl.constexpr,
    TILE_SEARCH: tl.constexpr,
    SCHEDULE_HEADER: tl.constexpr,
    MERGE_FIELDS: tl.constexpr,
):
    """Program c attends the units of its run and lists the tile and head it leaves
    unfinished, if any, for the merge kernel."""
    cta = tl.program_id(0)
    lanes = tl.arange(0, BLOCK_M)
    dims = tl.arange(0, BLOCK_D)
    dim_mask = dims < HEAD_DIM
    softmax_scale_log2 = softmax_scale * 1.4426950408889634
    unit_starts = schedule + SCHEDULE_HEADER

    num_tiles = tl.load(counts + 1)
    run_start, run_end, active, total = _cta_run(
        schedule, unit_starts, num_tiles, cta, num_ctas, Q_HEADS, SPLIT_COST_UNITS
    )
    first_tile_head = _tile_head_of(
        unit_starts, num_tiles, run_start, Q_HEADS, TILE_SEARCH
    )
    last_tile_head = _tile_head_of(
        unit_starts, num_tiles, tl.maximum(run_end - 1, 0), Q_HEADS, TILE_SEARCH
    )
    # When the run's last segment holds the first work unit of a tile and head that
    # later runs finish, list it for the merge, up to the CTA owning its last unit.
    head_start = _tile_head_start(unit_starts, last_tile_head, Q_HEADS)
    head_end = _tile_head_start(unit_starts, last_tile_head + 1, Q_HEADS)
    listed = (
        (run_start < run_end)
        & (tl.maximum(run_start, head_start) <= head_start + SETUP_UNITS)
        & (head_start + SETUP_UNITS < run_end)
        & (run_end < head_end)
    )
    entry = merge_list + MERGE_FIELDS * cta
    tl.store(entry, tl.where(listed, last_tile_head // Q_HEADS, -1))
    tl.store(entry + 1, last_tile_head % Q_HEADS)
    last_cta = (head_end.to(tl.int64) * active - 1) // tl.maximum(total, 1)
    tl.store(entry + 2, last_cta.to(tl.int32))
    tl.store(entry + 3, 2 * cta + (run_start < head_start).to(tl.int32))
    if run_start == run_end:
        # A merge spanning this CTA finds no segment here.
        tl.store(
            partial_lse + 2 * cta * BLOCK_M + lanes,
            tl.full((BLOCK_M,), -float("inf"), tl.float32),
        )
        last_tile_head = first_tile_head - 1

    # Partial slot 2 * cta holds the run's first segment, 2 * cta + 1 its last.
    partial_rows = partial_output + 2 * cta * BLOCK_M * HEAD_DIM
    partial_lse_rows = partial_lse + 2 * cta * BLOCK_M
    for tile_head in range(first_tile_head, last_tile_head + 1):
        head_start = _tile_head_start(unit_starts, tile_head, Q_HEADS)
        head_end = _tile_head_start(unit_starts, tile_head + 1, Q_HEADS)
        # Work units of the tile and head in this run; a run that holds only
        # their setup unit is skipped.
        work_start = (
            tl.maximum(tl.maximum(run_start, head_start) - head_start, SETUP_UNITS)
            - SETUP_UNITS
        )
        work_end = tl.minimum(run_end, head_end) - head_start - SETUP_UNITS
        if work_start < work_end:
            q_head = tile_head % Q_HEADS
            rows, beam_mask, max_values, normalizers, accumulator = _attend_tile_head(
                q,
                k_buffer,
                v_buffer,
                req_to_token,
                req_pool_indices,
                seq_lens,
                groups,
                tiles,
                tile_head // Q_HEADS,
                q_head,
                work_start,
                work_end,
                softmax_scale_log2,
                stride_q_row,
                stride_q_head,
                stride_k_slot,
                stride_k_head,
                stride_v_slot,
                stride_v_head,
                stride_req_to_token,
                Q_PER_KV,
                HEAD_DIM,
                BLOCK_M,
                BLOCK_N,
                BLOCK_D,
            )
            row_mask = beam_mask[:, None] & dim_mask[None, :]
            if (work_start == 0) & (run_end >= head_end):
                tl.store(
                    output
                    + rows[:, None] * stride_out_row
                    + q_head * stride_out_head
                    + dims[None, :],
                    accumulator / normalizers[:, None],
                    mask=row_mask,
                )
            else:
                lane_offset = (run_start < head_start).to(tl.int32) * BLOCK_M + lanes
                tl.store(
                    partial_rows + (lane_offset * HEAD_DIM)[:, None] + dims[None, :],
                    accumulator / normalizers[:, None],
                    mask=row_mask,
                )
                # Every work unit holds a token, so every segment has softmax mass.
                tl.store(
                    partial_lse_rows + lane_offset,
                    max_values * softmax_scale_log2 + tl.log2(normalizers),
                    mask=beam_mask,
                )

    # Rows past the real ones pad a captured row count; zero them.
    num_real_rows = tl.load(counts + 2)
    for row_start in range(num_real_rows + cta * BLOCK_M, num_rows, num_ctas * BLOCK_M):
        pad_rows = row_start + lanes
        for pad_head in range(0, Q_HEADS):
            tl.store(
                output
                + pad_rows[:, None] * stride_out_row
                + pad_head * stride_out_head
                + dims[None, :],
                tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32),
                mask=(pad_rows < num_rows)[:, None] & dim_mask[None, :],
            )


@triton.jit
def _ragged_merge_kernel(
    partial_output,
    partial_lse,
    merge_list,
    groups,
    tiles,
    output,
    stride_out_row,
    stride_out_head,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
    MERGE_ROWS: tl.constexpr,
    MERGE_FIELDS: tl.constexpr,
):
    """Program (c, chunk) merges MERGE_ROWS rows of the tile and head CTA c listed:
    its slot, then slot 2 * c' of every later CTA c' up to the listed last one."""
    cta = tl.program_id(0)
    entry = merge_list + MERGE_FIELDS * cta
    tile = tl.load(entry)
    if tile >= 0:
        q_head = tl.load(entry + 1)
        last_cta = tl.load(entry + 2)
        first_slot = tl.load(entry + 3)
        lanes = tl.program_id(1) * MERGE_ROWS + tl.arange(0, MERGE_ROWS)
        rows, beam_mask, _, _ = _tile_rows(groups, tiles, tile, lanes)
        beam_mask = beam_mask & (lanes < BLOCK_M)
        if tl.sum(beam_mask.to(tl.int32)) > 0:
            dims = tl.arange(0, BLOCK_D)
            mask = beam_mask[:, None] & (dims < HEAD_DIM)[None, :]
            negative_infinity = -float("inf")
            max_lse = tl.full((MERGE_ROWS,), negative_infinity, tl.float32)
            total_weight = tl.zeros((MERGE_ROWS,), tl.float32)
            accumulator = tl.zeros((MERGE_ROWS, BLOCK_D), tl.float32)
            for owner in range(cta, last_cta + 1):
                slot = tl.where(owner == cta, first_slot, 2 * owner)
                lse = tl.load(
                    partial_lse + slot * BLOCK_M + lanes,
                    mask=beam_mask,
                    other=negative_infinity,
                )
                # A CTA without units wrote -inf and no output.
                part = tl.load(
                    partial_output
                    + ((slot * BLOCK_M + lanes) * HEAD_DIM)[:, None]
                    + dims[None, :],
                    mask=mask & (lse > negative_infinity)[:, None],
                    other=0.0,
                )
                new_max = tl.maximum(max_lse, lse)
                safe_max = tl.where(new_max == negative_infinity, 0.0, new_max)
                previous_scale = tl.exp2(max_lse - safe_max)
                weight = tl.exp2(lse - safe_max)
                accumulator = (
                    accumulator * previous_scale[:, None] + part * weight[:, None]
                )
                total_weight = total_weight * previous_scale + weight
                max_lse = new_max
            tl.store(
                output
                + rows[:, None] * stride_out_row
                + q_head * stride_out_head
                + dims[None, :],
                accumulator / total_weight[:, None],
                mask=mask,
            )


def shared_context_attention_ragged_stream_k(
    q: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    counts: torch.Tensor,
    groups: torch.Tensor,
    tiles: torch.Tensor,
    schedule: torch.Tensor,
    *,
    block_m: int,
    output: torch.Tensor | None = None,
    softmax_scale: float,
    num_ctas: int | None = None,
    split_cost_units: int = _SPLIT_COST_UNITS,
) -> torch.Tensor:
    """shared_context_attention_ragged on a persistent stream-K grid.

    `schedule` is [most units of a tile, units of all tiles, unit_starts...] for the
    first counts[1] + 1 tiles, per query head (`tile_units` per tile); rows past
    `counts[2]` are written as zeros. `num_ctas` defaults to one CTA per SM; a very
    large or negative `split_cost_units` forces whole tiles or the split.
    """
    output = check_ragged_inputs(
        q,
        k_buffer,
        v_buffer,
        req_to_token,
        req_pool_indices,
        seq_lens,
        counts,
        groups,
        tiles,
        output,
    )
    if schedule.dtype != torch.int32 or schedule.shape != (
        schedule_size(tiles.shape[0]),
    ):
        raise ValueError("Expected an int32 schedule [2 + tile capacity + 1]")
    if max(k_buffer.numel(), v_buffer.numel()) >= _MAX_OFFSET:
        raise ValueError(
            f"K/V buffers of {k_buffer.numel()} elements reach 2^31; this kernel "
            "indexes them with 32-bit offsets"
        )
    num_rows, q_heads, head_dim = q.shape
    kv_heads = k_buffer.shape[1]

    if num_ctas is None:
        # One CTA per SM: 8 warps of this kernel fill an SM's register file.
        num_ctas = num_sms(q.device.index)
    # Two slots per CTA (its run's first and last segments) and one merge entry.
    partial_output = torch.empty(
        (2 * num_ctas, block_m, head_dim), dtype=torch.float32, device=q.device
    )
    partial_lse = torch.empty(
        (2 * num_ctas, block_m), dtype=torch.float32, device=q.device
    )
    merge_list = torch.empty(
        (num_ctas, _MERGE_FIELDS), dtype=torch.int32, device=q.device
    )
    block_d = triton.next_power_of_2(head_dim)
    _ragged_stream_k_kernel[(num_ctas,)](
        q,
        k_buffer,
        v_buffer,
        req_to_token,
        req_pool_indices,
        seq_lens,
        counts,
        groups,
        tiles,
        schedule,
        output,
        partial_output,
        partial_lse,
        merge_list,
        softmax_scale,
        num_rows,
        num_ctas,
        q.stride(0),
        q.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        v_buffer.stride(0),
        v_buffer.stride(1),
        req_to_token.stride(0),
        output.stride(0),
        output.stride(1),
        Q_HEADS=q_heads,
        Q_PER_KV=q_heads // kv_heads,
        HEAD_DIM=head_dim,
        BLOCK_M=block_m,
        BLOCK_N=_BLOCK_N,
        BLOCK_D=block_d,
        SETUP_UNITS=_SETUP_UNITS,
        SPLIT_COST_UNITS=split_cost_units,
        TILE_SEARCH=_TILE_SEARCH,
        SCHEDULE_HEADER=_SCHEDULE_HEADER,
        MERGE_FIELDS=_MERGE_FIELDS,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )
    _ragged_merge_kernel[(num_ctas, triton.cdiv(block_m, _MERGE_ROWS))](
        partial_output,
        partial_lse,
        merge_list,
        groups,
        tiles,
        output,
        output.stride(0),
        output.stride(1),
        HEAD_DIM=head_dim,
        BLOCK_M=block_m,
        BLOCK_D=block_d,
        MERGE_ROWS=_MERGE_ROWS,
        MERGE_FIELDS=_MERGE_FIELDS,
        num_warps=4,
    )
    return output
