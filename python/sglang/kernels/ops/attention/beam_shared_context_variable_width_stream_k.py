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
"""Stream-K shared-context attention for beam requests of any widths.

Opt-in with SGLANG_BEAM_SHARED_CONTEXT_KERNEL=stream_k on a server whose beam
decode graphs take any beam width; every batch with the variable-width layout
there runs it, replayed or eager. It computes what the default kernel
(beam_shared_context_variable_width.py) computes, from the same layout:
- The default kernel gives a program whole tiles, so one request with a long
  prompt keeps a few SMs busy and leaves the others idle.
- This kernel has one CTA per SM for any batch and shares the batch's work evenly
  over them.

Terms (group, tile, pool row, real rows, work unit and the layout are defined in
the default kernel's module):
- range: the consecutive work units that one CTA attends. Work units are ordered
  by tile, then by query head, then within their (tile, head). A range has a
  start and an end, each an index (tile, head, work unit); the end is exclusive.
- partial slot: where a CTA leaves its part of a (tile, head) that its range does
  not hold whole. CTA c has slots 2 * c and 2 * c + 1, so the workspace has one
  size for any batch.
- merge end of CTA c: the first CTA whose range starts after the (tile, head)
  that CTA c's range ends in. The parts of that (tile, head) are in the CTAs
  before it.
- plan: one row per CTA, with its range and its merge end. The host builds it
  once per decode step (build_stream_k_plan) and puts it in the layout after the
  header; every layer's launch reads it.

How the kernels read the row of CTA c:
1. Attend every (tile, head) from the range's start to its end:
   - from work unit 0, except in the start's (tile, head);
   - to the last work unit, except in the end's (tile, head), which is attended
     only if the end's work unit is above 0.
2. Store each result:
   - a (tile, head) attended from work unit 0 to its last: to the output;
   - a part that starts after work unit 0: to slot 2 * c;
   - a part that starts at work unit 0 and stops early: to slot 2 * c + 1.
   A CTA with an empty range (its start is its end) attends nothing and marks
   slot 2 * c empty.
3. Merge, if the range ends inside a (tile, head) that it started: the merge
   kernel combines slot 2 * c + 1 with slot 2 * k of every CTA k after c and
   before the merge end. The row holds the merge end so that the merge kernel
   reads no other row to find it.

How the host places the ranges:
- The global line holds every (tile, head) in order: _SETUP_UNITS entries for its
  setup (S below), then one entry per work unit. A position on it is a global
  index.
- The plan cuts the line into ranges of equal length, so a range may start and
  end inside a (tile, head). Where ranges that end only between (tile, head)
  pairs load no CTA more than that, it takes those instead
  (_range_global_starts).
- A range that starts inside a setup starts at that (tile, head)'s first work
  unit.

Example: 1 tile, 2 heads, 4 prompt blocks (P) and 6 decode steps (D), 3 CTAs.

  global index  0  1  2  3  4  5  6  7  8  9  10 11 12 13 14 15 16 17 18 19 20 21
  entry         S  P0 P1 P2 P3 D0 D1 D2 D3 D4 D5 S  P0 P1 P2 P3 D0 D1 D2 D3 D4 D5
  work unit     -  0  1  2  3  4  5  6  7  8  9  -  0  1  2  3  4  5  6  7  8  9
  query head    0  0  0  0  0  0  0  0  0  0  0  1  1  1  1  1  1  1  1  1  1  1
  CTA           0  0  0  0  0  0  0  1  1  1  1  1  1  1  2  2  2  2  2  2  2  2
  partial slot  -  1  1  1  1  1  1  2  2  2  2  -  3  3  4  4  4  4  4  4  4  4

- The ranges start at global indices 0, 7 and 14; the global length is 22.
- The plan is [[0, 0, 0, 0, 0, 6, 2], [0, 0, 6, 0, 1, 2, 3], [0, 1, 2, 1, 0, 0, 4]].
  CTA 1's range starts at (tile 0, head 0, work unit 6) and ends at (tile 0,
  head 1, work unit 2). Its merge end is 3: the parts of (tile 0, head 1) are in
  CTAs 1 and 2.
- The merge kernel combines slots 1 and 2 for head 0, slots 3 and 4 for head 1.

A zeroed plan gives every CTA an empty range and nothing to merge, so a graph
captured on a zeroed layout attends nothing.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.beam_shared_context import (
    max_element_offset,
    num_sms,
)
from sglang.kernels.ops.attention.beam_shared_context_variable_width import (
    TILE_FIELDS,
    WHOLE_TILE,
    attend_tile,
    check_variable_width_inputs,
    read_tile,
    store_output,
    zero_padding_rows,
)

# The fixed-width stream-K kernel's values (beam_shared_context_stream_k.py), not
# re-tuned for this kernel.
_BLOCK_N = 64
_NUM_WARPS = 8
_NUM_STAGES = 2
# Entries of the global line that starting a (tile, head) takes: a Q load and an
# output store. Also the fixed-width stream-K kernel's value.
_SETUP_UNITS = 1
# What splitting a (tile, head) adds to a CTA's load, in entries of the global
# line: a second Q load, the partial stores and the merge. Chosen by measurement.
_SPLIT_COST_UNITS = 2
# Beams per merge program. Chosen by measurement.
_MERGE_BEAMS = 8
# K/V offsets are int32 while every element of the caches is below this offset
# (attend_tile's INT64_KV_OFFSETS); int32 offsets take fewer registers.
_INT32_OFFSET_LIMIT = 2**31

# int32 fields of a plan row, in the order the kernels read them. The CTA's range:
#   0 start tile, 1 start head, 2 start work unit
#   3 end tile, 4 end head, 5 end work unit (exclusive)
# and for the merge kernel:
#   6 merge end, the first CTA whose range starts after the end's (tile, head)
_PLAN_FIELDS = 7


def default_num_ctas(device_index: int) -> int:
    """The grid of the stream-K kernel on a device: one CTA per SM."""
    return num_sms(device_index)


def work_units(prompt_len: int, decode_len: int) -> int:
    """Work units of a tile for one query head, from its group's prompt length and
    the tokens each of its beams has generated."""
    return triton.cdiv(prompt_len, _BLOCK_N) + decode_len


def plan_len(num_ctas: int) -> int:
    """int32 entries of the plan of a grid of num_ctas CTAs."""
    return _PLAN_FIELDS * num_ctas


def _range_global_starts(
    tile_qhead_global_starts: np.ndarray, longest_tile_qhead_len: int, num_ctas: int
) -> np.ndarray:
    """The global index where each CTA's range starts, then the global length,
    [num_ctas + 1].

    tile_qhead_global_starts is the global index where each (tile, head) starts,
    then the global length. Two ways to place the ranges:
    1. Between (tile, head) pairs: CTA c takes pairs [c * n, (c + 1) * n), with
       n = ceil(pairs / CTAs). No pair is split; the last CTAs get fewer or none.
    2. Equal ranges: every CTA takes the same length of the global line. A pair
       that is split costs _SPLIT_COST_UNITS more.
    The first way is taken unless splitting saves the busiest CTA more than that
    cost, counting n of the longest pair as its load. That is when the pairs are
    short, or when their count is at or just under a multiple of the CTA count.
    """
    num_tile_qheads = len(tile_qhead_global_starts) - 1
    global_len = int(tile_qhead_global_starts[-1])
    ctas = np.arange(num_ctas + 1, dtype=np.int64)
    tile_qheads_per_cta = triton.cdiv(num_tile_qheads, num_ctas)

    between_tile_qheads_cost = tile_qheads_per_cta * longest_tile_qhead_len
    equal_ranges_cost = triton.cdiv(global_len, num_ctas) + _SPLIT_COST_UNITS
    if between_tile_qheads_cost <= equal_ranges_cost:
        return tile_qhead_global_starts[
            np.minimum(ctas * tile_qheads_per_cta, num_tile_qheads)
        ]
    return ctas * global_len // num_ctas


def build_stream_k_plan(
    work_units_per_tile: Sequence[int], *, q_heads: int, num_ctas: int
) -> np.ndarray:
    """The plan of a batch, [num_ctas, _PLAN_FIELDS] int32: each CTA's range and
    merge end.

    work_units_per_tile[t] is the `work_units` of tile t, at least one.
    """
    # The global line: every (tile, head) in order, each taking _SETUP_UNITS global
    # indices for its setup, then one per work unit.
    tile_qhead_lens = np.repeat(
        np.asarray(work_units_per_tile, dtype=np.int64) + _SETUP_UNITS, q_heads
    )
    tile_qhead_global_starts = np.concatenate(([0], np.cumsum(tile_qhead_lens)))
    range_global_starts = _range_global_starts(
        tile_qhead_global_starts,
        longest_tile_qhead_len=int(tile_qhead_lens.max()),
        num_ctas=num_ctas,
    )
    # Each global start as an index (tile, head, work unit). One inside a setup is
    # the (tile, head)'s first work unit; the global length is (num_tiles, 0, 0).
    tile_qhead = (
        np.searchsorted(tile_qhead_global_starts, range_global_starts, side="right") - 1
    )
    work_unit = np.maximum(
        range_global_starts - tile_qhead_global_starts[tile_qhead] - _SETUP_UNITS, 0
    )
    range_starts = np.stack(
        [tile_qhead // q_heads, tile_qhead % q_heads, work_unit], axis=1
    )
    # CTA c's range ends where CTA c + 1's starts.
    range_ends = range_starts[1:]
    # The first CTA whose range starts after the (tile, head) of each end.
    merge_end = np.searchsorted(tile_qhead, tile_qhead[1:], side="right")
    return np.concatenate(
        (range_starts[:-1], range_ends, merge_end[:, None]), axis=1
    ).astype(np.int32)


@triton.jit
def _variable_width_stream_k_kernel(
    q,  # [rows, q_heads, HEAD_DIM]; only the real rows are read
    k_buffer,  # [kv_slots, kv_heads, HEAD_DIM], the layer's K cache
    v_buffer,  # [kv_slots, kv_heads, HEAD_DIM], the layer's V cache
    req_to_token,  # [pool rows, max context] int32, the KV slot of each token
    req_pool_indices,  # [rows] int32 or int64, each row's pool row
    seq_lens,  # [rows] int32 or int64; only the leaders' are read
    row_count,  # [1] int32, the batch's real rows
    tiles,  # [tiles, TILE_FIELDS] int32, the layout's tiles
    plan,  # [num_ctas, PLAN_FIELDS] int32, the layout's plan
    output,  # like q; unsplit (tile, head) pairs and the padding rows are written
    partial_output,  # [2 * num_ctas, BLOCK_M, HEAD_DIM] fp32, the parts' outputs
    partial_lse,  # [2 * num_ctas, BLOCK_M] fp32, log2 of the parts' softmax sums
    softmax_scale,  # float, scales q.k before the softmax
    num_rows,  # q.shape[0]
    num_ctas,  # the grid size, which the plan was built for
    stride_q_row,  # q.stride(0); all strides are in elements
    stride_q_head,  # q.stride(1)
    stride_k_slot,  # k_buffer.stride(0)
    stride_k_head,  # k_buffer.stride(1)
    stride_v_slot,  # v_buffer.stride(0)
    stride_v_head,  # v_buffer.stride(1)
    stride_req_to_token,  # req_to_token.stride(0)
    stride_out_row,  # output.stride(0)
    stride_out_head,  # output.stride(1)
    Q_HEADS: tl.constexpr,
    Q_PER_KV: tl.constexpr,  # query heads per KV head
    HEAD_DIM: tl.constexpr,  # q.shape[2]
    BLOCK_M: tl.constexpr,  # beams per tile, the value the layout was built with
    BLOCK_N: tl.constexpr,  # prompt tokens per block
    BLOCK_D: tl.constexpr,  # HEAD_DIM rounded up to a power of 2
    TILE_FIELDS: tl.constexpr,
    PLAN_FIELDS: tl.constexpr,  # int32 entries of a plan row
    WHOLE_TILE: tl.constexpr,
    INT64_KV_OFFSETS: tl.constexpr,  # for K/V caches of 2^31 elements or more
):
    """CTA c attends its range, the plan's row c. A (tile, head) that lies in the
    range goes to the output; a part of one goes to a partial slot, for the merge
    kernel."""
    cta = tl.program_id(0)
    beam_offsets = tl.arange(0, BLOCK_M)
    dim_offsets = tl.arange(0, BLOCK_D)
    dim_mask = dim_offsets < HEAD_DIM
    # exp2 is cheaper than exp: fold log2(e) into the scale.
    softmax_scale_log2 = softmax_scale * 1.4426950408889634
    plan_row = plan + PLAN_FIELDS * cta
    start_tile = tl.load(plan_row)
    start_head = tl.load(plan_row + 1)
    start_work = tl.load(plan_row + 2)
    end_tile = tl.load(plan_row + 3)
    end_head = tl.load(plan_row + 4)
    end_work = tl.load(plan_row + 5)

    # The range's (tile, head) pairs in order, numbered tile * Q_HEADS + head. The
    # end's pair is attended only if the range ends inside it.
    first_tile_qhead = start_tile * Q_HEADS + start_head
    end_tile_qhead = end_tile * Q_HEADS + end_head
    is_empty = (first_tile_qhead == end_tile_qhead) & (start_work == end_work)
    stop_tile_qhead = tl.where(
        is_empty, first_tile_qhead, end_tile_qhead + tl.where(end_work > 0, 1, 0)
    )

    if is_empty:
        # The merge of a (tile, head) split around this CTA still reads its slot
        # 2 * cta.
        tl.store(
            partial_lse + 2 * cta * BLOCK_M + beam_offsets,
            tl.full((BLOCK_M,), -float("inf"), tl.float32),
        )

    for tile_qhead in range(first_tile_qhead, stop_tile_qhead):
        q_head = tile_qhead % Q_HEADS
        work_start = tl.where(tile_qhead == first_tile_qhead, start_work, 0)
        work_end = tl.where(tile_qhead == end_tile_qhead, end_work, WHOLE_TILE)
        rows, beam_mask, max_values, normalizers, accumulator = attend_tile(
            q=q,
            k_buffer=k_buffer,
            v_buffer=v_buffer,
            req_to_token=req_to_token,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            tiles=tiles,
            tile_idx=tile_qhead // Q_HEADS,
            q_head=q_head,
            work_start=work_start,
            work_end=work_end,
            softmax_scale_log2=softmax_scale_log2,
            stride_q_row=stride_q_row,
            stride_q_head=stride_q_head,
            stride_k_slot=stride_k_slot,
            stride_k_head=stride_k_head,
            stride_v_slot=stride_v_slot,
            stride_v_head=stride_v_head,
            stride_req_to_token=stride_req_to_token,
            Q_PER_KV=Q_PER_KV,
            HEAD_DIM=HEAD_DIM,
            BLOCK_M=BLOCK_M,
            BLOCK_N=BLOCK_N,
            BLOCK_D=BLOCK_D,
            TILE_FIELDS=TILE_FIELDS,
            INT64_KV_OFFSETS=INT64_KV_OFFSETS,
        )
        output_mask = beam_mask[:, None] & dim_mask[None, :]
        if (work_start > 0) | (work_end < WHOLE_TILE):
            # Slot 2 * cta for a part that starts after work unit 0, 2 * cta + 1
            # for one that starts at 0 and stops early.
            partial_slot = 2 * cta + tl.where(work_start > 0, 0, 1)
            slot_beams = partial_slot * BLOCK_M + beam_offsets
            tl.store(
                partial_output
                + (slot_beams * HEAD_DIM)[:, None]
                + dim_offsets[None, :],
                accumulator / normalizers[:, None],
                mask=output_mask,
            )
            # Every work unit holds a token, so every part has softmax mass.
            tl.store(
                partial_lse + slot_beams,
                max_values * softmax_scale_log2 + tl.log2(normalizers),
                mask=beam_mask,
            )
        else:
            store_output(
                output=output,
                rows=rows,
                q_head=q_head,
                dim_offsets=dim_offsets,
                values=accumulator / normalizers[:, None],
                mask=output_mask,
                stride_out_row=stride_out_row,
                stride_out_head=stride_out_head,
            )

    num_real_rows = tl.load(row_count)
    for q_head in range(0, Q_HEADS):
        zero_padding_rows(
            output=output,
            num_real_rows=num_real_rows,
            num_rows=num_rows,
            program=cta,
            num_programs=num_ctas,
            q_head=q_head,
            stride_out_row=stride_out_row,
            stride_out_head=stride_out_head,
            HEAD_DIM=HEAD_DIM,
            BLOCK_M=BLOCK_M,
            BLOCK_D=BLOCK_D,
        )


@triton.jit
def _variable_width_merge_kernel(
    partial_output,  # [2 * num_ctas, BLOCK_M, HEAD_DIM] fp32, the parts' outputs
    partial_lse,  # [2 * num_ctas, BLOCK_M] fp32, log2 of the parts' softmax sums
    tiles,  # [tiles, TILE_FIELDS] int32, the layout's tiles
    plan,  # [num_ctas, PLAN_FIELDS] int32, the layout's plan
    output,  # [rows, q_heads, HEAD_DIM]; split (tile, head) pairs are written
    stride_out_row,  # output.stride(0); in elements
    stride_out_head,  # output.stride(1)
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,  # beams per tile
    BLOCK_D: tl.constexpr,  # HEAD_DIM rounded up to a power of 2
    MERGE_BEAMS: tl.constexpr,  # beams per program
    TILE_FIELDS: tl.constexpr,
    PLAN_FIELDS: tl.constexpr,  # int32 entries of a plan row
):
    """Program (c, chunk) writes MERGE_BEAMS beams of the (tile, head) that CTA c's
    range ends inside, if that range started it.

    Its parts are in slot 2 * c + 1, then in slot 2 * k of every later CTA k before
    the row's merge end. The output is their mean, weighted by each part's softmax
    sum.
    """
    cta = tl.program_id(0)
    plan_row = plan + PLAN_FIELDS * cta
    tile_idx = tl.load(plan_row + 3)
    q_head = tl.load(plan_row + 4)
    end_work = tl.load(plan_row + 5)
    # The range holds the first work unit of the (tile, head) it ends inside.
    starts_tile_qhead = (
        (tl.load(plan_row) != tile_idx)
        | (tl.load(plan_row + 1) != q_head)
        | (tl.load(plan_row + 2) == 0)
    )
    if (end_work > 0) & starts_tile_qhead:
        beam_offsets = tl.program_id(1) * MERGE_BEAMS + tl.arange(0, MERGE_BEAMS)
        rows, beam_mask, _, _ = read_tile(
            tiles=tiles,
            tile_idx=tile_idx,
            beam_offsets=beam_offsets,
            TILE_FIELDS=TILE_FIELDS,
        )
        # The last chunk reaches past the tile if MERGE_BEAMS does not divide BLOCK_M.
        beam_mask = beam_mask & (beam_offsets < BLOCK_M)
        if tl.sum(beam_mask.to(tl.int32)) > 0:
            dim_offsets = tl.arange(0, BLOCK_D)
            output_mask = beam_mask[:, None] & (dim_offsets < HEAD_DIM)[None, :]
            negative_infinity = -float("inf")
            max_lse = tl.full((MERGE_BEAMS,), negative_infinity, tl.float32)
            total_weight = tl.zeros((MERGE_BEAMS,), tl.float32)
            accumulator = tl.zeros((MERGE_BEAMS, BLOCK_D), tl.float32)
            merge_end = tl.load(plan_row + 6)
            for part_cta in range(cta, merge_end):
                # This CTA's part starts at work unit 0, the later CTAs' after it.
                partial_slot = 2 * part_cta + tl.where(part_cta == cta, 1, 0)
                slot_beams = partial_slot * BLOCK_M + beam_offsets
                part_lse = tl.load(
                    partial_lse + slot_beams,
                    mask=beam_mask,
                    other=negative_infinity,
                )
                # A CTA with an empty range wrote -inf and no output.
                part_output = tl.load(
                    partial_output
                    + (slot_beams * HEAD_DIM)[:, None]
                    + dim_offsets[None, :],
                    mask=output_mask & (part_lse > negative_infinity)[:, None],
                    other=0.0,
                )
                new_max = tl.maximum(max_lse, part_lse)
                safe_max = tl.where(new_max == negative_infinity, 0.0, new_max)
                previous_scale = tl.exp2(max_lse - safe_max)
                weight = tl.exp2(part_lse - safe_max)
                accumulator = (
                    accumulator * previous_scale[:, None]
                    + part_output * weight[:, None]
                )
                total_weight = total_weight * previous_scale + weight
                max_lse = new_max
            store_output(
                output=output,
                rows=rows,
                q_head=q_head,
                dim_offsets=dim_offsets,
                values=accumulator / total_weight[:, None],
                mask=output_mask,
                stride_out_row=stride_out_row,
                stride_out_head=stride_out_head,
            )


def shared_context_attention_variable_width_stream_k(
    q: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    tile_count: torch.Tensor,
    row_count: torch.Tensor,
    tiles: torch.Tensor,
    plan: torch.Tensor,
    *,
    block_m: int,
    num_ctas: int,
    softmax_scale: float,
) -> torch.Tensor:
    """shared_context_attention_variable_width on a persistent stream-K grid.

    Same inputs and output, and:
    - `plan`: the layout's plan of its tiles (`build_stream_k_plan`).
    - `num_ctas`: the grid size; it must be the one the plan was built for.
    """
    check_variable_width_inputs(
        q=q,
        k_buffer=k_buffer,
        v_buffer=v_buffer,
        req_to_token=req_to_token,
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        tile_count=tile_count,
        row_count=row_count,
        tiles=tiles,
    )
    if plan.dtype != torch.int32 or plan.numel() != plan_len(num_ctas):
        raise ValueError("The plan must be int32 with one row per CTA")
    output = torch.empty_like(q)
    num_rows, q_heads, head_dim = q.shape
    kv_heads = k_buffer.shape[1]

    # Two partial slots per CTA.
    partial_output = torch.empty(
        (2 * num_ctas, block_m, head_dim), dtype=torch.float32, device=q.device
    )
    partial_lse = torch.empty(
        (2 * num_ctas, block_m), dtype=torch.float32, device=q.device
    )
    block_d = triton.next_power_of_2(head_dim)
    _variable_width_stream_k_kernel[(num_ctas,)](
        q,
        k_buffer,
        v_buffer,
        req_to_token,
        req_pool_indices,
        seq_lens,
        row_count,
        tiles,
        plan,
        output,
        partial_output,
        partial_lse,
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
        TILE_FIELDS=TILE_FIELDS,
        PLAN_FIELDS=_PLAN_FIELDS,
        WHOLE_TILE=WHOLE_TILE,
        INT64_KV_OFFSETS=max(map(max_element_offset, (k_buffer, v_buffer)))
        >= _INT32_OFFSET_LIMIT,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )
    _variable_width_merge_kernel[(num_ctas, triton.cdiv(block_m, _MERGE_BEAMS))](
        partial_output,
        partial_lse,
        tiles,
        plan,
        output,
        output.stride(0),
        output.stride(1),
        HEAD_DIM=head_dim,
        BLOCK_M=block_m,
        BLOCK_D=block_d,
        MERGE_BEAMS=_MERGE_BEAMS,
        TILE_FIELDS=TILE_FIELDS,
        PLAN_FIELDS=_PLAN_FIELDS,
        # Triton's default; a merge program of 8 beams has no use for more.
        num_warps=4,
    )
    return output
