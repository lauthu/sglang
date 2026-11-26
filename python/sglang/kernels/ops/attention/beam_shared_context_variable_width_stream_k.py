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
(beam_shared_context_variable_width.py) computes, from the same layout. That kernel
gives a program whole tiles, so one request with a long prompt keeps a few SMs
busy and leaves the others idle; this one shares the work of a batch evenly over
the SMs.

Terms (group, tile, pool row, real rows, work unit and the layout are defined in
that module):
- task: one tile attended for one query head. Tasks are numbered tile by tile,
  then head by head: task = tile * Q_HEADS + q_head. (The fixed-width
  beam_shared_context_stream_k.py calls this a tile; here a tile has no head.)
- unit: the measure of work that the plan shares over the CTAs. A task is
  SETUP_UNITS for starting it, then one unit per work unit of its tile. The units
  of a batch are numbered task by task.
- run: the consecutive units that one CTA executes.
- segment: the work units of a task that lie in one run.
- plan: what the host puts in the layout after its header (build_stream_k_plan),
  one row per CTA: the tasks the CTA attends, over which work units, and the split task it
  lists for the merge kernel.

The grid is persistent: one CTA per SM for any batch. The plan gives each CTA an
equal run of the batch's units, so a run may start and end inside a task. Where
whole tasks load no CTA more than that, the runs end at task bounds instead
(_run_bounds).

A task that lies in one run is written to the output directly. A task split over
several runs leaves one partial result per segment, which the merge kernel
combines:
- partial slots: two per CTA, so the workspace has one size for any batch. Slot
  2 * c holds the segment of the first task of CTA c's run, slot 2 * c + 1 the
  segment of its last task; only those two can be partial.
- A split task is listed once, in the row of the CTA whose run holds its first
  work unit, with the last CTA that has a segment of it.

The host builds the plan once per decode step and every layer's launch reads it.
A zeroed plan gives every CTA no task and nothing to merge, so a graph captured
on a zeroed layout attends nothing.

Against the default kernel, per launch on SM120 with 2,048-token prompts: 3.0x to
4.1x faster for one request of 16 to 256 beams, 1.2x to 1.3x for 4 or 8 requests
of 256 beams, and as fast for a batch of 8 requests that mixes 16, 64 and 256
beams. That batch already gives each SM about one task, so nothing is left to
balance.
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

# The fixed-width stream-K kernel's values (beam_shared_context_stream_k.py). This
# kernel was timed with them on RTX PRO 6000 (SM120), Qwen3-0.6B heads, 1-8 groups
# of 1-2,000 beams.
_BLOCK_N = 64
_NUM_WARPS = 8
_NUM_STAGES = 2
# What starting a task costs (a Q load, an output store), in units. Not swept
# here: beam_shared_context_stream_k.py measured 1 as the best of 0-3.
_SETUP_UNITS = 1
# Units that splitting tasks adds to a CTA's equal share (a second Q load, partial
# stores, the merge). Measured: of 0-16, 2 picks the faster of equal runs and
# whole tasks at 39 batch shapes.
_SPLIT_COST_UNITS = 2
# Measured: merge programs of 8 beams are fastest at 1-4 groups (8-64 tried).
_MERGE_BEAMS = 8
# K/V offsets are int32 while every element of the caches is below this offset
# (attend_tile's INT64_KV_OFFSETS). It saves registers: int64 offsets take as many
# as a third pipeline stage would.
_INT32_OFFSET_LIMIT = 2**31

# int32 fields of a plan row, in the order the kernels read them:
#   0 first task, 1 end task: the CTA attends tasks [first task, end task)
#   2 the work unit its first task starts at, 3 the work unit its last task ends at
#   4 whether its first task's segment is partial, 5 the same for its last task
#   6 tile + 1 of the split task it lists for the merge kernel, 0 for none
#   7 that task's query head, 8 the last CTA with a segment of it,
#   9 the partial slot of this CTA's own segment of it
_PLAN_FIELDS = 10
# Index of a row's first merge field.
_PLAN_MERGE_FIELD = 6


def default_num_ctas(device_index: int) -> int:
    """The grid of the stream-K kernel on a device: one CTA per SM. On SM120, 8
    warps of this kernel fill an SM's register file."""
    return num_sms(device_index)


def task_units(prompt_len: int, decode_len: int) -> int:
    """Units of one task of a tile, from its group's prompt length and the tokens
    each of its beams has generated."""
    return _SETUP_UNITS + triton.cdiv(prompt_len, _BLOCK_N) + decode_len


def plan_len(num_ctas: int) -> int:
    """int32 entries of the plan of a grid of num_ctas CTAs."""
    return _PLAN_FIELDS * num_ctas


def _run_bounds(
    task_bounds: np.ndarray, longest_task_units: int, num_ctas: int
) -> np.ndarray:
    """Where the CTAs' runs start and end, [num_ctas + 1]: CTA c executes units
    [bounds[c], bounds[c + 1]).

    Two ways to share a batch's units:
    - whole tasks: CTA c takes tasks [c * n, (c + 1) * n), n = ceil(tasks / CTAs);
      the last CTAs get fewer or none, and no task is split;
    - equal runs: every CTA takes the same number of units, and a task that is
      split costs _SPLIT_COST_UNITS more.
    Whole tasks win unless splitting saves the busiest CTA more than that cost,
    taking n of the longest task as its load. So they win when the tasks are
    short, or when their count is at or just under a multiple of the CTA count.
    """
    num_tasks = len(task_bounds) - 1
    total_units = int(task_bounds[-1])
    cuts = np.arange(num_ctas + 1, dtype=np.int64)
    tasks_per_cta = triton.cdiv(num_tasks, num_ctas)

    whole_tasks_cost = tasks_per_cta * longest_task_units
    equal_runs_cost = triton.cdiv(total_units, num_ctas) + _SPLIT_COST_UNITS
    if whole_tasks_cost <= equal_runs_cost:
        return task_bounds[np.minimum(cuts * tasks_per_cta, num_tasks)]
    return cuts * total_units // num_ctas


def build_stream_k_plan(
    task_units_per_tile: Sequence[int], *, q_heads: int, num_ctas: int
) -> np.ndarray:
    """The plan of a batch, [num_ctas, _PLAN_FIELDS] int32.

    task_units_per_tile[t] is the `task_units` of each of tile t's q_heads tasks.
    """
    units = np.asarray(task_units_per_tile, dtype=np.int64)
    num_tasks = len(units) * q_heads

    # The unit each task starts at, in task order, then the batch's unit count.
    tile_bounds = np.concatenate(([0], np.cumsum(units))) * q_heads
    task_starts = tile_bounds[:-1, None] + np.arange(q_heads) * units[:, None]
    task_bounds = np.append(task_starts.reshape(-1), tile_bounds[-1])
    # The same, counting work units only: every task starts with its setup units.
    # From here on a work unit is numbered in the batch, task by task.
    work_bounds = task_bounds - _SETUP_UNITS * np.arange(num_tasks + 1)

    run_bounds = _run_bounds(
        task_bounds, longest_task_units=int(units.max()), num_ctas=num_ctas
    )
    # The work units before each run bound: CTA c attends work units
    # [work_start[c], work_end[c]). A bound inside a task's setup units is at the
    # task's first work unit.
    bound_task = np.searchsorted(task_bounds, run_bounds, side="right") - 1
    work_cuts = work_bounds[bound_task] + np.maximum(
        run_bounds - task_bounds[bound_task] - _SETUP_UNITS, 0
    )
    work_start, work_end = work_cuts[:-1], work_cuts[1:]
    has_work = work_start < work_end

    # The tasks of a run's first and last work unit. A run without work gets some
    # valid task here; its row is zeroed below.
    first_task = np.searchsorted(work_bounds, work_start, side="right") - 1
    first_task = np.minimum(first_task, num_tasks - 1)
    last_task = np.searchsorted(work_bounds, work_end, side="left") - 1
    last_task = np.maximum(last_task, 0)
    first_task_end = work_bounds[first_task + 1]
    last_task_start = work_bounds[last_task]
    last_task_end = work_bounds[last_task + 1]

    # A run attends its first task from first_work_start on and its last task up
    # to last_work_end, both counted from the task's first work unit; the tasks
    # between are whole. A task is partial when the run has only a part of it.
    first_work_start = work_start - work_bounds[first_task]
    last_work_end = work_end - last_task_start
    first_is_partial = (first_work_start > 0) | (work_end < first_task_end)
    last_is_partial = np.where(
        first_task == last_task, first_is_partial, work_end < last_task_end
    )

    # A split task is listed by the CTA whose run holds its first work unit; it is
    # that run's last task. The merge reads its segments up to the first CTA whose
    # run reaches the task's end. The listing CTA's own segment is in the partial
    # slot of its first task (2 * c) or of its last task (2 * c + 1).
    lists_task = has_work & (work_start <= last_task_start) & (work_end < last_task_end)
    merge_last_cta = np.searchsorted(work_end, last_task_end, side="left")
    merge_first_slot = 2 * np.arange(num_ctas) + (first_task != last_task)

    # The kernels' field order (_PLAN_FIELDS). A CTA without a task has zeros in
    # the first six fields, one that lists no task in the last four.
    segment_fields = np.stack(
        [
            first_task,
            last_task + 1,
            first_work_start,
            last_work_end,
            first_is_partial,
            last_is_partial,
        ]
    )
    merge_fields = np.stack(
        [
            last_task // q_heads + 1,
            last_task % q_heads,
            merge_last_cta,
            merge_first_slot,
        ]
    )
    plan = np.concatenate((segment_fields * has_work, merge_fields * lists_task))
    return plan.T.astype(np.int32)


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
    output,  # like q; whole tasks and the padding rows are written here
    partial_output,  # [2 * num_ctas, BLOCK_M, HEAD_DIM] fp32, the segments' outputs
    partial_lse,  # [2 * num_ctas, BLOCK_M] fp32, log2 of the segments' softmax sums
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
    """CTA c attends the tasks of its plan row. A task that lies in its run goes to
    the output; a segment of a split task goes to a partial slot, for the merge
    kernel."""
    cta = tl.program_id(0)
    beam_offsets = tl.arange(0, BLOCK_M)
    dim_offsets = tl.arange(0, BLOCK_D)
    dim_mask = dim_offsets < HEAD_DIM
    # exp2 is cheaper than exp: fold log2(e) into the scale.
    softmax_scale_log2 = softmax_scale * 1.4426950408889634
    plan_row = plan + PLAN_FIELDS * cta
    first_task = tl.load(plan_row)
    end_task = tl.load(plan_row + 1)
    first_work_start = tl.load(plan_row + 2)
    last_work_end = tl.load(plan_row + 3)
    first_is_partial = tl.load(plan_row + 4) != 0
    last_is_partial = tl.load(plan_row + 5) != 0

    if first_task == end_task:
        # The plan gives this CTA no task. The merge of a task split around it
        # still reads its slot 2 * cta.
        tl.store(
            partial_lse + 2 * cta * BLOCK_M + beam_offsets,
            tl.full((BLOCK_M,), -float("inf"), tl.float32),
        )

    for task_idx in range(first_task, end_task):
        is_first_task = task_idx == first_task
        is_last_task = task_idx == end_task - 1
        q_head = task_idx % Q_HEADS
        rows, beam_mask, max_values, normalizers, accumulator = attend_tile(
            q=q,
            k_buffer=k_buffer,
            v_buffer=v_buffer,
            req_to_token=req_to_token,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            tiles=tiles,
            tile_idx=task_idx // Q_HEADS,
            q_head=q_head,
            work_start=tl.where(is_first_task, first_work_start, 0),
            work_end=tl.where(is_last_task, last_work_end, WHOLE_TILE),
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
        is_partial = (is_first_task & first_is_partial) | (
            is_last_task & last_is_partial
        )
        if is_partial:
            # Slot 2 * cta for the run's first task, 2 * cta + 1 for its last.
            partial_slot = 2 * cta + tl.where(is_first_task, 0, 1)
            slot_beams = partial_slot * BLOCK_M + beam_offsets
            tl.store(
                partial_output
                + (slot_beams * HEAD_DIM)[:, None]
                + dim_offsets[None, :],
                accumulator / normalizers[:, None],
                mask=output_mask,
            )
            # Every work unit holds a token, so every segment has softmax mass.
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
    partial_output,  # [2 * num_ctas, BLOCK_M, HEAD_DIM] fp32, the segments' outputs
    partial_lse,  # [2 * num_ctas, BLOCK_M] fp32, log2 of the segments' softmax sums
    tiles,  # [tiles, TILE_FIELDS] int32, the layout's tiles
    plan,  # [num_ctas, PLAN_FIELDS] int32, the layout's plan
    output,  # [rows, q_heads, HEAD_DIM]; the rows of split tasks are written
    stride_out_row,  # output.stride(0); in elements
    stride_out_head,  # output.stride(1)
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,  # beams per tile
    BLOCK_D: tl.constexpr,  # HEAD_DIM rounded up to a power of 2
    MERGE_BEAMS: tl.constexpr,  # beams per program
    TILE_FIELDS: tl.constexpr,
    PLAN_FIELDS: tl.constexpr,  # int32 entries of a plan row
    PLAN_MERGE_FIELD: tl.constexpr,
):
    """Program (c, chunk) writes MERGE_BEAMS beams of the split task that CTA c's
    plan row lists.

    The task's segments are in the slot that the row names, then in slot 2 * c' of
    every later CTA c' up to the row's last CTA. The output is their mean,
    weighted by each segment's softmax sum.
    """
    cta = tl.program_id(0)
    merge_fields = plan + PLAN_FIELDS * cta + PLAN_MERGE_FIELD
    tile_idx = tl.load(merge_fields) - 1
    if tile_idx >= 0:
        q_head = tl.load(merge_fields + 1)
        last_cta = tl.load(merge_fields + 2)
        first_segment_slot = tl.load(merge_fields + 3)
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
            for segment_cta in range(cta, last_cta + 1):
                partial_slot = tl.where(
                    segment_cta == cta, first_segment_slot, 2 * segment_cta
                )
                slot_beams = partial_slot * BLOCK_M + beam_offsets
                segment_lse = tl.load(
                    partial_lse + slot_beams,
                    mask=beam_mask,
                    other=negative_infinity,
                )
                # A CTA without a task wrote -inf and no output.
                segment_output = tl.load(
                    partial_output
                    + (slot_beams * HEAD_DIM)[:, None]
                    + dim_offsets[None, :],
                    mask=output_mask & (segment_lse > negative_infinity)[:, None],
                    other=0.0,
                )
                new_max = tl.maximum(max_lse, segment_lse)
                safe_max = tl.where(new_max == negative_infinity, 0.0, new_max)
                previous_scale = tl.exp2(max_lse - safe_max)
                weight = tl.exp2(segment_lse - safe_max)
                accumulator = (
                    accumulator * previous_scale[:, None]
                    + segment_output * weight[:, None]
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
    Tuned on SM120 only.
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
        PLAN_MERGE_FIELD=_PLAN_MERGE_FIELD,
        # Triton's default; a merge program of 8 beams has no use for more.
        num_warps=4,
    )
    return output
