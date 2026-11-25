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
decode graphs take any beam width; every beam decode batch there runs it,
replayed or eager. It computes what the default kernel
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
  SETUP_UNITS for starting it, then its tile's work units. The units of a batch
  are numbered task by task.
- run: the consecutive units [run_start, run_end) that one CTA executes.
- segment: the work units of a task that lie in one run.
- plan: what the host appends to the layout (build_stream_k_plan), one row per
  CTA: the tasks the CTA attends, over which work units, and the split task it
  lists for the merge kernel.

The grid is persistent: one CTA per SM for any batch. The plan gives each CTA an
equal run of the batch's units, so a run may start and end inside a task. When
whole tasks load the busiest CTA no more than an equal run plus the cost of a
split, the runs end at task bounds instead.

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
of 256 beams, and as fast for 8 requests of 16, 64 and 256 beams. That batch
already gives each SM about one task, so nothing is left to balance.
"""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np
import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.beam_shared_context import num_sms
from sglang.kernels.ops.attention.beam_shared_context_variable_width import (
    LAYOUT_HEADER,
    TILE_FIELDS,
    WHOLE_TILE,
    attend_tile,
    check_variable_width_inputs,
    store_output,
    tile_rows,
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
# Measured: 8-row merge programs are fastest at 1-4 groups (8-64 tried).
_MERGE_ROWS = 8
# K/V offsets are computed in 32 bits (attend_tile's WIDE_KV_OFFSETS is off), which
# saves registers: 64-bit offsets take as many as a third pipeline stage would.
_MAX_OFFSET = 2**31

# int32 fields of a plan row, in the order the kernels read them:
#   0 first task, 1 end task: the CTA attends tasks [first task, end task)
#   2 the work unit its first task starts at, 3 the work unit its last task ends at
#   4 whether its first task's segment is partial, 5 the same for its last task
#   6 tile + 1 of the split task it lists for the merge kernel, 0 for none
#   7 that task's query head, 8 the last CTA with a segment of it,
#   9 the partial slot of this CTA's own segment of it
PLAN_FIELDS = 10
# Index of a row's first merge field.
_PLAN_MERGE_FIELD = 6


def default_num_ctas(device_index: int) -> int:
    """The grid of the stream-K kernel on a device: one CTA per SM. 8 warps of
    this kernel fill an SM's register file."""
    return num_sms(device_index)


def task_units(prompt_len: int, decode_len: int) -> int:
    """Units of one task of a tile, from its group's prompt length and the tokens
    each of its beams has generated."""
    return _SETUP_UNITS + triton.cdiv(prompt_len, _BLOCK_N) + decode_len


def plan_len(num_ctas: int) -> int:
    """int32 entries of the plan of a grid of num_ctas CTAs."""
    return PLAN_FIELDS * num_ctas


def _runs(
    task_starts: np.ndarray,
    longest_task_units: int,
    num_ctas: int,
    split_cost_units: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Every CTA's run, as (run_start, run_end), [num_ctas] each.

    Two ways to share a batch's units, compared by the units of the busiest CTA:
    - whole tasks: CTA c takes tasks [c * n, (c + 1) * n), n = ceil(tasks / CTAs);
      the last CTAs get fewer or none, and no task is split;
    - equal runs: every CTA takes the same number of units, and a task that is
      split costs split_cost_units more.
    Whole tasks win when splitting would save the busiest CTA at most
    split_cost_units units: the tasks are short, or their count is at or just
    under a multiple of the CTA count.
    """
    num_tasks = len(task_starts) - 1
    total_units = int(task_starts[-1])
    ctas = np.arange(num_ctas, dtype=np.int64)
    tasks_per_cta = max(triton.cdiv(num_tasks, num_ctas), 1)

    whole_tasks_cost = tasks_per_cta * longest_task_units
    equal_runs_cost = triton.cdiv(total_units, num_ctas) + split_cost_units
    if whole_tasks_cost <= equal_runs_cost:
        first_task = np.minimum(ctas * tasks_per_cta, num_tasks)
        end_task = np.minimum(first_task + tasks_per_cta, num_tasks)
        return task_starts[first_task], task_starts[end_task]
    return ctas * total_units // num_ctas, (ctas + 1) * total_units // num_ctas


def build_stream_k_plan(
    units_per_task: Sequence[int],
    *,
    q_heads: int,
    num_ctas: int,
    split_cost_units: Optional[int] = None,
) -> np.ndarray:
    """The plan of a batch, [num_ctas, PLAN_FIELDS] int32.

    units_per_task[t] is the `task_units` of tile t; each tile has q_heads tasks.
    split_cost_units defaults to the measured value; tests pass a very large one
    to force whole tasks and a negative one to force equal runs.
    """
    if split_cost_units is None:
        split_cost_units = _SPLIT_COST_UNITS
    units = np.asarray(units_per_task, dtype=np.int64)
    num_tasks = len(units) * q_heads

    # The first unit of every task, in task order, then the units of the batch.
    tile_starts = np.concatenate(([0], np.cumsum(units)))
    task_starts = tile_starts[:-1, None] * q_heads + np.arange(q_heads) * units[:, None]
    total_units = int(tile_starts[-1]) * q_heads
    task_starts = np.append(task_starts.reshape(-1), total_units)
    task_ends = task_starts[1:]
    # Work units are counted from a task's first one, after its setup units.
    first_work_units = task_starts[:-1] + _SETUP_UNITS

    run_start, run_end = _runs(
        task_starts=task_starts,
        longest_task_units=int(units.max()),
        num_ctas=num_ctas,
        split_cost_units=split_cost_units,
    )

    # The tasks of a run's first and last unit. An empty run gets some valid task
    # here and no task at the end.
    first_task = np.searchsorted(task_starts, run_start, side="right") - 1
    last_task = (
        np.searchsorted(task_starts, np.maximum(run_end - 1, 0), side="right") - 1
    )
    first_task = np.minimum(first_task, num_tasks - 1)
    last_task = np.minimum(last_task, num_tasks - 1)
    first_task_end = task_ends[first_task]
    last_task_end = task_ends[last_task]
    last_first_work_unit = first_work_units[last_task]

    # The segments of the first and the last task; the tasks between are whole.
    first_work_start = np.maximum(run_start - first_work_units[first_task], 0)
    last_work_start = np.where(last_task == first_task, first_work_start, 0)
    last_work_end = np.minimum(run_end, last_task_end) - last_first_work_unit

    # A run that ends in its last task's setup units has no work for that task: the
    # task before it is then the run's last one, attended to its end.
    last_has_work = last_work_end > last_work_start
    end_task = last_task + last_has_work
    work_units_per_task = task_ends - first_work_units
    last_work_end = np.where(
        last_has_work, last_work_end, work_units_per_task[end_task - 1]
    )
    has_tasks = (run_start < run_end) & (first_task < end_task)

    # A segment is partial when another run has work units of its task.
    first_is_partial = (first_work_start > 0) | (run_end < first_task_end)
    last_is_partial = np.where(
        end_task - 1 == first_task,
        first_is_partial,
        last_has_work & (run_end < last_task_end),
    )

    # A split task is listed by the run that holds its first work unit; only a
    # run's last task can be one. The merge reads its segments up to the CTA that
    # runs its last unit: CTA c runs units [c * total // n, (c + 1) * total // n).
    lists_task = (
        (run_start <= last_first_work_unit)
        & (last_first_work_unit < run_end)
        & (run_end < last_task_end)
    )
    merge_last_cta = (last_task_end * num_ctas - 1) // max(total_units, 1)
    starts_with_task = run_start >= last_first_work_unit - _SETUP_UNITS
    merge_first_slot = 2 * np.arange(num_ctas) + np.where(starts_with_task, 0, 1)

    # The kernels' field order (PLAN_FIELDS). A CTA without a task has zeros in
    # the first six fields, one that lists no task in the last four.
    segments = np.stack(
        [
            first_task,
            end_task,
            first_work_start,
            last_work_end,
            first_is_partial,
            last_is_partial,
        ]
    )
    merge = np.stack(
        [
            last_task // q_heads + 1,
            last_task % q_heads,
            merge_last_cta,
            merge_first_slot,
        ]
    )
    plan = np.concatenate((segments * has_tasks, merge * lists_task))
    return plan.T.astype(np.int32)


@triton.jit
def _variable_width_stream_k_kernel(
    q,  # [rows, q_heads, HEAD_DIM]; only the real rows are read
    k_buffer,  # [kv_slots, kv_heads, HEAD_DIM], the layer's K cache
    v_buffer,  # [kv_slots, kv_heads, HEAD_DIM], the layer's V cache
    req_to_token,  # [pool rows, max context] int32, the KV slot of each token
    req_pool_indices,  # [rows] int32 or int64, each row's pool row
    seq_lens,  # [rows] int32 or int64; only the leaders' are read
    layout,  # int32: header, tiles, plan
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
    LAYOUT_HEADER: tl.constexpr,
    TILE_FIELDS: tl.constexpr,
    PLAN_FIELDS: tl.constexpr,
    WHOLE_TILE: tl.constexpr,
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
    # The layout's parts: the header, the tiles, the plan.
    num_tiles = tl.load(layout)
    tiles = layout + LAYOUT_HEADER
    plan_row = tiles + TILE_FIELDS * num_tiles + PLAN_FIELDS * cta
    first_task = tl.load(plan_row)
    end_task = tl.load(plan_row + 1)
    first_work_start = tl.load(plan_row + 2)
    last_work_end = tl.load(plan_row + 3)
    first_is_partial = tl.load(plan_row + 4) != 0
    last_is_partial = tl.load(plan_row + 5) != 0

    if first_task == end_task:
        # No task for this CTA: the batch has fewer units, or tasks, than CTAs.
        # The merge of a task split around it still reads its slot 2 * cta.
        tl.store(
            partial_lse + 2 * cta * BLOCK_M + beam_offsets,
            tl.full((BLOCK_M,), -float("inf"), tl.float32),
        )

    for task in range(first_task, end_task):
        is_first = task == first_task
        is_last = task == end_task - 1
        q_head = task % Q_HEADS
        rows, beam_mask, max_values, normalizers, accumulator = attend_tile(
            q=q,
            k_buffer=k_buffer,
            v_buffer=v_buffer,
            req_to_token=req_to_token,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            tiles=tiles,
            tile=task // Q_HEADS,
            q_head=q_head,
            work_start=tl.where(is_first, first_work_start, 0),
            work_end=tl.where(is_last, last_work_end, WHOLE_TILE),
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
            PROMPT_LOOP_STEP=1,
            WIDE_KV_OFFSETS=False,
        )
        row_mask = beam_mask[:, None] & dim_mask[None, :]
        is_partial = (is_first & first_is_partial) | (is_last & last_is_partial)
        if is_partial:
            # Slot 2 * cta for the run's first task, 2 * cta + 1 for its last.
            slot = 2 * cta + tl.where(is_first, 0, 1)
            slot_beams = slot * BLOCK_M + beam_offsets
            tl.store(
                partial_output
                + (slot_beams * HEAD_DIM)[:, None]
                + dim_offsets[None, :],
                accumulator / normalizers[:, None],
                mask=row_mask,
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
                mask=row_mask,
                stride_out_row=stride_out_row,
                stride_out_head=stride_out_head,
            )

    num_real_rows = tl.load(layout + 1)
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
    layout,  # int32: header, tiles, plan
    output,  # [rows, q_heads, HEAD_DIM]; the rows of split tasks are written
    stride_out_row,  # output.stride(0); in elements
    stride_out_head,  # output.stride(1)
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,  # beams per tile
    BLOCK_D: tl.constexpr,  # HEAD_DIM rounded up to a power of 2
    MERGE_ROWS: tl.constexpr,  # beams per program
    LAYOUT_HEADER: tl.constexpr,
    TILE_FIELDS: tl.constexpr,
    PLAN_FIELDS: tl.constexpr,
    PLAN_MERGE_FIELD: tl.constexpr,
):
    """Program (c, chunk) writes MERGE_ROWS beams of the split task that CTA c's
    plan row lists.

    The task's segments are in the slot that the row names, then in slot 2 * c' of
    every later CTA c' up to the row's last CTA. The output is their mean,
    weighted by each segment's softmax sum.
    """
    cta = tl.program_id(0)
    num_tiles = tl.load(layout)
    tiles = layout + LAYOUT_HEADER
    merge_fields = (
        tiles + TILE_FIELDS * num_tiles + PLAN_FIELDS * cta + PLAN_MERGE_FIELD
    )
    tile = tl.load(merge_fields) - 1
    if tile >= 0:
        q_head = tl.load(merge_fields + 1)
        last_cta = tl.load(merge_fields + 2)
        first_segment_slot = tl.load(merge_fields + 3)
        beam_offsets = tl.program_id(1) * MERGE_ROWS + tl.arange(0, MERGE_ROWS)
        rows, beam_mask, _, _ = tile_rows(
            tiles=tiles, tile=tile, beam_offsets=beam_offsets, TILE_FIELDS=TILE_FIELDS
        )
        # The last chunk reaches past the tile if MERGE_ROWS does not divide BLOCK_M.
        beam_mask = beam_mask & (beam_offsets < BLOCK_M)
        if tl.sum(beam_mask.to(tl.int32)) > 0:
            dim_offsets = tl.arange(0, BLOCK_D)
            row_mask = beam_mask[:, None] & (dim_offsets < HEAD_DIM)[None, :]
            negative_infinity = -float("inf")
            max_lse = tl.full((MERGE_ROWS,), negative_infinity, tl.float32)
            total_weight = tl.zeros((MERGE_ROWS,), tl.float32)
            accumulator = tl.zeros((MERGE_ROWS, BLOCK_D), tl.float32)
            for segment_cta in range(cta, last_cta + 1):
                slot = tl.where(segment_cta == cta, first_segment_slot, 2 * segment_cta)
                slot_beams = slot * BLOCK_M + beam_offsets
                segment_lse = tl.load(
                    partial_lse + slot_beams,
                    mask=beam_mask,
                    other=negative_infinity,
                )
                # A CTA without units wrote -inf and no output.
                segment_output = tl.load(
                    partial_output
                    + (slot_beams * HEAD_DIM)[:, None]
                    + dim_offsets[None, :],
                    mask=row_mask & (segment_lse > negative_infinity)[:, None],
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
                mask=row_mask,
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
    layout: torch.Tensor,
    *,
    block_m: int,
    num_ctas: int,
    output: torch.Tensor | None = None,
    softmax_scale: float,
) -> torch.Tensor:
    """shared_context_attention_variable_width on a persistent stream-K grid.

    Same inputs and output, and:
    - `layout` must carry the plan of its tiles (`build_stream_k_plan`) after them.
    - `num_ctas`: the grid size; it must be the one the plan was built for.
    Rejects K/V buffers of 2^31 elements or more. Tuned on SM120 only.
    """
    output = check_variable_width_inputs(
        q=q,
        k_buffer=k_buffer,
        v_buffer=v_buffer,
        req_to_token=req_to_token,
        req_pool_indices=req_pool_indices,
        seq_lens=seq_lens,
        layout=layout,
        output=output,
    )
    if max(k_buffer.numel(), v_buffer.numel()) >= _MAX_OFFSET:
        raise ValueError(
            f"K/V buffers of {k_buffer.numel()} elements reach 2^31; this kernel "
            "indexes them with 32-bit offsets"
        )
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
        layout,
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
        LAYOUT_HEADER=LAYOUT_HEADER,
        TILE_FIELDS=TILE_FIELDS,
        PLAN_FIELDS=PLAN_FIELDS,
        WHOLE_TILE=WHOLE_TILE,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )
    _variable_width_merge_kernel[(num_ctas, triton.cdiv(block_m, _MERGE_ROWS))](
        partial_output,
        partial_lse,
        layout,
        output,
        output.stride(0),
        output.stride(1),
        HEAD_DIM=head_dim,
        BLOCK_M=block_m,
        BLOCK_D=block_d,
        MERGE_ROWS=_MERGE_ROWS,
        LAYOUT_HEADER=LAYOUT_HEADER,
        TILE_FIELDS=TILE_FIELDS,
        PLAN_FIELDS=PLAN_FIELDS,
        PLAN_MERGE_FIELD=_PLAN_MERGE_FIELD,
        # Triton's default; an 8-row merge program has no use for more.
        num_warps=4,
    )
    return output
