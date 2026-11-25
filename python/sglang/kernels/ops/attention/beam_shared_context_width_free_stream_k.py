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
(beam_shared_context_width_free.py) computes, from the same layout. That kernel
gives a program whole tiles, so one request with a long prompt keeps a few SMs
busy and leaves the others idle; this one shares the work of a batch evenly over
the SMs.

Terms (group, tile, pool row, real rows and the layout are defined in that module):
- task: one tile attended for one query head. Tasks are numbered tile by tile,
  then head by head: task = tile * Q_HEADS + q_head. (The fixed-width
  beam_shared_context_stream_k.py calls this a tile; here a tile has no head.)
- unit: the measure of work that the schedule counts and the CTAs share. A task
  is SETUP_UNITS for starting it, then one work unit per BLOCK_N prompt tokens,
  then one per decode step. The units of a batch are numbered task by task.
- run: the consecutive units [run_start, run_end) that one CTA executes.
- segment: the part of a task that lies in one run.
- schedule: what the host appends to the layout (build_schedule), counted for
  one query head: [units of the longest task, units of all tiles, unit_starts[0],
  ..., unit_starts[tiles]], where unit_starts[t] is the units of the tiles
  before tile t.

The grid is persistent: one CTA per SM for any batch. Each CTA takes an equal run
of the batch's units, so a run may start and end inside a task. When whole tasks
load the busiest CTA no more than an equal run plus the cost of a split, the runs
end at task bounds instead (_cta_run).

A task that lies in one run is written to the output directly. A task split over
several runs leaves one partial result per segment, which the merge kernel
combines:
- partial slots: two per CTA, so the workspace has one size for any batch. Slot
  2 * c holds the segment that CTA c's run starts with, slot 2 * c + 1 a segment
  that starts later in the run; only a run's first and last segments can be
  partial.
- merge list: one entry per CTA, (tile or -1, query head, last CTA, first
  segment's slot). The CTA whose run holds the first work unit of a split task
  lists it: the task, the CTA that runs the task's last unit, and the partial
  slot (2 * c or 2 * c + 1) of the listing CTA's own segment.

Against the default kernel, per launch on SM120 with 2,048-token prompts: 2.7x to
3.8x faster for one request of 16 to 256 beams, 1.2x for 4 or 8 requests of 256
beams, and 6% slower for 8 requests of 16, 64 and 256 beams. That batch already
gives each SM about one task, so nothing is left to balance, and a unit costs
more here (2 pipeline stages against 3).
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
from sglang.kernels.ops.attention.beam_shared_context_width_free import (
    LAYOUT_HEADER,
    TILE_FIELDS,
    check_width_free_inputs,
    load_kv,
    load_q,
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
# Tiles read per step of the tile search; one step covers most decode batches.
_TILE_SEARCH = 128
# Schedule fields before unit_starts.
_SCHEDULE_HEADER = 2
# Merge list fields per CTA: tile (-1: none), query head, last CTA, first
# segment's slot.
_MERGE_FIELDS = 4
# K/V offsets are computed in 32 bits, which saves registers: 64-bit offsets take
# as many as a third pipeline stage would.
_MAX_OFFSET = 2**31


def task_units(prompt_len: int, decode_len: int) -> int:
    """Units of one task of a tile, from its group's prompt length and the tokens
    each of its beams has generated."""
    return _SETUP_UNITS + triton.cdiv(prompt_len, _BLOCK_N) + decode_len


def build_schedule(units_per_task: Sequence[int]) -> list[int]:
    """The schedule of a batch; units_per_task[t] is the `task_units` of tile t."""
    unit_starts = [0, *itertools.accumulate(units_per_task)]
    return [max(units_per_task), unit_starts[-1], *unit_starts]


def schedule_len(num_tiles: int) -> int:
    """int32 entries of the schedule of a batch of num_tiles tiles."""
    return _SCHEDULE_HEADER + num_tiles + 1


@triton.jit
def _task_first_unit(
    unit_starts,  # [tiles + 1] int32, from the schedule
    task,  # in [0, tasks]
    Q_HEADS: tl.constexpr,
):
    """The first unit of a task. For the number past the last task, the units of
    the batch, which is where that task would start."""
    tile = task // Q_HEADS
    q_head = task % Q_HEADS
    tile_start = tl.load(unit_starts + tile)
    # Query head 0 does not need the next entry, which the number past the last
    # task does not have.
    units_per_task = (
        tl.load(unit_starts + tile + 1, mask=q_head > 0, other=tile_start) - tile_start
    )
    return tile_start * Q_HEADS + q_head * units_per_task


@triton.jit
def _find_tile(
    unit_starts,  # [tiles + 1] int32, increasing
    num_tiles,
    head_unit,  # a unit number // Q_HEADS: a position on unit_starts' scale
    TILE_SEARCH: tl.constexpr,
):
    """The last tile t with unit_starts[t] <= head_unit; -1 without tiles.
    Counts the tiles that start by then, TILE_SEARCH of them per step."""
    found = -1
    for base in range(0, num_tiles, TILE_SEARCH):
        candidates = base + tl.arange(0, TILE_SEARCH)
        valid = candidates < num_tiles
        starts = tl.load(unit_starts + candidates, mask=valid, other=0)
        found += tl.sum(((starts <= head_unit) & valid).to(tl.int32))
    return found


@triton.jit
def _task_of_unit(
    unit_starts,  # [tiles + 1] int32, from the schedule
    num_tiles,
    unit,
    Q_HEADS: tl.constexpr,
    TILE_SEARCH: tl.constexpr,
):
    """The task that holds a unit."""
    # Tile t holds units [unit_starts[t] * Q_HEADS, unit_starts[t + 1] * Q_HEADS).
    tile = tl.maximum(
        _find_tile(
            unit_starts=unit_starts,
            num_tiles=num_tiles,
            head_unit=unit // Q_HEADS,
            TILE_SEARCH=TILE_SEARCH,
        ),
        0,
    )
    tile_start = tl.load(unit_starts + tile)
    # At least 1: a zeroed layout, which a graph is captured with, has no units.
    units_per_task = tl.maximum(tl.load(unit_starts + tile + 1) - tile_start, 1)
    return tile * Q_HEADS + (unit - tile_start * Q_HEADS) // units_per_task


@triton.jit
def _unit_owner(unit, total_units, num_ctas):
    """The CTA whose equal run holds a unit: CTA c runs units
    [c * total_units // num_ctas, (c + 1) * total_units // num_ctas)."""
    # unit < (c + 1) * total // n  <=>  c >= ((unit + 1) * n - 1) // total.
    # The divisor is at least 1 for a zeroed layout, which has no units.
    owner = ((unit + 1).to(tl.int64) * num_ctas - 1) // tl.maximum(total_units, 1)
    return owner.to(tl.int32)


@triton.jit
def _cta_run(
    schedule,  # the layout's schedule
    unit_starts,  # [tiles + 1] int32, from the schedule
    num_tiles,
    cta,
    num_ctas,
    Q_HEADS: tl.constexpr,
    SPLIT_COST_UNITS: tl.constexpr,
):
    """This CTA's run, [run_start, run_end), and the units of the batch.

    Two ways to share a batch's units, compared by the units of the busiest CTA:
    - whole tasks: CTA c takes tasks [c * n, (c + 1) * n), n = ceil(tasks / CTAs);
      the last CTAs get fewer or none, and no task is split;
    - equal runs: every CTA takes the same number of units, and a task that is
      split costs SPLIT_COST_UNITS more.
    Whole tasks win when splitting would save the busiest CTA at most
    SPLIT_COST_UNITS units: the tasks are short, or their count is at or just
    under a multiple of the CTA count.
    """
    longest_task_units = tl.load(schedule)
    total_units = tl.load(schedule + 1) * Q_HEADS
    num_tasks = num_tiles * Q_HEADS
    tasks_per_cta = tl.maximum(tl.cdiv(num_tasks, num_ctas), 1)

    whole_tasks_cost = tasks_per_cta * longest_task_units
    equal_runs_cost = tl.cdiv(total_units, num_ctas) + SPLIT_COST_UNITS
    if whole_tasks_cost <= equal_runs_cost:
        first_task = tl.minimum(cta * tasks_per_cta, num_tasks)
        end_task = tl.minimum(first_task + tasks_per_cta, num_tasks)
        run_start = _task_first_unit(
            unit_starts=unit_starts, task=first_task, Q_HEADS=Q_HEADS
        )
        run_end = _task_first_unit(
            unit_starts=unit_starts, task=end_task, Q_HEADS=Q_HEADS
        )
    else:
        run_start = ((cta.to(tl.int64) * total_units) // num_ctas).to(tl.int32)
        run_end = (((cta + 1).to(tl.int64) * total_units) // num_ctas).to(tl.int32)
    return run_start, run_end, total_units


@triton.jit
def _segment_slot(cta, run_start, task_start):
    """The partial slot for a run's segment of a task: 2 * cta if the run starts
    with it, 2 * cta + 1 if the task starts later in the run."""
    return 2 * cta + (run_start < task_start).to(tl.int32)


@triton.jit
def _write_merge_entry(
    merge_list,  # [num_ctas, MERGE_FIELDS] int32
    unit_starts,  # [tiles + 1] int32, from the schedule
    cta,
    num_ctas,
    run_start,
    run_end,
    total_units,
    last_task,  # the task of the run's last unit
    Q_HEADS: tl.constexpr,
    SETUP_UNITS: tl.constexpr,
    MERGE_FIELDS: tl.constexpr,
):
    """Write this CTA's merge list entry: the split task to merge, or tile -1.

    A split task is listed once, by the CTA whose run holds its first work unit.
    Only the run's last task can be one: an earlier task of the run ends in it.
    """
    task_start = _task_first_unit(
        unit_starts=unit_starts, task=last_task, Q_HEADS=Q_HEADS
    )
    task_end = _task_first_unit(
        unit_starts=unit_starts, task=last_task + 1, Q_HEADS=Q_HEADS
    )
    first_work_unit = task_start + SETUP_UNITS
    holds_first_work_unit = (run_start <= first_work_unit) & (first_work_unit < run_end)
    finished_later = run_end < task_end
    listed = holds_first_work_unit & finished_later

    entry = merge_list + MERGE_FIELDS * cta
    tl.store(entry, tl.where(listed, last_task // Q_HEADS, -1))
    tl.store(entry + 1, last_task % Q_HEADS)
    # The merge reads the segments up to the CTA that runs the task's last unit.
    last_cta = _unit_owner(
        unit=task_end - 1, total_units=total_units, num_ctas=num_ctas
    )
    tl.store(entry + 2, last_cta)
    first_segment_slot = _segment_slot(
        cta=cta, run_start=run_start, task_start=task_start
    )
    tl.store(entry + 3, first_segment_slot)


@triton.jit
def _work_range(
    run_start,
    run_end,
    task_start,
    task_end,
    SETUP_UNITS: tl.constexpr,
):
    """The work units of a task that lie in a run, [work_start, work_end), counted
    from the task's first work unit. Empty if the run holds only setup units."""
    first_work_unit = task_start + SETUP_UNITS
    work_start = tl.maximum(run_start, first_work_unit) - first_work_unit
    work_end = tl.minimum(run_end, task_end) - first_work_unit
    return work_start, work_end


@triton.jit
def _attend_task(
    q,  # [rows, q_heads, HEAD_DIM]
    k_buffer,  # [kv_slots, kv_heads, HEAD_DIM], the layer's K cache
    v_buffer,  # [kv_slots, kv_heads, HEAD_DIM], the layer's V cache
    req_to_token,  # [pool rows, max context] int32, the KV slot of each token
    req_pool_indices,  # [rows] int32 or int64, each row's pool row
    seq_lens,  # [rows] int32 or int64; only the leaders' are read
    tiles,  # the layout's tiles
    tile,
    q_head,
    work_start,  # work units of the task to attend: [work_start, work_end)
    work_end,
    softmax_scale_log2,  # softmax_scale * log2(e), for exp2
    stride_q_row,  # q.stride(0); all strides are in elements
    stride_q_head,  # q.stride(1)
    stride_k_slot,  # k_buffer.stride(0)
    stride_k_head,  # k_buffer.stride(1)
    stride_v_slot,  # v_buffer.stride(0)
    stride_v_head,  # v_buffer.stride(1)
    stride_req_to_token,  # req_to_token.stride(0)
    Q_PER_KV: tl.constexpr,  # query heads per KV head
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,  # beams per tile
    BLOCK_N: tl.constexpr,  # prompt tokens per block
    BLOCK_D: tl.constexpr,  # HEAD_DIM rounded up to a power of 2
    TILE_FIELDS: tl.constexpr,
):
    """Softmax state of a task after its work units [work_start, work_end).

    The two loops of the default kernel (_width_free_shared_context_kernel), each
    limited to the range: a task's first work units are its prompt blocks, the
    following ones its decode steps. KV slots stay 32-bit here (_MAX_OFFSET).
    Returns (rows, beam_mask, max_values, normalizers, accumulator).
    """
    kv_head = q_head // Q_PER_KV
    beam_offsets = tl.arange(0, BLOCK_M)
    context_offsets = tl.arange(0, BLOCK_N)
    dim_offsets = tl.arange(0, BLOCK_D)
    dim_mask = dim_offsets < HEAD_DIM
    negative_infinity = -float("inf")

    rows, beam_mask, leader_row, prompt_len = tile_rows(
        tiles=tiles, tile=tile, beam_offsets=beam_offsets, TILE_FIELDS=TILE_FIELDS
    )
    row_mask = beam_mask[:, None] & dim_mask[None, :]
    q_values = load_q(
        q=q,
        rows=rows,
        q_head=q_head,
        dim_offsets=dim_offsets,
        mask=row_mask,
        stride_q_row=stride_q_row,
        stride_q_head=stride_q_head,
    )
    max_values, normalizers, accumulator = empty_softmax_state(
        beam_mask=beam_mask, BLOCK_M=BLOCK_M, BLOCK_D=BLOCK_D
    )

    # The prompt: all beams of the tile attend the same tokens, BLOCK_N at a time.
    # Their KV slots are in the leader's pool row.
    num_prompt_blocks = tl.cdiv(prompt_len, BLOCK_N)
    leader_pool_row = tl.load(req_pool_indices + leader_row).to(tl.int64)
    prompt_slots = req_to_token + leader_pool_row * stride_req_to_token
    for block in tl.range(work_start, tl.minimum(work_end, num_prompt_blocks)):
        tokens = block * BLOCK_N + context_offsets
        token_mask = tokens < prompt_len
        slots = tl.load(prompt_slots + tokens, mask=token_mask, other=0)
        kv_mask = token_mask[:, None] & dim_mask[None, :]
        k_values = load_kv(
            buffer=k_buffer,
            slots=slots,
            kv_head=kv_head,
            dim_offsets=dim_offsets,
            mask=kv_mask,
            stride_slot=stride_k_slot,
            stride_head=stride_k_head,
        )
        v_values = load_kv(
            buffer=v_buffer,
            slots=slots,
            kv_head=kv_head,
            dim_offsets=dim_offsets,
            mask=kv_mask,
            stride_slot=stride_v_slot,
            stride_head=stride_v_head,
        )
        scores = tl.dot(q_values, tl.trans(k_values))
        scores = tl.where(
            beam_mask[:, None] & token_mask[None, :], scores, negative_infinity
        )
        max_values, normalizers, accumulator = attend_prompt_block(
            scores=scores,
            v_values=v_values,
            max_values=max_values,
            normalizers=normalizers,
            accumulator=accumulator,
            softmax_scale_log2=softmax_scale_log2,
        )

    # The generated tokens: each beam attends its own, one decode step at a time.
    # Their KV slots follow the prompt in the beam's own pool row.
    decode_len = tl.load(seq_lens + leader_row).to(tl.int32) - prompt_len
    beam_pool_rows = tl.load(req_pool_indices + rows, mask=beam_mask, other=0).to(
        tl.int64
    )
    decode_slots = req_to_token + beam_pool_rows * stride_req_to_token + prompt_len
    for step in range(
        tl.maximum(work_start - num_prompt_blocks, 0),
        tl.minimum(work_end - num_prompt_blocks, decode_len),
    ):
        slots = tl.load(decode_slots + step, mask=beam_mask, other=0)
        k_values = load_kv(
            buffer=k_buffer,
            slots=slots,
            kv_head=kv_head,
            dim_offsets=dim_offsets,
            mask=row_mask,
            stride_slot=stride_k_slot,
            stride_head=stride_k_head,
        )
        v_values = load_kv(
            buffer=v_buffer,
            slots=slots,
            kv_head=kv_head,
            dim_offsets=dim_offsets,
            mask=row_mask,
            stride_slot=stride_v_slot,
            stride_head=stride_v_head,
        )
        scores = tl.sum(q_values.to(tl.float32) * k_values.to(tl.float32), axis=1)
        scores = tl.where(beam_mask, scores, negative_infinity)
        max_values, normalizers, accumulator = attend_beam_token(
            scores=scores,
            v_values=v_values,
            max_values=max_values,
            normalizers=normalizers,
            accumulator=accumulator,
            softmax_scale_log2=softmax_scale_log2,
        )
    return rows, beam_mask, max_values, normalizers, accumulator


@triton.jit
def _width_free_stream_k_kernel(
    q,  # [rows, q_heads, HEAD_DIM]; only the real rows are read
    k_buffer,  # [kv_slots, kv_heads, HEAD_DIM], the layer's K cache
    v_buffer,  # [kv_slots, kv_heads, HEAD_DIM], the layer's V cache
    req_to_token,  # [pool rows, max context] int32, the KV slot of each token
    req_pool_indices,  # [rows] int32 or int64, each row's pool row
    seq_lens,  # [rows] int32 or int64; only the leaders' are read
    layout,  # int32: header, tiles, schedule
    output,  # like q; whole tasks and the padding rows are written here
    partial_output,  # [2 * num_ctas, BLOCK_M, HEAD_DIM] fp32, the segments' outputs
    partial_lse,  # [2 * num_ctas, BLOCK_M] fp32, log2 of the segments' softmax sums
    merge_list,  # [num_ctas, MERGE_FIELDS] int32, written for the merge kernel
    softmax_scale,  # float, scales q.k before the softmax
    num_rows,  # q.shape[0]
    num_ctas,  # the grid size
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
    SETUP_UNITS: tl.constexpr,
    SPLIT_COST_UNITS: tl.constexpr,
    TILE_SEARCH: tl.constexpr,
    SCHEDULE_HEADER: tl.constexpr,
    MERGE_FIELDS: tl.constexpr,
    LAYOUT_HEADER: tl.constexpr,
    TILE_FIELDS: tl.constexpr,
):
    """CTA c attends the units of its run. A task that lies in the run goes to the
    output; a segment of a split task goes to a partial slot, and the merge list
    names the split task for the merge kernel."""
    cta = tl.program_id(0)
    beam_offsets = tl.arange(0, BLOCK_M)
    dim_offsets = tl.arange(0, BLOCK_D)
    dim_mask = dim_offsets < HEAD_DIM
    # exp2 is cheaper than exp: fold log2(e) into the scale.
    softmax_scale_log2 = softmax_scale * 1.4426950408889634
    # The layout's parts: the header, the tiles, the schedule.
    num_tiles = tl.load(layout)
    tiles = layout + LAYOUT_HEADER
    schedule = tiles + TILE_FIELDS * num_tiles
    unit_starts = schedule + SCHEDULE_HEADER

    run_start, run_end, total_units = _cta_run(
        schedule=schedule,
        unit_starts=unit_starts,
        num_tiles=num_tiles,
        cta=cta,
        num_ctas=num_ctas,
        Q_HEADS=Q_HEADS,
        SPLIT_COST_UNITS=SPLIT_COST_UNITS,
    )
    first_task = _task_of_unit(
        unit_starts=unit_starts,
        num_tiles=num_tiles,
        unit=run_start,
        Q_HEADS=Q_HEADS,
        TILE_SEARCH=TILE_SEARCH,
    )
    last_task = _task_of_unit(
        unit_starts=unit_starts,
        num_tiles=num_tiles,
        unit=tl.maximum(run_end - 1, 0),
        Q_HEADS=Q_HEADS,
        TILE_SEARCH=TILE_SEARCH,
    )
    _write_merge_entry(
        merge_list=merge_list,
        unit_starts=unit_starts,
        cta=cta,
        num_ctas=num_ctas,
        run_start=run_start,
        run_end=run_end,
        total_units=total_units,
        last_task=last_task,
        Q_HEADS=Q_HEADS,
        SETUP_UNITS=SETUP_UNITS,
        MERGE_FIELDS=MERGE_FIELDS,
    )
    if run_start == run_end:
        # This CTA has no units: the batch has fewer units, or tasks, than CTAs.
        # The merge of a task split around it still reads its slot 2 * cta.
        tl.store(
            partial_lse + 2 * cta * BLOCK_M + beam_offsets,
            tl.full((BLOCK_M,), -float("inf"), tl.float32),
        )
        # No task to attend: the loop below is empty.
        last_task = first_task - 1

    for task in range(first_task, last_task + 1):
        task_start = _task_first_unit(
            unit_starts=unit_starts, task=task, Q_HEADS=Q_HEADS
        )
        task_end = _task_first_unit(
            unit_starts=unit_starts, task=task + 1, Q_HEADS=Q_HEADS
        )
        work_start, work_end = _work_range(
            run_start=run_start,
            run_end=run_end,
            task_start=task_start,
            task_end=task_end,
            SETUP_UNITS=SETUP_UNITS,
        )
        if work_start < work_end:
            q_head = task % Q_HEADS
            rows, beam_mask, max_values, normalizers, accumulator = _attend_task(
                q=q,
                k_buffer=k_buffer,
                v_buffer=v_buffer,
                req_to_token=req_to_token,
                req_pool_indices=req_pool_indices,
                seq_lens=seq_lens,
                tiles=tiles,
                tile=task // Q_HEADS,
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
            )
            row_mask = beam_mask[:, None] & dim_mask[None, :]
            covers_whole_task = (work_start == 0) & (run_end >= task_end)
            if covers_whole_task:
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
            else:
                slot = _segment_slot(
                    cta=cta, run_start=run_start, task_start=task_start
                )
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
def _width_free_merge_kernel(
    partial_output,  # [2 * num_ctas, BLOCK_M, HEAD_DIM] fp32, the segments' outputs
    partial_lse,  # [2 * num_ctas, BLOCK_M] fp32, log2 of the segments' softmax sums
    merge_list,  # [num_ctas, MERGE_FIELDS] int32, from the stream-K kernel
    layout,  # int32: header, tiles, schedule
    output,  # [rows, q_heads, HEAD_DIM]; the rows of split tasks are written
    stride_out_row,  # output.stride(0); in elements
    stride_out_head,  # output.stride(1)
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,  # beams per tile
    BLOCK_D: tl.constexpr,  # HEAD_DIM rounded up to a power of 2
    MERGE_ROWS: tl.constexpr,  # beams per program
    MERGE_FIELDS: tl.constexpr,
    LAYOUT_HEADER: tl.constexpr,
    TILE_FIELDS: tl.constexpr,
):
    """Program (c, chunk) writes MERGE_ROWS beams of the split task CTA c listed.

    The task's segments are in the slot that CTA c listed, then in slot 2 * c' of
    every later CTA c' up to the listed last one. The output is their mean,
    weighted by each segment's softmax sum.
    """
    cta = tl.program_id(0)
    entry = merge_list + MERGE_FIELDS * cta
    tile = tl.load(entry)
    if tile >= 0:
        q_head = tl.load(entry + 1)
        last_cta = tl.load(entry + 2)
        first_segment_slot = tl.load(entry + 3)
        beam_offsets = tl.program_id(1) * MERGE_ROWS + tl.arange(0, MERGE_ROWS)
        tiles = layout + LAYOUT_HEADER
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


def shared_context_attention_width_free_stream_k(
    q: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    layout: torch.Tensor,
    *,
    block_m: int,
    output: torch.Tensor | None = None,
    softmax_scale: float,
    num_ctas: int | None = None,
    split_cost_units: int = _SPLIT_COST_UNITS,
) -> torch.Tensor:
    """shared_context_attention_width_free on a persistent stream-K grid.

    Same inputs and output; `layout` must carry the schedule of its tiles
    (`build_schedule`) after them. Rejects K/V buffers of 2^31 elements or more.
    Tuned on SM120 only.

    For tests and benchmarks:
    - `num_ctas`: the grid size; one CTA per SM when None.
    - `split_cost_units`: a very large value forces whole tasks, a negative one
      equal runs.
    """
    output = check_width_free_inputs(
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

    if num_ctas is None:
        # One CTA per SM: 8 warps of this kernel fill an SM's register file.
        num_ctas = num_sms(q.device.index)
    # Two partial slots per CTA and one merge list entry.
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
    _width_free_stream_k_kernel[(num_ctas,)](
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
        LAYOUT_HEADER=LAYOUT_HEADER,
        TILE_FIELDS=TILE_FIELDS,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )
    _width_free_merge_kernel[(num_ctas, triton.cdiv(block_m, _MERGE_ROWS))](
        partial_output,
        partial_lse,
        merge_list,
        layout,
        output,
        output.stride(0),
        output.stride(1),
        HEAD_DIM=head_dim,
        BLOCK_M=block_m,
        BLOCK_D=block_d,
        MERGE_ROWS=_MERGE_ROWS,
        MERGE_FIELDS=_MERGE_FIELDS,
        LAYOUT_HEADER=LAYOUT_HEADER,
        TILE_FIELDS=TILE_FIELDS,
        # Triton's default; an 8-row merge program has no use for more.
        num_warps=4,
    )
    return output
