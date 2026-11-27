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
"""Shared-context beam attention for beam requests of any widths, in one launch.

Every beam of a request attends to the request's prompt and to the tokens that
beam has generated. The beams of a request share the prompt's K/V, so a program
reads each prompt block once for up to BLOCK_M beams.

Terms:
- group: the rows of one beam request, a leader row and member rows. Beam 0 is
  the leader; beam i >= 1 is member row `first member row + i - 1`.
- pool row: a batch row's row of req_to_token, req_pool_indices[row]. It lists
  the KV slots of the row's tokens.
- tile: up to BLOCK_M consecutive beams of one group. A kernel attends a tile's
  beams together.
- real rows: the rows of the batch; each is in exactly one tile. A CUDA graph
  captured for more rows has padding rows after them; they are in no tile and
  their output is written as zeros.
- work unit: one step of attending a tile for one query head. A tile's work
  units are its prompt blocks (BLOCK_N tokens each), then its decode steps.

The layout is one int32 tensor that the host writes per batch
(srt/beam_search/shared_context_attention_variable_width.py), its parts back to back:
- header: [number of tiles, number of real rows]
- for the stream-K kernel only: its plan
  (beam_shared_context_variable_width_stream_k.py), of one length for any batch
- per tile: (leader row, first member row, members, prompt length, first beam)
The kernels take the parts as separate tensors, views of that one upload
(split_layout). The tensor may be longer than a batch's layout:
the header counts the tiles, and no tile past them is read.
Beam widths live in this tensor, not in the launch arguments, so a graph captured
for a row count replays for any widths that add up to at most that many rows.

KV slots come from req_to_token. Prompt tokens are read through the leader's pool
row. A beam's generated tokens are read through its own pool row, from the prompt
length up to the leader's sequence length: the beams of a group are equally long.
"""

from __future__ import annotations

from typing import NamedTuple

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.beam_shared_context import (
    attend_beam_token,
    attend_prompt_block,
    empty_softmax_state,
    num_sms,
)

# Tokens per prompt block, pipeline stages and warps. Chosen by measurement.
_BLOCK_N = 64
_NUM_STAGES = 3
_NUM_WARPS = 8
# Programs per SM. The grid is fixed, so that a graph can keep it, and program p
# of a query head takes tiles p, p + programs_per_head, ...; more programs than
# SMs even out tiles of different lengths. Chosen by measurement.
_PROGRAMS_PER_SM = 8

# int32 entries of the layout's header and of one tile.
HEADER_FIELDS = 2
TILE_FIELDS = 5
# A work_end past any tile's last work unit: attend to the tile's end.
WHOLE_TILE = 1 << 30


def header_and_tiles_len(num_tiles: int) -> int:
    """int32 entries of a layout's header and tiles, without a stream-K plan."""
    return HEADER_FIELDS + TILE_FIELDS * num_tiles


class LayoutParts(NamedTuple):
    """A layout's parts as the kernels take them: views of the one tensor."""

    # One number each. They stay tensors because a CUDA graph keeps the addresses
    # of a launch's inputs, not the numbers in them.
    tile_count: torch.Tensor  # [1] int32, the batch's tiles
    row_count: torch.Tensor  # [1] int32, the batch's real rows
    plan: torch.Tensor  # the stream-K plan; empty in a layout without one
    tiles: torch.Tensor  # TILE_FIELDS int32 per tile


def split_layout(layout: torch.Tensor, plan_len: int = 0) -> LayoutParts:
    """The parts of a layout whose plan has plan_len entries (0: no plan)."""
    tiles_start = HEADER_FIELDS + plan_len
    return LayoutParts(
        tile_count=layout[0:1],
        row_count=layout[1:2],
        plan=layout[HEADER_FIELDS:tiles_start],
        tiles=layout[tiles_start:],
    )


@triton.jit
def read_tile(
    tiles,  # the layout's tiles, TILE_FIELDS int32 each
    tile_idx,
    beam_offsets,  # [n] int32 positions in the tile; a caller masks those >= BLOCK_M
    TILE_FIELDS: tl.constexpr,
):
    """Batch rows of a tile's beams: (rows, beam_mask, leader_row, prompt_len).

    A group's last tile can hold fewer than BLOCK_M beams. beam_mask is False at
    the positions past them; their row is the leader's, so that it is a valid
    index, and callers mask every load and store with beam_mask.
    """
    tile_fields = tiles + TILE_FIELDS * tile_idx
    leader_row = tl.load(tile_fields)
    first_member_row = tl.load(tile_fields + 1)
    num_members = tl.load(tile_fields + 2)
    prompt_len = tl.load(tile_fields + 3)
    first_beam = tl.load(tile_fields + 4)

    # Beam 0 of a group is its leader, beam i >= 1 its member i - 1.
    beam_indices = first_beam + beam_offsets
    beam_mask = beam_indices <= num_members
    rows = tl.where(beam_indices == 0, leader_row, first_member_row + beam_indices - 1)
    rows = tl.where(beam_mask, rows, leader_row)
    return rows, beam_mask, leader_row, prompt_len


@triton.jit
def _load_q(
    q,  # [rows, q_heads, HEAD_DIM]
    rows,  # [n] int32 rows of q
    q_head,
    dim_offsets,  # [BLOCK_D] 0..BLOCK_D-1
    mask,  # [n, BLOCK_D] bool
    stride_q_row,  # q.stride(0); all strides are in elements
    stride_q_head,  # q.stride(1)
):
    """One query head of the given rows, 0 where masked."""
    return tl.load(
        q
        + rows[:, None] * stride_q_row
        + q_head * stride_q_head
        + dim_offsets[None, :],
        mask=mask,
        other=0.0,
    )


@triton.jit
def _load_kv(
    buffer,  # [kv_slots, kv_heads, HEAD_DIM], the layer's K or V cache
    slots,  # [n] KV slots
    kv_head,
    dim_offsets,  # [BLOCK_D] 0..BLOCK_D-1
    mask,  # [n, BLOCK_D] bool
    stride_slot,  # buffer.stride(0)
    stride_head,  # buffer.stride(1)
):
    """K or V of the given cache slots for one KV head, 0 where masked."""
    return tl.load(
        buffer
        + slots[:, None] * stride_slot
        + kv_head * stride_head
        + dim_offsets[None, :],
        mask=mask,
        other=0.0,
    )


@triton.jit
def store_output(
    output,  # [rows, q_heads, HEAD_DIM]
    rows,  # [n] int32 rows of output
    q_head,
    dim_offsets,  # [BLOCK_D] 0..BLOCK_D-1
    values,  # [n, BLOCK_D]
    mask,  # [n, BLOCK_D] bool
    stride_out_row,  # output.stride(0)
    stride_out_head,  # output.stride(1)
):
    """Write one query head of the given rows where the mask is set."""
    tl.store(
        output
        + rows[:, None] * stride_out_row
        + q_head * stride_out_head
        + dim_offsets[None, :],
        values,
        mask=mask,
    )


@triton.jit
def zero_padding_rows(
    output,  # [rows, q_heads, HEAD_DIM]
    num_real_rows,  # rows of the batch, from the layout's header
    num_rows,  # rows of q and output, at least num_real_rows
    program,  # this program's index among the num_programs that share the rows
    num_programs,
    q_head,
    stride_out_row,  # output.stride(0)
    stride_out_head,  # output.stride(1)
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,  # rows zeroed per store
    BLOCK_D: tl.constexpr,  # HEAD_DIM rounded up to a power of 2
):
    """Zero one query head of the padding rows [num_real_rows, num_rows).

    No tile writes them and the output buffer starts uninitialized. The programs
    share the rows in blocks of BLOCK_M: program p takes blocks p,
    p + num_programs, and so on.
    """
    row_offsets = tl.arange(0, BLOCK_M)
    dim_offsets = tl.arange(0, BLOCK_D)
    first_row = num_real_rows + program * BLOCK_M
    for row_start in range(first_row, num_rows, num_programs * BLOCK_M):
        padding_rows = row_start + row_offsets
        store_output(
            output=output,
            rows=padding_rows,
            q_head=q_head,
            dim_offsets=dim_offsets,
            values=tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32),
            mask=(padding_rows < num_rows)[:, None] & (dim_offsets < HEAD_DIM)[None, :],
            stride_out_row=stride_out_row,
            stride_out_head=stride_out_head,
        )


@triton.jit
def _attend_prompt_blocks(
    q_values,  # [BLOCK_M, BLOCK_D], the tile's queries of one query head
    k_buffer,  # [kv_slots, kv_heads, HEAD_DIM], the layer's K cache
    v_buffer,  # [kv_slots, kv_heads, HEAD_DIM], the layer's V cache
    prompt_slots,  # pointer to the KV slots of the group's prompt tokens
    prompt_len,
    first_block,  # prompt blocks to attend: [first_block, end_block)
    end_block,
    kv_head,
    beam_mask,  # [BLOCK_M] bool, the tile's beams that exist
    max_values,  # the softmax state so far, as attend_prompt_block documents it
    normalizers,
    accumulator,
    softmax_scale_log2,  # softmax_scale * log2(e), for exp2
    stride_k_slot,  # k_buffer.stride(0); all strides are in elements
    stride_k_head,  # k_buffer.stride(1)
    stride_v_slot,  # v_buffer.stride(0)
    stride_v_head,  # v_buffer.stride(1)
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,  # prompt tokens per block
    BLOCK_D: tl.constexpr,  # HEAD_DIM rounded up to a power of 2
    INT64_KV_OFFSETS: tl.constexpr,  # False needs every K/V element offset < 2^31
):
    """All beams of a tile attend blocks [first_block, end_block) of their group's
    prompt, the same BLOCK_N tokens at a time. Returns the softmax state after."""
    token_offsets = tl.arange(0, BLOCK_N)
    dim_offsets = tl.arange(0, BLOCK_D)
    dim_mask = dim_offsets < HEAD_DIM
    negative_infinity = -float("inf")
    # The blocks' tokens; the prompt's last block may hold fewer than BLOCK_N.
    first_token = first_block * BLOCK_N
    end_token = tl.minimum(end_block * BLOCK_N, prompt_len)
    # Stepping by tokens, not by a block index: the two forms compile to code of
    # different speed, so re-time both kernels before changing it.
    for block_start in tl.range(first_token, end_token, BLOCK_N):
        token_indices = block_start + token_offsets
        token_mask = token_indices < prompt_len
        slots = tl.load(prompt_slots + token_indices, mask=token_mask, other=0)
        if INT64_KV_OFFSETS:
            slots = slots.to(tl.int64)
        kv_mask = token_mask[:, None] & dim_mask[None, :]
        k_values = _load_kv(
            buffer=k_buffer,
            slots=slots,
            kv_head=kv_head,
            dim_offsets=dim_offsets,
            mask=kv_mask,
            stride_slot=stride_k_slot,
            stride_head=stride_k_head,
        )
        v_values = _load_kv(
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
    return max_values, normalizers, accumulator


@triton.jit
def _attend_decode_steps(
    q_values,  # [BLOCK_M, BLOCK_D], the tile's queries of one query head
    k_buffer,  # [kv_slots, kv_heads, HEAD_DIM], the layer's K cache
    v_buffer,  # [kv_slots, kv_heads, HEAD_DIM], the layer's V cache
    decode_slots,  # [BLOCK_M] pointers to the KV slots of each beam's generated tokens
    first_step,  # decode steps to attend: [first_step, end_step)
    end_step,
    kv_head,
    beam_mask,  # [BLOCK_M] bool, the tile's beams that exist
    max_values,  # the softmax state so far, as attend_prompt_block documents it
    normalizers,
    accumulator,
    softmax_scale_log2,  # softmax_scale * log2(e), for exp2
    stride_k_slot,  # k_buffer.stride(0); all strides are in elements
    stride_k_head,  # k_buffer.stride(1)
    stride_v_slot,  # v_buffer.stride(0)
    stride_v_head,  # v_buffer.stride(1)
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,  # HEAD_DIM rounded up to a power of 2
    INT64_KV_OFFSETS: tl.constexpr,  # False needs every K/V element offset < 2^31
):
    """Each beam of a tile attends the tokens it generated at decode steps
    [first_step, end_step), one step at a time. Returns the softmax state after."""
    dim_offsets = tl.arange(0, BLOCK_D)
    kv_mask = beam_mask[:, None] & (dim_offsets < HEAD_DIM)[None, :]
    negative_infinity = -float("inf")
    for step_idx in range(first_step, end_step):
        slots = tl.load(decode_slots + step_idx, mask=beam_mask, other=0)
        if INT64_KV_OFFSETS:
            slots = slots.to(tl.int64)
        k_values = _load_kv(
            buffer=k_buffer,
            slots=slots,
            kv_head=kv_head,
            dim_offsets=dim_offsets,
            mask=kv_mask,
            stride_slot=stride_k_slot,
            stride_head=stride_k_head,
        )
        v_values = _load_kv(
            buffer=v_buffer,
            slots=slots,
            kv_head=kv_head,
            dim_offsets=dim_offsets,
            mask=kv_mask,
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
    return max_values, normalizers, accumulator


@triton.jit
def attend_tile(
    q,  # [rows, q_heads, HEAD_DIM]
    k_buffer,  # [kv_slots, kv_heads, HEAD_DIM], the layer's K cache
    v_buffer,  # [kv_slots, kv_heads, HEAD_DIM], the layer's V cache
    req_to_token,  # [pool rows, max context] int32, the KV slot of each token
    req_pool_indices,  # [rows] int32 or int64, each row's pool row
    seq_lens,  # [rows] int32 or int64; only the leaders' are read
    tiles,  # the layout's tiles
    tile_idx,
    q_head,
    work_start,  # work units to attend: [work_start, work_end), clipped to the tile's
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
    INT64_KV_OFFSETS: tl.constexpr,  # False needs every K/V element offset < 2^31
):
    """Attend one tile for one query head over its work units [work_start, work_end).

    Returns (rows, beam_mask, max_values, normalizers, accumulator): the tile's
    batch rows, which of its beams exist, and the softmax state after those work
    units. The attention output of a whole tile is accumulator / normalizers.
    """
    kv_head = q_head // Q_PER_KV
    beam_offsets = tl.arange(0, BLOCK_M)
    dim_offsets = tl.arange(0, BLOCK_D)

    rows, beam_mask, leader_row, prompt_len = read_tile(
        tiles=tiles,
        tile_idx=tile_idx,
        beam_offsets=beam_offsets,
        TILE_FIELDS=TILE_FIELDS,
    )
    q_values = _load_q(
        q=q,
        rows=rows,
        q_head=q_head,
        dim_offsets=dim_offsets,
        mask=beam_mask[:, None] & (dim_offsets < HEAD_DIM)[None, :],
        stride_q_row=stride_q_row,
        stride_q_head=stride_q_head,
    )
    max_values, normalizers, accumulator = empty_softmax_state(
        beam_mask=beam_mask, BLOCK_M=BLOCK_M, BLOCK_D=BLOCK_D
    )

    # The first work units: the prompt's blocks. Its KV slots are in the leader's
    # pool row.
    num_prompt_blocks = tl.cdiv(prompt_len, BLOCK_N)
    leader_pool_row = tl.load(req_pool_indices + leader_row).to(tl.int64)
    max_values, normalizers, accumulator = _attend_prompt_blocks(
        q_values=q_values,
        k_buffer=k_buffer,
        v_buffer=v_buffer,
        prompt_slots=req_to_token + leader_pool_row * stride_req_to_token,
        prompt_len=prompt_len,
        first_block=work_start,
        end_block=tl.minimum(work_end, num_prompt_blocks),
        kv_head=kv_head,
        beam_mask=beam_mask,
        max_values=max_values,
        normalizers=normalizers,
        accumulator=accumulator,
        softmax_scale_log2=softmax_scale_log2,
        stride_k_slot=stride_k_slot,
        stride_k_head=stride_k_head,
        stride_v_slot=stride_v_slot,
        stride_v_head=stride_v_head,
        HEAD_DIM=HEAD_DIM,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
        INT64_KV_OFFSETS=INT64_KV_OFFSETS,
    )

    # The following work units: the decode steps. Each beam's generated tokens
    # follow the prompt in the beam's own pool row.
    decode_len = tl.load(seq_lens + leader_row).to(tl.int32) - prompt_len
    beam_pool_rows = tl.load(req_pool_indices + rows, mask=beam_mask, other=0).to(
        tl.int64
    )
    max_values, normalizers, accumulator = _attend_decode_steps(
        q_values=q_values,
        k_buffer=k_buffer,
        v_buffer=v_buffer,
        decode_slots=req_to_token + beam_pool_rows * stride_req_to_token + prompt_len,
        first_step=tl.maximum(work_start - num_prompt_blocks, 0),
        end_step=tl.minimum(work_end - num_prompt_blocks, decode_len),
        kv_head=kv_head,
        beam_mask=beam_mask,
        max_values=max_values,
        normalizers=normalizers,
        accumulator=accumulator,
        softmax_scale_log2=softmax_scale_log2,
        stride_k_slot=stride_k_slot,
        stride_k_head=stride_k_head,
        stride_v_slot=stride_v_slot,
        stride_v_head=stride_v_head,
        HEAD_DIM=HEAD_DIM,
        BLOCK_D=BLOCK_D,
        INT64_KV_OFFSETS=INT64_KV_OFFSETS,
    )
    return rows, beam_mask, max_values, normalizers, accumulator


@triton.jit
def _variable_width_shared_context_kernel(
    q,  # [rows, q_heads, HEAD_DIM]; only the real rows are read
    k_buffer,  # [kv_slots, kv_heads, HEAD_DIM], the layer's K cache
    v_buffer,  # [kv_slots, kv_heads, HEAD_DIM], the layer's V cache
    req_to_token,  # [pool rows, max context] int32, the KV slot of each token
    req_pool_indices,  # [rows] int32 or int64, each row's pool row
    seq_lens,  # [rows] int32 or int64; only the leaders' are read
    tile_count,  # [1] int32, the batch's tiles
    row_count,  # [1] int32, the batch's real rows
    tiles,  # [tiles, TILE_FIELDS] int32, see the module docstring
    output,  # like q; every row is written: attention, or zeros in padding rows
    softmax_scale,  # float, scales q.k before the softmax
    num_rows,  # q.shape[0]
    stride_q_row,  # q.stride(0); all strides are in elements
    stride_q_head,  # q.stride(1)
    stride_k_slot,  # k_buffer.stride(0)
    stride_k_head,  # k_buffer.stride(1)
    stride_v_slot,  # v_buffer.stride(0)
    stride_v_head,  # v_buffer.stride(1)
    stride_req_to_token,  # req_to_token.stride(0)
    stride_out_row,  # output.stride(0)
    stride_out_head,  # output.stride(1)
    Q_PER_KV: tl.constexpr,  # query heads per KV head
    HEAD_DIM: tl.constexpr,  # q.shape[2]
    BLOCK_M: tl.constexpr,  # beams per tile, the value the layout was built with
    BLOCK_N: tl.constexpr,  # prompt tokens per block
    BLOCK_D: tl.constexpr,  # HEAD_DIM rounded up to a power of 2
    TILE_FIELDS: tl.constexpr,
    WHOLE_TILE: tl.constexpr,
):
    """Program (p, q_head) attends tiles p, p + programs_per_head, ... for one
    query head, each from its first prompt token to its last generated token."""
    program = tl.program_id(0)
    q_head = tl.program_id(1)
    programs_per_head = tl.num_programs(0)
    dim_offsets = tl.arange(0, BLOCK_D)
    # exp2 is cheaper than exp: fold log2(e) into the scale.
    softmax_scale_log2 = softmax_scale * 1.4426950408889634

    num_tiles = tl.load(tile_count)
    for tile_idx in range(program, num_tiles, programs_per_head):
        rows, beam_mask, max_values, normalizers, accumulator = attend_tile(
            q=q,
            k_buffer=k_buffer,
            v_buffer=v_buffer,
            req_to_token=req_to_token,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            tiles=tiles,
            tile_idx=tile_idx,
            q_head=q_head,
            work_start=0,
            work_end=WHOLE_TILE,
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
            # 64-bit K/V offsets for any cache; the stream-K kernel picks per cache.
            INT64_KV_OFFSETS=True,
        )
        store_output(
            output=output,
            rows=rows,
            q_head=q_head,
            dim_offsets=dim_offsets,
            values=accumulator / normalizers[:, None],
            mask=beam_mask[:, None] & (dim_offsets < HEAD_DIM)[None, :],
            stride_out_row=stride_out_row,
            stride_out_head=stride_out_head,
        )

    num_real_rows = tl.load(row_count)
    zero_padding_rows(
        output=output,
        num_real_rows=num_real_rows,
        num_rows=num_rows,
        program=program,
        num_programs=programs_per_head,
        q_head=q_head,
        stride_out_row=stride_out_row,
        stride_out_head=stride_out_head,
        HEAD_DIM=HEAD_DIM,
        BLOCK_M=BLOCK_M,
        BLOCK_D=BLOCK_D,
    )


def _programs_per_head(device_index: int, q_heads: int) -> int:
    """Programs per query head, the grid's first axis; the whole grid then has
    about _PROGRAMS_PER_SM programs per SM. It depends on the device and the
    model only, so a graph keeps it for any batch."""
    return max(1, triton.cdiv(_PROGRAMS_PER_SM * num_sms(device_index), q_heads))


def check_variable_width_inputs(
    q: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    tile_count: torch.Tensor,
    row_count: torch.Tensor,
    tiles: torch.Tensor,
) -> None:
    """Check the inputs that both entry points of this layout take."""
    if q.ndim != 3 or q.stride(2) != 1:
        raise ValueError(
            f"Expected Q [rows, heads, dim] with unit dim stride, got {q.shape}"
        )
    if k_buffer.ndim != 3 or v_buffer.shape != k_buffer.shape:
        raise ValueError("K/V buffers must have matching [slots, heads, dim] shapes")
    if k_buffer.stride(2) != 1 or v_buffer.stride(2) != 1:
        raise ValueError("K/V buffers must have a unit dim stride")
    if q.dtype not in (torch.float16, torch.bfloat16) or not (
        q.dtype == k_buffer.dtype == v_buffer.dtype
    ):
        raise ValueError("Q, K and V must share one of float16 and bfloat16")
    num_rows, q_heads, _ = q.shape
    kv_heads = k_buffer.shape[1]
    if q_heads % kv_heads:
        raise ValueError(f"Q heads {q_heads} must be divisible by KV heads {kv_heads}")
    if req_to_token.dtype != torch.int32 or req_to_token.stride(1) != 1:
        raise ValueError("req_to_token must be int32 with a unit token stride")
    if any(t.dtype != torch.int32 for t in (tile_count, row_count, tiles)):
        raise ValueError("tile_count, row_count and tiles must be int32")
    if tile_count.numel() != 1 or row_count.numel() != 1:
        raise ValueError("tile_count and row_count must hold one number each")
    if req_pool_indices.shape[0] < num_rows or seq_lens.shape[0] < num_rows:
        raise ValueError("req_pool_indices and seq_lens must cover every row of Q")
    if any(
        t.dtype not in (torch.int32, torch.int64) for t in (req_pool_indices, seq_lens)
    ):
        raise ValueError("req_pool_indices and seq_lens must be int32 or int64")


def shared_context_attention_variable_width(
    q: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    tile_count: torch.Tensor,
    row_count: torch.Tensor,
    tiles: torch.Tensor,
    *,
    block_m: int,
    softmax_scale: float,
) -> torch.Tensor:
    """Attention of every beam of every group in one launch, for any beam widths.

    - `q`: [rows, q_heads, head_dim], float16 or bfloat16.
    - `k_buffer`, `v_buffer`: [kv_slots, kv_heads, head_dim], the dtype of `q`.
    - `req_to_token`: [pool rows, max context] int32, the KV slot of each token.
    - `req_pool_indices`, `seq_lens`: [rows] int32 or int64, each row's pool row
      and sequence length.
    - `tile_count`, `row_count`, `tiles`: the parts of the batch's layout
      (split_layout).
    - `block_m`: beams per tile; it must be the tile size the layout was built with.
    Returns the output, like `q`. The padding rows, those past the layout's real
    rows, are zeros.
    One program reads a tile's whole prompt; the stream-K kernel of this layout
    splits long prompts over programs.
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
    output = torch.empty_like(q)
    num_rows, q_heads, head_dim = q.shape
    kv_heads = k_buffer.shape[1]

    grid = (_programs_per_head(q.device.index, q_heads), q_heads)
    _variable_width_shared_context_kernel[grid](
        q,
        k_buffer,
        v_buffer,
        req_to_token,
        req_pool_indices,
        seq_lens,
        tile_count,
        row_count,
        tiles,
        output,
        softmax_scale,
        num_rows,
        q.stride(0),
        q.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        v_buffer.stride(0),
        v_buffer.stride(1),
        req_to_token.stride(0),
        output.stride(0),
        output.stride(1),
        Q_PER_KV=q_heads // kv_heads,
        HEAD_DIM=head_dim,
        BLOCK_M=block_m,
        BLOCK_N=_BLOCK_N,
        BLOCK_D=triton.next_power_of_2(head_dim),
        TILE_FIELDS=TILE_FIELDS,
        WHOLE_TILE=WHOLE_TILE,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )
    return output
