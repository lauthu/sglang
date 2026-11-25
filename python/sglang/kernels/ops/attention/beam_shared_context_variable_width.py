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
- tile: up to BLOCK_M consecutive beams of one group, the work of one program at
  a time.
- real rows: the rows of the batch. A CUDA graph captured for more rows has
  padding rows after them; they are in no tile and their output is written as
  zeros.
- work unit: one step of attending a tile for one query head. A tile's work
  units are its prompt blocks (BLOCK_N tokens each), then its decode steps.

The layout is one int32 tensor that the host writes per batch
(srt/beam_search/shared_context_variable_width.py), its parts back to back:
- header: [number of tiles, number of real rows]
- per tile: (leader row, first member row, members, prompt length, first beam)
- for the stream-K kernel only: its schedule
  (beam_shared_context_variable_width_stream_k.py)
The tensor may be longer than a batch's layout: the header bounds what is read.
Beam widths live in this tensor, not in the launch arguments, so a graph captured
for a row count replays for any widths that add up to at most that many rows.

KV slots come from req_to_token. Prompt tokens are read through the leader's pool
row. A beam's generated tokens are read through its own pool row, from the prompt
length up to the leader's sequence length: the beams of a group are equally long.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

from sglang.kernels.ops.attention.beam_shared_context import (
    attend_beam_token,
    attend_prompt_block,
    empty_softmax_state,
    num_sms,
)

# Measured on RTX PRO 6000 (SM120), Qwen3-0.6B heads, 1-8 groups of 16-2,000 beams:
# 64-token prompt blocks with 3 pipeline stages beat 32 or 128 tokens and 2 stages;
# 4 warps are 1.7x slower.
_BLOCK_N = 64
_NUM_STAGES = 3
_NUM_WARPS = 8
# Programs per SM. The grid is fixed, so that a graph can keep it, and program p
# of a query head takes tiles p, p + programs_per_head, ...; more programs than
# SMs even out tiles of different lengths. Measured, same setup (2-16 tried): 8
# is fastest for 4 groups of 2,000 beams and within 3 us of the best elsewhere.
_PROGRAMS_PER_SM = 8

# int32 entries of the layout's header and of one tile.
LAYOUT_HEADER = 2
TILE_FIELDS = 5
# An end of a work unit range that is past any tile's last unit: the whole tile.
WHOLE_TILE = 1 << 30


def header_and_tiles_len(num_tiles: int) -> int:
    """int32 entries of a layout's header and tiles, without a schedule."""
    return LAYOUT_HEADER + TILE_FIELDS * num_tiles


@triton.jit
def tile_rows(
    tiles,  # the layout's tiles, TILE_FIELDS int32 each
    tile,  # index of the tile to read
    beam_offsets,  # [n] int32 positions in the tile; a caller masks those >= BLOCK_M
    TILE_FIELDS: tl.constexpr,
):
    """Batch rows of a tile's beams: (rows, beam_mask, leader_row, prompt_len).

    A group's last tile can hold fewer than BLOCK_M beams. beam_mask is False at
    the positions past them; their row is the leader's, so that it is a valid
    index, and callers mask every load and store with beam_mask.
    """
    tile_fields = tiles + TILE_FIELDS * tile
    leader_row = tl.load(tile_fields)
    first_member_row = tl.load(tile_fields + 1)
    num_members = tl.load(tile_fields + 2)
    prompt_len = tl.load(tile_fields + 3)
    first_beam = tl.load(tile_fields + 4)

    # Beam 0 of a group is its leader, beam i >= 1 its member i - 1.
    beams = first_beam + beam_offsets
    beam_mask = beams <= num_members
    rows = tl.where(beams == 0, leader_row, first_member_row + beams - 1)
    rows = tl.where(beam_mask, rows, leader_row)
    return rows, beam_mask, leader_row, prompt_len


@triton.jit
def load_q(
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
def load_kv(
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
def attend_prompt_blocks(
    q_values,  # [BLOCK_M, BLOCK_D], the tile's queries of one query head
    k_buffer,  # [kv_slots, kv_heads, HEAD_DIM], the layer's K cache
    v_buffer,  # [kv_slots, kv_heads, HEAD_DIM], the layer's V cache
    prompt_slots,  # pointer to the KV slots of the group's prompt tokens
    prompt_len,
    first_block,  # prompt blocks to attend: [first_block, end_block)
    end_block,
    kv_head,
    beam_mask,  # [BLOCK_M] bool, the tile's beams that exist
    max_values,  # the softmax state so far, see empty_softmax_state
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
    WIDE_KV_OFFSETS: tl.constexpr,  # 64-bit K/V offsets; False needs < 2^31 elements
):
    """All beams of a tile attend blocks [first_block, end_block) of their group's
    prompt, the same BLOCK_N tokens at a time. Returns the softmax state after."""
    context_offsets = tl.arange(0, BLOCK_N)
    dim_offsets = tl.arange(0, BLOCK_D)
    dim_mask = dim_offsets < HEAD_DIM
    negative_infinity = -float("inf")
    # The loop steps by tokens, as the default kernel's own loop did before the two
    # kernels shared this one.
    first_token = first_block * BLOCK_N
    end_token = tl.minimum(end_block * BLOCK_N, prompt_len)
    for block_start in tl.range(first_token, end_token, BLOCK_N):
        tokens = block_start + context_offsets
        token_mask = tokens < prompt_len
        slots = tl.load(prompt_slots + tokens, mask=token_mask, other=0)
        if WIDE_KV_OFFSETS:
            slots = slots.to(tl.int64)
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
    return max_values, normalizers, accumulator


@triton.jit
def attend_decode_steps(
    q_values,  # [BLOCK_M, BLOCK_D], the tile's queries of one query head
    k_buffer,  # [kv_slots, kv_heads, HEAD_DIM], the layer's K cache
    v_buffer,  # [kv_slots, kv_heads, HEAD_DIM], the layer's V cache
    decode_slots,  # [BLOCK_M] pointers to the KV slots of each beam's generated tokens
    first_step,  # decode steps to attend: [first_step, end_step)
    end_step,
    kv_head,
    beam_mask,  # [BLOCK_M] bool, the tile's beams that exist
    max_values,  # the softmax state so far, see empty_softmax_state
    normalizers,
    accumulator,
    softmax_scale_log2,  # softmax_scale * log2(e), for exp2
    stride_k_slot,  # k_buffer.stride(0); all strides are in elements
    stride_k_head,  # k_buffer.stride(1)
    stride_v_slot,  # v_buffer.stride(0)
    stride_v_head,  # v_buffer.stride(1)
    HEAD_DIM: tl.constexpr,
    BLOCK_D: tl.constexpr,  # HEAD_DIM rounded up to a power of 2
    WIDE_KV_OFFSETS: tl.constexpr,  # 64-bit K/V offsets; False needs < 2^31 elements
):
    """Each beam of a tile attends the tokens it generated at decode steps
    [first_step, end_step), one step at a time. Returns the softmax state after."""
    dim_offsets = tl.arange(0, BLOCK_D)
    row_mask = beam_mask[:, None] & (dim_offsets < HEAD_DIM)[None, :]
    negative_infinity = -float("inf")
    for step in range(first_step, end_step):
        slots = tl.load(decode_slots + step, mask=beam_mask, other=0)
        if WIDE_KV_OFFSETS:
            slots = slots.to(tl.int64)
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
    tile,  # index of the tile to attend
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
    WIDE_KV_OFFSETS: tl.constexpr,  # 64-bit K/V offsets; False needs < 2^31 elements
):
    """Attend one tile for one query head over its work units [work_start, work_end).

    Returns (rows, beam_mask, max_values, normalizers, accumulator): the tile's
    batch rows, which of its beams exist, and the softmax state after the units.
    The attention output of a whole tile is accumulator / normalizers.
    """
    kv_head = q_head // Q_PER_KV
    beam_offsets = tl.arange(0, BLOCK_M)
    dim_offsets = tl.arange(0, BLOCK_D)

    rows, beam_mask, leader_row, prompt_len = tile_rows(
        tiles=tiles, tile=tile, beam_offsets=beam_offsets, TILE_FIELDS=TILE_FIELDS
    )
    q_values = load_q(
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
    max_values, normalizers, accumulator = attend_prompt_blocks(
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
        WIDE_KV_OFFSETS=WIDE_KV_OFFSETS,
    )

    # The following work units: the decode steps. Each beam's generated tokens
    # follow the prompt in the beam's own pool row.
    decode_len = tl.load(seq_lens + leader_row).to(tl.int32) - prompt_len
    beam_pool_rows = tl.load(req_pool_indices + rows, mask=beam_mask, other=0).to(
        tl.int64
    )
    max_values, normalizers, accumulator = attend_decode_steps(
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
        WIDE_KV_OFFSETS=WIDE_KV_OFFSETS,
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
    layout,  # int32, see the module docstring
    output,  # like q; every row is written, the padding rows with zeros
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
    LAYOUT_HEADER: tl.constexpr,
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

    num_tiles = tl.load(layout)
    tiles = layout + LAYOUT_HEADER
    for tile in range(program, num_tiles, programs_per_head):
        rows, beam_mask, max_values, normalizers, accumulator = attend_tile(
            q=q,
            k_buffer=k_buffer,
            v_buffer=v_buffer,
            req_to_token=req_to_token,
            req_pool_indices=req_pool_indices,
            seq_lens=seq_lens,
            tiles=tiles,
            tile=tile,
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
            WIDE_KV_OFFSETS=True,
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

    num_real_rows = tl.load(layout + 1)
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
    layout: torch.Tensor,
    output: torch.Tensor | None,
) -> torch.Tensor:
    """Check the inputs that both kernels of this layout read. Returns the output
    tensor to write: the given one, or a new one like `q`."""
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
    num_rows, q_heads, head_dim = q.shape
    kv_heads = k_buffer.shape[1]
    if q_heads % kv_heads:
        raise ValueError(f"Q heads {q_heads} must be divisible by KV heads {kv_heads}")
    if req_to_token.dtype != torch.int32 or req_to_token.stride(1) != 1:
        raise ValueError("req_to_token must be int32 with a unit token stride")
    if layout.dtype != torch.int32 or layout.ndim != 1 or len(layout) < LAYOUT_HEADER:
        raise ValueError("The layout must be a 1-D int32 tensor with its header")
    if req_pool_indices.shape[0] < num_rows or seq_lens.shape[0] < num_rows:
        raise ValueError("req_pool_indices and seq_lens must cover every row of Q")
    if any(
        t.dtype not in (torch.int32, torch.int64) for t in (req_pool_indices, seq_lens)
    ):
        raise ValueError("req_pool_indices and seq_lens must be int32 or int64")
    if output is None:
        return torch.empty_like(q)
    if output.shape != q.shape or output.stride(2) != 1:
        raise ValueError("Output must match Q's shape with a unit dim stride")
    return output


def shared_context_attention_variable_width(
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
) -> torch.Tensor:
    """Attention of every beam of every group in one launch, for any group widths.

    - `q`: [rows, q_heads, head_dim], float16 or bfloat16, like the 3-D K/V buffers.
    - `layout`: the module's int32 layout of the batch.
    - `block_m`: beams per tile; it must be the tile size the layout was built with.
    - `output`: like `q`, allocated when None. The padding rows, those past the
      layout's real rows, are written as zeros.
    Tuned on SM120 only. One program reads a tile's whole prompt; the stream-K
    kernel of this layout splits long prompts over programs.
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
        layout,
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
        LAYOUT_HEADER=LAYOUT_HEADER,
        TILE_FIELDS=TILE_FIELDS,
        WHOLE_TILE=WHOLE_TILE,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )
    return output
