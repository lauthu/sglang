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
"""Shared-context beam attention for any split of a batch's rows into groups.

A group is one request: a leader row plus member rows that share its prompt KV.
Groups may have any width, so one launch serves every width in a batch and a
CUDA graph captured for a row count replays for any split of those rows.

The per-batch layout is one int32 tensor, read on the GPU, its parts back to back:
- header: [tiles, real rows]
- tiles: 5 per tile, (leader row, first member row, members, prompt length, first
  beam); a tile covers block_m beams of one group, beam 0 is the leader, and
  member rows are contiguous
- for the stream-K kernel, its schedule (beam_shared_context_width_free_stream_k.py)
A buffer may be longer than its batch's layout: the tile count bounds what is read.

KV slots come from req_to_token: the prompt through the leader's pool row and
each beam's decode tokens through its own pool row, up to its sequence length.
"""

from __future__ import annotations

import functools

import torch
import triton
import triton.language as tl

# Measured on RTX PRO 6000 (SM120), Qwen3-0.6B heads, 256-2,000 beams: 64-token
# blocks with 3 stages beat 128 or 2 stages; 8 waves of programs balance many tiles.
_BLOCK_N = 64
_NUM_WARPS = 8
_NUM_STAGES = 3
_WAVES = 8


# int32 entries of the layout's header and of one tile.
LAYOUT_HEADER = 2
TILE_FIELDS = 5


def layout_tables_size(num_tiles: int) -> int:
    """int32 entries of a layout's header and tile table."""
    return LAYOUT_HEADER + TILE_FIELDS * num_tiles


@triton.jit
def tile_rows(tiles, tile, lanes, TILE_FIELDS: tl.constexpr):
    """Rows of a tile's beams and their mask, then its group's leader row and
    prompt length."""
    entry = tiles + TILE_FIELDS * tile
    leader = tl.load(entry)
    first_member = tl.load(entry + 1)
    num_members = tl.load(entry + 2)
    prompt_len = tl.load(entry + 3)
    beams = tl.load(entry + 4) + lanes
    beam_mask = beams <= num_members
    rows = tl.where(beams == 0, leader, first_member + beams - 1)
    rows = tl.where(beam_mask, rows, leader)
    return rows, beam_mask, leader, prompt_len


@triton.jit
def load_kv(buffer, slots, kv_head, dims, mask, stride_slot, stride_head):
    """K or V of the given cache slots for one KV head, 0 where masked."""
    return tl.load(
        buffer + slots[:, None] * stride_slot + kv_head * stride_head + dims[None, :],
        mask=mask,
        other=0.0,
    )


@triton.jit
def zero_padded_rows(
    output,
    num_real_rows,
    num_rows,
    cta,
    num_ctas,
    q_head,
    stride_out_row,
    stride_out_head,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    """Zero one query head of the rows past the real ones, which pad a captured row
    count: no tile writes them, and the output buffer starts uninitialized. Program
    cta takes every num_ctas-th block of BLOCK_M rows."""
    lanes = tl.arange(0, BLOCK_M)
    dims = tl.arange(0, BLOCK_D)
    for row_start in range(num_real_rows + cta * BLOCK_M, num_rows, num_ctas * BLOCK_M):
        pad_rows = row_start + lanes
        tl.store(
            output
            + pad_rows[:, None] * stride_out_row
            + q_head * stride_out_head
            + dims[None, :],
            tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32),
            mask=(pad_rows < num_rows)[:, None] & (dims < HEAD_DIM)[None, :],
        )


@triton.jit
def empty_softmax_state(beam_mask, BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr):
    max_values = tl.where(beam_mask, -float("inf"), 0.0)
    normalizers = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    return max_values, normalizers, accumulator


@triton.jit
def attend_prompt_block(
    scores,
    v_values,
    max_values,
    normalizers,
    accumulator,
    softmax_scale_log2,
):
    # Online softmax step over a block of prompt tokens; masked scores are -inf.
    new_max = tl.maximum(max_values, tl.max(scores, axis=1))
    previous_scale = tl.exp2((max_values - new_max) * softmax_scale_log2)
    probabilities = tl.exp2((scores - new_max[:, None]) * softmax_scale_log2)
    normalizers = normalizers * previous_scale + tl.sum(probabilities, axis=1)
    accumulator = accumulator * previous_scale[:, None] + tl.dot(
        probabilities.to(v_values.dtype),
        v_values,
    )
    return new_max, normalizers, accumulator


@triton.jit
def attend_beam_token(
    scores,
    v_values,
    max_values,
    normalizers,
    accumulator,
    softmax_scale_log2,
):
    # Online softmax step over one decode token per beam; masked scores are -inf.
    new_max = tl.maximum(max_values, scores)
    previous_scale = tl.exp2((max_values - new_max) * softmax_scale_log2)
    current_scale = tl.exp2((scores - new_max) * softmax_scale_log2)
    normalizers = normalizers * previous_scale + current_scale
    accumulator = (
        accumulator * previous_scale[:, None]
        + v_values.to(tl.float32) * current_scale[:, None]
    )
    return new_max, normalizers, accumulator


@functools.lru_cache(maxsize=None)
def num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


@triton.jit
def _width_free_shared_context_kernel(
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
    stride_q_row,
    stride_q_head,
    stride_k_slot,
    stride_k_head,
    stride_v_slot,
    stride_v_head,
    stride_req_to_token,
    stride_out_row,
    stride_out_head,
    Q_PER_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
    LAYOUT_HEADER: tl.constexpr,
    TILE_FIELDS: tl.constexpr,
):
    cta = tl.program_id(0)
    q_head = tl.program_id(1)
    num_ctas = tl.num_programs(0)
    kv_head = q_head // Q_PER_KV

    lanes = tl.arange(0, BLOCK_M)
    context_offsets = tl.arange(0, BLOCK_N)
    dims = tl.arange(0, BLOCK_D)
    dim_mask = dims < HEAD_DIM
    negative_infinity = -float("inf")
    # exp2 is cheaper than exp: fold log2(e) into the scale.
    softmax_scale_log2 = softmax_scale * 1.4426950408889634

    num_tiles = tl.load(layout)
    tiles = layout + LAYOUT_HEADER
    for tile in range(cta, num_tiles, num_ctas):
        rows, beam_mask, leader, prompt_len = tile_rows(tiles, tile, lanes, TILE_FIELDS)
        row_mask = beam_mask[:, None] & dim_mask[None, :]

        q_values = tl.load(
            q + rows[:, None] * stride_q_row + q_head * stride_q_head + dims[None, :],
            mask=row_mask,
            other=0.0,
        )
        max_values, normalizers, accumulator = empty_softmax_state(
            beam_mask, BLOCK_M, BLOCK_D
        )

        leader_pool = tl.load(req_pool_indices + leader).to(tl.int64)
        prompt_slots = req_to_token + leader_pool * stride_req_to_token
        for block_start in tl.range(0, prompt_len, BLOCK_N):
            tokens = block_start + context_offsets
            token_mask = tokens < prompt_len
            slots = tl.load(prompt_slots + tokens, mask=token_mask, other=0).to(
                tl.int64
            )
            kv_mask = token_mask[:, None] & dim_mask[None, :]
            k_values = load_kv(
                k_buffer, slots, kv_head, dims, kv_mask, stride_k_slot, stride_k_head
            )
            v_values = load_kv(
                v_buffer, slots, kv_head, dims, kv_mask, stride_v_slot, stride_v_head
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
        row_pools = tl.load(req_pool_indices + rows, mask=beam_mask, other=0).to(
            tl.int64
        )
        decode_slots = req_to_token + row_pools * stride_req_to_token + prompt_len
        for step in range(0, decode_len):
            slots = tl.load(decode_slots + step, mask=beam_mask, other=0).to(tl.int64)
            k_values = load_kv(
                k_buffer, slots, kv_head, dims, row_mask, stride_k_slot, stride_k_head
            )
            v_values = load_kv(
                v_buffer, slots, kv_head, dims, row_mask, stride_v_slot, stride_v_head
            )
            scores = tl.sum(
                q_values.to(tl.float32) * k_values.to(tl.float32),
                axis=1,
            )
            scores = tl.where(beam_mask, scores, negative_infinity)
            max_values, normalizers, accumulator = attend_beam_token(
                scores,
                v_values,
                max_values,
                normalizers,
                accumulator,
                softmax_scale_log2,
            )

        tl.store(
            output
            + rows[:, None] * stride_out_row
            + q_head * stride_out_head
            + dims[None, :],
            accumulator / normalizers[:, None],
            mask=row_mask,
        )

    zero_padded_rows(
        output,
        tl.load(layout + 1),
        num_rows,
        cta,
        num_ctas,
        q_head,
        stride_out_row,
        stride_out_head,
        HEAD_DIM,
        BLOCK_M,
        BLOCK_D,
    )


def width_free_num_ctas(device_index: int, q_heads: int) -> int:
    """Programs per query head; fixed per device and model, so a graph keeps it."""
    return max(1, triton.cdiv(_WAVES * num_sms(device_index), q_heads))


def check_width_free_inputs(
    q: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    layout: torch.Tensor,
    output: torch.Tensor | None,
) -> torch.Tensor:
    """Check the inputs every kernel of this layout reads; returns the output to
    write."""
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


def shared_context_attention_width_free(
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

    `q` is [rows, q_heads, head_dim] float16 or bfloat16, as are the 3-D K/V
    buffers; `layout` is the module's int32 layout; rows past its real rows are
    written as zeros. Tuned on SM120 only; one CTA reads a tile's whole prompt.
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
    num_rows, q_heads, head_dim = q.shape
    kv_heads = k_buffer.shape[1]

    grid = (width_free_num_ctas(q.device.index, q_heads), q_heads)
    _width_free_shared_context_kernel[grid](
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
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )
    return output
