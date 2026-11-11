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
"""Shared-context beam attention for any split of a step's rows into groups.

A group is one request: a leader row plus member rows that share its prompt KV.
Groups may have any width, so one launch serves every width in a step and a
CUDA graph captured for a row count replays for any split of those rows.

The per-step layout is one int32 tensor, read on the GPU:
- counts: [groups, tiles, real rows]
- groups: [group_capacity, 4] of (leader row, first member row, members,
  prompt length); member rows are contiguous
- tiles: [tile_capacity, 2] of (group, first beam); beam 0 is the leader and a
  tile covers block_m beams of one group (block_m comes with the layout)

KV slots come from req_to_token: the prompt through the leader's pool row and
each beam's decode tokens through its own pool row, up to its sequence length.
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

# Measured on RTX PRO 6000 (SM120), Qwen3-0.6B heads, 256-2,000 beams: 64-token
# blocks with 3 stages beat 128 or 2 stages; 8 waves of programs balance many tiles.
_BLOCK_N = 64
_NUM_WARPS = 8
_NUM_STAGES = 3
_WAVES = 8


@triton.jit
def _ragged_shared_context_kernel(
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
    softmax_scale_log2 = softmax_scale * 1.4426950408889634

    num_tiles = tl.load(counts + 1)
    for tile in range(cta, num_tiles, num_ctas):
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
        row_pools = tl.load(req_pool_indices + rows, mask=beam_mask, other=0).to(
            tl.int64
        )
        decode_slots = req_to_token + row_pools * stride_req_to_token + prompt_len
        for step in range(0, decode_len):
            slots = tl.load(decode_slots + step, mask=beam_mask, other=0).to(tl.int64)
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

    # Rows past the real ones pad a captured row count; zero them.
    num_real_rows = tl.load(counts + 2)
    for row_start in range(num_real_rows + cta * BLOCK_M, num_rows, num_ctas * BLOCK_M):
        pad_rows = row_start + lanes
        tl.store(
            output
            + pad_rows[:, None] * stride_out_row
            + q_head * stride_out_head
            + dims[None, :],
            tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32),
            mask=(pad_rows < num_rows)[:, None] & dim_mask[None, :],
        )


def ragged_num_ctas(device_index: int, q_heads: int) -> int:
    """Programs per query head; fixed per device and model, so a graph keeps it."""
    return max(1, triton.cdiv(_WAVES * num_sms(device_index), q_heads))


def check_ragged_inputs(
    q: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    counts: torch.Tensor,
    groups: torch.Tensor,
    tiles: torch.Tensor,
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
    num_rows, q_heads, head_dim = q.shape
    kv_heads = k_buffer.shape[1]
    if q_heads % kv_heads:
        raise ValueError(f"Q heads {q_heads} must be divisible by KV heads {kv_heads}")
    if req_to_token.dtype != torch.int32 or req_to_token.stride(1) != 1:
        raise ValueError("req_to_token must be int32 with a unit token stride")
    if counts.shape != (3,) or groups.ndim != 2 or groups.shape[1] != 4:
        raise ValueError("Expected counts [3] and groups [capacity, 4]")
    if tiles.ndim != 2 or tiles.shape[1] != 2:
        raise ValueError("Expected tiles [capacity, 2]")
    if not all(t.dtype == torch.int32 for t in (counts, groups, tiles)):
        raise ValueError("The group layout must be int32")
    if req_pool_indices.shape[0] < num_rows or seq_lens.shape[0] < num_rows:
        raise ValueError("req_pool_indices and seq_lens must cover every row of Q")
    if output is None:
        return torch.empty_like(q)
    if output.shape != q.shape or output.stride(2) != 1:
        raise ValueError("Output must match Q's shape with a unit dim stride")
    return output


def shared_context_attention_ragged(
    q: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    req_to_token: torch.Tensor,
    req_pool_indices: torch.Tensor,
    seq_lens: torch.Tensor,
    counts: torch.Tensor,
    groups: torch.Tensor,
    tiles: torch.Tensor,
    *,
    block_m: int,
    output: torch.Tensor | None = None,
    softmax_scale: float,
) -> torch.Tensor:
    """Attention of every beam of every group in one launch, for any group widths.

    `q` is [rows, q_heads, head_dim]; rows past `counts[2]` are written as zeros.
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
    num_rows, q_heads, head_dim = q.shape
    kv_heads = k_buffer.shape[1]

    grid = (ragged_num_ctas(q.device.index, q_heads), q_heads)
    _ragged_shared_context_kernel[grid](
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
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )
    return output
