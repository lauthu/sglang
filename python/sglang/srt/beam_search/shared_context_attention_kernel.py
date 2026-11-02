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
"""Native Triton shared-context attention for wide beam decoding.

Every beam of a group attends to the group's shared prompt and to its own
decode tokens. The prompt part runs as tensor-core tiles of BLOCK_M beams of
one query head, split along the prompt (split-KV) so that a few groups still
fill the GPU; the last split adds the beams' own decode tokens with CUDA-core
dot products, and a merge kernel combines the splits' partial results.
"""

from __future__ import annotations

import functools
from typing import Optional

import torch
import triton
import triton.language as tl

# Measured on RTX PRO 6000 (SM120) at 256 beams, head_dim 128: fastest at 1-8
# groups and 256-2048 prompt tokens.
_BLOCK_M = 128
_BLOCK_N = 64
_NUM_WARPS = 8
_NUM_STAGES = 2
_MAX_SPLITS = 8
# Measured: the smallest split count that fills this share of the last wave is
# within 1-2% of the best one at 1-8 groups; more splits only add merge traffic.
_MIN_WAVE_EFFICIENCY = 0.8


@triton.jit
def _beam_rows(group, beams, num_groups, BEAM_WIDTH: tl.constexpr):
    # Scheduler row layout: one leader row per group, then the groups' other beams.
    return tl.where(
        beams == 0,
        group,
        num_groups + group * (BEAM_WIDTH - 1) + beams - 1,
    )


@triton.jit
def _shared_context_attention_kernel(
    q,
    k_buffer,
    v_buffer,
    context_slots,
    context_cu_seqlens,
    decode_slots,
    decode_lens,
    output,
    partial_output,
    partial_lse,
    softmax_scale,
    num_groups,
    num_splits,
    stride_q_row,
    stride_q_head,
    stride_k_slot,
    stride_k_head,
    stride_v_slot,
    stride_v_head,
    stride_decode_group,
    stride_out_row,
    stride_out_head,
    stride_partial_split,
    stride_partial_row,
    stride_partial_head,
    stride_lse_split,
    stride_lse_row,
    BEAM_WIDTH: tl.constexpr,
    Q_PER_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    WRITE_PARTIALS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    group = tl.program_id(0) // num_splits
    split = tl.program_id(0) % num_splits
    q_head = tl.program_id(1)
    kv_head = q_head // Q_PER_KV

    beams = tl.program_id(2) * BLOCK_M + tl.arange(0, BLOCK_M)
    beam_mask = beams < BEAM_WIDTH
    rows = _beam_rows(group, beams, num_groups, BEAM_WIDTH)
    context_offsets = tl.arange(0, BLOCK_N)
    dims = tl.arange(0, BLOCK_D)
    dim_mask = dims < HEAD_DIM
    row_mask = beam_mask[:, None] & dim_mask[None, :]

    q_values = tl.load(
        q + rows[:, None] * stride_q_row + q_head * stride_q_head + dims[None, :],
        mask=row_mask,
        other=0.0,
    )

    negative_infinity = -float("inf")
    max_values = tl.where(beam_mask, negative_infinity, 0.0)
    normalizers = tl.zeros((BLOCK_M,), dtype=tl.float32)
    accumulator = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    softmax_scale_log2 = softmax_scale * 1.4426950408889634

    context_start = tl.load(context_cu_seqlens + group)
    context_length = tl.load(context_cu_seqlens + group + 1) - context_start
    # Split bounds come from the runtime length: a graph replays shorter prompts
    # than it was captured with, which leaves trailing splits empty.
    split_blocks = tl.cdiv(tl.cdiv(context_length, BLOCK_N), num_splits)
    split_start = split * split_blocks * BLOCK_N
    split_end = tl.minimum(split_start + split_blocks * BLOCK_N, context_length)

    for block_start in tl.range(split_start, split_end, BLOCK_N):
        tokens = block_start + context_offsets
        token_mask = tokens < split_end
        slots = tl.load(
            context_slots + context_start + tokens,
            mask=token_mask,
            other=0,
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
        scores = tl.where(token_mask[None, :], scores, negative_infinity)
        new_max = tl.maximum(max_values, tl.max(scores, axis=1))
        previous_scale = tl.exp2((max_values - new_max) * softmax_scale_log2)
        probabilities = tl.exp2((scores - new_max[:, None]) * softmax_scale_log2)
        normalizers = normalizers * previous_scale + tl.sum(probabilities, axis=1)
        accumulator = accumulator * previous_scale[:, None] + tl.dot(
            probabilities.to(v_values.dtype),
            v_values,
        )
        max_values = new_max

    if split == num_splits - 1:
        decode_length = tl.load(decode_lens + group)
        for step in range(0, decode_length):
            slots = tl.load(
                decode_slots + group * stride_decode_group + step * BEAM_WIDTH + beams,
                mask=beam_mask,
                other=0,
            )
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
            new_max = tl.maximum(max_values, scores)
            previous_scale = tl.exp2((max_values - new_max) * softmax_scale_log2)
            current_scale = tl.exp2((scores - new_max) * softmax_scale_log2)
            normalizers = normalizers * previous_scale + current_scale
            accumulator = (
                accumulator * previous_scale[:, None]
                + v_values.to(tl.float32) * current_scale[:, None]
            )
            max_values = new_max

    if WRITE_PARTIALS:
        # An empty split has no mass; its -inf log-sum-exp drops it from the merge.
        has_mass = normalizers > 0
        result = accumulator / tl.where(has_mass, normalizers, 1.0)[:, None]
        lse = tl.where(
            has_mass,
            max_values * softmax_scale_log2 + tl.log2(normalizers),
            negative_infinity,
        )
        tl.store(
            partial_output
            + split * stride_partial_split
            + rows[:, None] * stride_partial_row
            + q_head * stride_partial_head
            + dims[None, :],
            result,
            mask=row_mask,
        )
        tl.store(
            partial_lse + split * stride_lse_split + rows * stride_lse_row + q_head,
            lse,
            mask=beam_mask,
        )
    else:
        tl.store(
            output
            + rows[:, None] * stride_out_row
            + q_head * stride_out_head
            + dims[None, :],
            accumulator / normalizers[:, None],
            mask=row_mask,
        )


@triton.jit
def _merge_splits_kernel(
    partial_output,
    partial_lse,
    output,
    num_splits,
    stride_partial_split,
    stride_partial_row,
    stride_partial_head,
    stride_lse_split,
    stride_lse_row,
    stride_out_row,
    stride_out_head,
    Q_HEADS: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_H: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    row = tl.program_id(0)
    heads = tl.arange(0, BLOCK_H)
    dims = tl.arange(0, BLOCK_D)
    head_mask = heads < Q_HEADS
    mask = head_mask[:, None] & (dims < HEAD_DIM)[None, :]
    lse_pointers = partial_lse + row * stride_lse_row + heads

    negative_infinity = -float("inf")
    max_lse = tl.full((BLOCK_H,), negative_infinity, tl.float32)
    for split in range(0, num_splits):
        lse = tl.load(
            lse_pointers + split * stride_lse_split,
            mask=head_mask,
            other=negative_infinity,
        )
        max_lse = tl.maximum(max_lse, lse)
    max_lse = tl.where(max_lse == negative_infinity, 0.0, max_lse)

    total = tl.zeros((BLOCK_H,), tl.float32)
    accumulator = tl.zeros((BLOCK_H, BLOCK_D), tl.float32)
    for split in range(0, num_splits):
        lse = tl.load(
            lse_pointers + split * stride_lse_split,
            mask=head_mask,
            other=negative_infinity,
        )
        weight = tl.exp2(lse - max_lse)
        part = tl.load(
            partial_output
            + split * stride_partial_split
            + row * stride_partial_row
            + heads[:, None] * stride_partial_head
            + dims[None, :],
            mask=mask,
            other=0.0,
        )
        accumulator += part * weight[:, None]
        total += weight

    tl.store(
        output
        + row * stride_out_row
        + heads[:, None] * stride_out_head
        + dims[None, :],
        accumulator / total[:, None],
        mask=mask,
    )


@functools.lru_cache(maxsize=None)
def _num_sms(device_index: int) -> int:
    return torch.cuda.get_device_properties(device_index).multi_processor_count


def num_kv_splits(num_tiles: int, num_sms: int) -> int:
    """Pick the prompt split count from static shapes only, so graphs keep it."""
    best_splits, best_efficiency = 1, 0.0
    for splits in range(1, _MAX_SPLITS + 1):
        ctas = num_tiles * splits
        efficiency = ctas / (triton.cdiv(ctas, num_sms) * num_sms)
        if efficiency >= _MIN_WAVE_EFFICIENCY:
            return splits
        if efficiency > best_efficiency:
            best_splits, best_efficiency = splits, efficiency
    return best_splits


def shared_context_attention(
    q: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    context_slots: torch.Tensor,
    context_cu_seqlens: torch.Tensor,
    decode_slots: torch.Tensor,
    decode_lens: torch.Tensor,
    *,
    beam_width: int,
    softmax_scale: float,
    num_splits: Optional[int] = None,
) -> torch.Tensor:
    """Apply shared prompt and per-beam suffix attention to beam rows.

    `q` is [rows, q_heads, head_dim] in the scheduler's row layout (each group's
    leader row first, then all groups' other beams); the output has the same
    shape and layout. `decode_slots` is [groups, capacity * beam_width], step-major.
    """
    if q.ndim != 3:
        raise ValueError(f"Expected Q with 3 dimensions, got {q.shape}")
    if k_buffer.ndim != 3 or v_buffer.shape != k_buffer.shape:
        raise ValueError("K/V buffers must have matching [slots, heads, dim] shapes")
    if q.stride(2) != 1 or k_buffer.stride(2) != 1 or v_buffer.stride(2) != 1:
        raise ValueError("Q and K/V buffers must be contiguous along head_dim")

    num_rows, q_heads, head_dim = q.shape
    num_groups = decode_slots.shape[0]
    kv_heads = k_buffer.shape[1]
    if q_heads % kv_heads:
        raise ValueError(f"Q heads {q_heads} must be divisible by KV heads {kv_heads}")
    if num_rows != num_groups * beam_width:
        raise ValueError(
            f"Expected {num_groups} groups of {beam_width} beam rows, got {num_rows}"
        )
    if decode_slots.shape[1] % beam_width or decode_slots.stride(1) != 1:
        raise ValueError("Decode slots must be contiguous [groups, steps * beam_width]")
    if decode_lens.shape != (num_groups,):
        raise ValueError("Decode lengths must contain one value per beam group")
    if context_cu_seqlens.shape != (num_groups + 1,):
        raise ValueError("Context offsets must contain one boundary per beam group")

    beam_blocks = triton.cdiv(beam_width, _BLOCK_M)
    if num_splits is None:
        num_splits = num_kv_splits(
            num_groups * q_heads * beam_blocks,
            _num_sms(q.device.index),
        )
    output = torch.empty(
        (num_rows, q_heads, head_dim),
        dtype=q.dtype,
        device=q.device,
    )
    write_partials = num_splits > 1
    if write_partials:
        partial_output = torch.empty(
            (num_splits, num_rows, q_heads, head_dim),
            dtype=torch.float32,
            device=q.device,
        )
        partial_lse = torch.empty(
            (num_splits, num_rows, q_heads),
            dtype=torch.float32,
            device=q.device,
        )
    else:
        partial_output = partial_lse = output.new_empty((1, 1, 1))

    block_d = triton.next_power_of_2(head_dim)
    _shared_context_attention_kernel[(num_groups * num_splits, q_heads, beam_blocks)](
        q,
        k_buffer,
        v_buffer,
        context_slots,
        context_cu_seqlens,
        decode_slots,
        decode_lens,
        output,
        partial_output,
        partial_lse,
        softmax_scale,
        num_groups,
        num_splits,
        q.stride(0),
        q.stride(1),
        k_buffer.stride(0),
        k_buffer.stride(1),
        v_buffer.stride(0),
        v_buffer.stride(1),
        decode_slots.stride(0),
        output.stride(0),
        output.stride(1),
        partial_output.stride(0),
        partial_output.stride(1),
        partial_output.stride(2),
        partial_lse.stride(0),
        partial_lse.stride(1),
        BEAM_WIDTH=beam_width,
        Q_PER_KV=q_heads // kv_heads,
        HEAD_DIM=head_dim,
        WRITE_PARTIALS=write_partials,
        BLOCK_M=_BLOCK_M,
        BLOCK_N=_BLOCK_N,
        BLOCK_D=block_d,
        num_warps=_NUM_WARPS,
        num_stages=_NUM_STAGES,
    )
    if write_partials:
        _merge_splits_kernel[(num_rows,)](
            partial_output,
            partial_lse,
            output,
            num_splits,
            partial_output.stride(0),
            partial_output.stride(1),
            partial_output.stride(2),
            partial_lse.stride(0),
            partial_lse.stride(1),
            output.stride(0),
            output.stride(1),
            Q_HEADS=q_heads,
            HEAD_DIM=head_dim,
            BLOCK_H=triton.next_power_of_2(q_heads),
            BLOCK_D=block_d,
            num_warps=4,
        )
    return output
