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
"""Fused trie-constrained beam selection: one batched step for many groups.

Replaces trie_constraint.topk_logprobs followed by joint_select (or
select_final_topk) and advance_prefixes. For every group it scores each
(frontier row, code) candidate as cum + ((logit - row_max) - row_lse), -inf
where the trie forbids the code: the torch path's fp32 operations in the same
order, so the selected scores are bit-identical. The result is the best
beam_width candidates by score, ties broken by the lower flat index
row * codebook_size + code; torch.topk leaves the order of ties unspecified.

The torch path's per-row top-k is mirrored exactly where it matters:
- A candidate whose masked logit is -inf ("dead") takes its row's first valid
  code as its token, so a later step never indexes an empty prefix.
- Per row, only the first min(beam_width, codebook_size) - finite dead
  candidates in code order compete; the per-row top-k keeps exactly those.
No trie token can be a stop token (admission rejects them), so every selected
candidate survives and joint_select's stop handling has nothing to do.

Two launches per step. Each band program scores a band of frontier rows and
keeps its best beam_width sort keys; one merge program per group selects from
its bands and writes tokens, parents, scores and the next prefix codes.
"""

from __future__ import annotations

from typing import Optional

import msgspec
import torch
import triton
import triton.language as tl

# Keys one band or merge program sorts in registers. Measured on RTX PRO 6000:
# 4096 int64 keys take ~60 us, and larger sorts spill.
_SORT_KEYS = 4096
_INT64_MAX = tl.constexpr(0x7FFFFFFFFFFFFFFF)


class TrieSelection(msgspec.Struct):
    """One fused step's selections, [num_groups, beam_width] each."""

    tokens: torch.Tensor  # int64 token ids
    parent_idx: torch.Tensor  # int64 frontier row of each selection
    cum_logprobs: torch.Tensor  # float32
    prefix_codes: Optional[torch.Tensor]  # [.., depth + 1] int64; None on a final step


class _BandLayout(msgspec.Struct, frozen=True):
    code_block: int
    beam_block: int
    band_rows: int
    num_bands: int

    @classmethod
    def of(cls, *, beam_width: int, num_rows: int, codebook_size: int) -> _BandLayout:
        code_block = triton.next_power_of_2(codebook_size)
        beam_block = triton.next_power_of_2(beam_width)
        # Enough rows for beam_width keys, up to _SORT_KEYS keys.
        band_rows = max(
            triton.cdiv(beam_block, code_block),
            min(_SORT_KEYS // code_block, triton.next_power_of_2(num_rows)),
        )
        return cls(
            code_block=code_block,
            beam_block=beam_block,
            band_rows=band_rows,
            num_bands=triton.cdiv(num_rows, band_rows),
        )

    @property
    def merge_keys(self) -> int:
        return triton.next_power_of_2(self.num_bands * self.beam_block)


def fused_select_supported(
    *, beam_width: int, num_rows: int, codebook_size: int
) -> bool:
    """Shapes whose sorts fit one program; fewer candidates than beam_width
    need the torch path's -inf padding."""
    if beam_width > num_rows * codebook_size:
        return False
    layout = _BandLayout.of(
        beam_width=beam_width, num_rows=num_rows, codebook_size=codebook_size
    )
    return (
        layout.band_rows * layout.code_block <= _SORT_KEYS
        and layout.merge_keys <= _SORT_KEYS
    )


@triton.jit
def _score_band_kernel(
    leader_logits,
    tail_logits,
    leader_normalizer,
    tail_normalizer,
    frontier_rows,
    cum_logprobs,
    prefix_codes,
    child_masks,
    band_keys,
    fallback_codes,
    num_rows,
    codebook_size,
    code_offset,
    max_row_candidates,
    leader_stride,
    tail_stride,
    leader_normalizer_stride,
    tail_normalizer_stride,
    DEPTH: tl.constexpr,
    HAS_ROW_MAX: tl.constexpr,
    BAND_ROWS: tl.constexpr,
    CODE_BLOCK: tl.constexpr,
    BEAM_BLOCK: tl.constexpr,
):
    group = tl.program_id(0)
    band = tl.program_id(1)
    num_bands = tl.num_programs(1)
    rows = band * BAND_ROWS + tl.arange(0, BAND_ROWS)
    codes = tl.arange(0, CODE_BLOCK)
    row_ok = rows < num_rows
    ok = row_ok[:, None] & (codes < codebook_size)[None, :]

    # Frontier row 0 is the leader's logits row; rows 1.. are consecutive
    # member rows of the tail.
    leader_row = tl.load(frontier_rows + group * 2).to(tl.int64)
    member_rows = tl.load(frontier_rows + group * 2 + 1).to(tl.int64) + rows - 1
    is_leader = rows == 0
    logit_rows = tl.where(
        is_leader[:, None],
        leader_logits + leader_row * leader_stride,
        tail_logits + member_rows[:, None] * tail_stride,
    )
    logits = tl.load(
        logit_rows + code_offset + codes[None, :], mask=ok, other=float("-inf")
    ).to(tl.float32)
    normalizer_rows = tl.where(
        is_leader,
        leader_normalizer + leader_row * leader_normalizer_stride,
        tail_normalizer + member_rows * tail_normalizer_stride,
    )

    state_rows = group * num_rows + rows
    if DEPTH == 0:
        mask_rows = tl.zeros([BAND_ROWS], dtype=tl.int64)
    elif DEPTH == 1:
        mask_rows = tl.load(prefix_codes + state_rows * DEPTH, mask=row_ok, other=0)
    else:
        mask_rows = tl.load(
            prefix_codes + state_rows * DEPTH, mask=row_ok, other=0
        ) * codebook_size + tl.load(
            prefix_codes + state_rows * DEPTH + 1, mask=row_ok, other=0
        )
    allowed = ok & (
        tl.load(
            child_masks + mask_rows[:, None] * codebook_size + codes[None, :],
            mask=ok,
            other=0,
        )
        != 0
    )

    masked = tl.where(allowed, logits, float("-inf"))
    if HAS_ROW_MAX:
        row_max = tl.load(normalizer_rows, mask=row_ok, other=0.0)
        row_lse = tl.load(normalizer_rows + 1, mask=row_ok, other=0.0)
        logprobs = (masked - row_max[:, None]) - row_lse[:, None]
    else:
        row_lse = tl.load(normalizer_rows, mask=row_ok, other=0.0)
        logprobs = masked - row_lse[:, None]
    cum = tl.load(cum_logprobs + state_rows, mask=row_ok, other=0.0)
    scores = cum[:, None] + logprobs

    dead = ok & (masked == float("-inf"))
    num_finite = tl.sum((ok & ~dead).to(tl.int32), axis=1)
    dead_rank = tl.cumsum(dead.to(tl.int32), axis=1) - 1
    competes = ok & (~dead | (dead_rank < max_row_candidates - num_finite[:, None]))

    first_valid = tl.min(tl.where(allowed, codes[None, :], CODE_BLOCK), axis=1)
    tl.store(
        fallback_codes + state_rows,
        tl.where(first_valid == CODE_BLOCK, 0, first_valid).to(tl.int64),
        mask=row_ok,
    )

    # Ascending keys order candidates by score descending, then flat index;
    # the low bit carries the dead flag past the sort.
    bits = scores.to(tl.int32, bitcast=True)
    ordered = tl.where(bits < 0, bits ^ 0x7FFFFFFF, bits)
    flat = rows[:, None].to(tl.int64) * codebook_size + codes[None, :]
    keys = ((~ordered).to(tl.int64) << 32) | (flat << 1) | dead.to(tl.int64)
    keys = tl.where(competes, keys, _INT64_MAX)
    keys = tl.sort(tl.reshape(keys, [BAND_ROWS * CODE_BLOCK]))
    slots = tl.arange(0, BAND_ROWS * CODE_BLOCK)
    tl.store(
        band_keys + (group * num_bands + band) * BEAM_BLOCK + slots,
        keys,
        mask=slots < BEAM_BLOCK,
    )


@triton.jit
def _merge_bands_kernel(
    band_keys,
    fallback_codes,
    prefix_codes,
    tokens,
    parent_idx,
    cum_logprobs,
    next_prefix_codes,
    num_band_keys,
    num_rows,
    beam_width,
    codebook_size,
    token_start,
    DEPTH: tl.constexpr,
    EMIT_PREFIX: tl.constexpr,
    MERGE_BLOCK: tl.constexpr,
    BEAM_BLOCK: tl.constexpr,
):
    group = tl.program_id(0)
    slots = tl.arange(0, MERGE_BLOCK)
    keys = tl.load(
        band_keys + group * num_band_keys + slots,
        mask=slots < num_band_keys,
        other=_INT64_MAX,
    )
    keys = tl.reshape(tl.sort(keys), [MERGE_BLOCK // BEAM_BLOCK, BEAM_BLOCK])
    best = tl.sum(
        tl.where(tl.arange(0, MERGE_BLOCK // BEAM_BLOCK)[:, None] == 0, keys, 0),
        axis=0,
    )

    beams = tl.arange(0, BEAM_BLOCK)
    beam_ok = beams < beam_width
    flat = (best & 0xFFFFFFFF) >> 1
    dead = (best & 1) != 0
    parents = flat // codebook_size
    ordered = ~((best >> 32).to(tl.int32))
    scores = tl.where(ordered < 0, ordered ^ 0x7FFFFFFF, ordered).to(
        tl.float32, bitcast=True
    )
    fallback = tl.load(
        fallback_codes + group * num_rows + parents, mask=beam_ok, other=0
    )
    codes = tl.where(dead, fallback, flat % codebook_size)

    out = group * beam_width + beams
    tl.store(tokens + out, codes + token_start, mask=beam_ok)
    tl.store(parent_idx + out, parents, mask=beam_ok)
    tl.store(cum_logprobs + out, scores, mask=beam_ok)
    if EMIT_PREFIX:
        for d in tl.static_range(DEPTH):
            tl.store(
                next_prefix_codes + out * (DEPTH + 1) + d,
                tl.load(
                    prefix_codes + (group * num_rows + parents) * DEPTH + d,
                    mask=beam_ok,
                    other=0,
                ),
                mask=beam_ok,
            )
        tl.store(next_prefix_codes + out * (DEPTH + 1) + DEPTH, codes, mask=beam_ok)


def select_trie_beams(
    *,
    leader_logits: torch.Tensor,
    tail_logits: torch.Tensor,
    leader_normalizer: torch.Tensor,
    tail_normalizer: torch.Tensor,
    frontier_rows: torch.Tensor,
    cum_logprobs: torch.Tensor,
    prefix_codes: torch.Tensor,
    child_masks: torch.Tensor,
    codebook_size: int,
    code_offset: int,
    token_start: int,
    beam_width: int,
    final: bool,
) -> TrieSelection:
    """One selection step for every group, as described in the module docstring.

    leader_logits / tail_logits: [rows, >= code_offset + codebook_size] logits;
        group g's frontier row 0 is leader row frontier_rows[g, 0], and row
        b > 0 is tail row frontier_rows[g, 1] + b - 1.
    leader_normalizer / tail_normalizer: float32 per logits row, [rows, 2] as
        (row max, log-sum-exp after the max) or [rows, 1] as a log-sum-exp.
    frontier_rows: [num_groups, 2] integer, on the logits' device.
    cum_logprobs: [num_groups, num_rows] float32 frontier scores.
    prefix_codes: [num_groups, num_rows, depth] int64 codes chosen so far.
    child_masks: [num_prefixes, codebook_size] bool valid children, indexed by
        the prefix's flat code (one row at depth 0).
    """
    num_groups, num_rows, depth = prefix_codes.shape
    assert fused_select_supported(
        beam_width=beam_width, num_rows=num_rows, codebook_size=codebook_size
    )
    assert leader_normalizer.dtype == torch.float32
    assert tail_normalizer.dtype == torch.float32
    device = cum_logprobs.device
    # Depth 0 has no prefix to read, and an empty tensor has no device pointer.
    prefix_arg = prefix_codes if depth > 0 else frontier_rows
    layout = _BandLayout.of(
        beam_width=beam_width, num_rows=num_rows, codebook_size=codebook_size
    )
    code_block, beam_block = layout.code_block, layout.beam_block
    band_rows, num_bands = layout.band_rows, layout.num_bands
    band_keys = torch.empty(
        (num_groups, num_bands, beam_block), dtype=torch.int64, device=device
    )
    fallback_codes = torch.empty(
        (num_groups, num_rows), dtype=torch.int64, device=device
    )
    _score_band_kernel[(num_groups, num_bands)](
        leader_logits,
        tail_logits,
        leader_normalizer,
        tail_normalizer,
        frontier_rows,
        cum_logprobs,
        prefix_arg,
        child_masks.view(torch.uint8),
        band_keys,
        fallback_codes,
        num_rows,
        codebook_size,
        code_offset,
        min(beam_width, codebook_size),
        leader_logits.stride(0),
        tail_logits.stride(0),
        leader_normalizer.stride(0),
        tail_normalizer.stride(0),
        DEPTH=depth,
        HAS_ROW_MAX=leader_normalizer.shape[1] == 2,
        BAND_ROWS=band_rows,
        CODE_BLOCK=code_block,
        BEAM_BLOCK=beam_block,
        num_warps=8,
    )

    tokens = torch.empty((num_groups, beam_width), dtype=torch.int64, device=device)
    parent_idx = torch.empty_like(tokens)
    scores = torch.empty((num_groups, beam_width), dtype=torch.float32, device=device)
    next_prefix_codes = (
        None
        if final
        else torch.empty(
            (num_groups, beam_width, depth + 1), dtype=torch.int64, device=device
        )
    )
    _merge_bands_kernel[(num_groups,)](
        band_keys,
        fallback_codes,
        prefix_arg,
        tokens,
        parent_idx,
        scores,
        tokens if next_prefix_codes is None else next_prefix_codes,
        num_bands * beam_block,
        num_rows,
        beam_width,
        codebook_size,
        token_start,
        DEPTH=depth,
        EMIT_PREFIX=next_prefix_codes is not None,
        MERGE_BLOCK=layout.merge_keys,
        BEAM_BLOCK=beam_block,
        num_warps=8,
    )
    return TrieSelection(
        tokens=tokens,
        parent_idx=parent_idx,
        cum_logprobs=scores,
        prefix_codes=next_prefix_codes,
    )
