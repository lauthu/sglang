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
"""Trie-constrained beam selection by radix select, with the beam KV remap folded in.

One call selects for every beam group of a step that shares a trie depth; the groups may
have different beam widths. For frontier row r and code c of a group the score is
cum[r] + ((logit - row_max) - row_lse), -inf where the trie forbids c: the torch path's
fp32 operations in the same order, so the selected scores are bit-identical. The result is
the group's best beam_width candidates by score, ties to the lower flat index
r * codebook_size + c; torch.topk leaves the order of ties unspecified.

The torch path's per-row top-k is mirrored where it changes the result:
- a forbidden ("dead") candidate takes its row's first valid code as its token;
- per row only the first min(beam_width, codebook_size) - valid dead candidates in code
  order compete;
- at depth 0 a beam width above codebook_size pads with -inf copies of the best token.

Each candidate gets a 32-bit key that orders like its score; key 0 never competes. The
selection is a radix select on 11-bit digits (tl.histogram), so no step has to fit one sort:
- a group of up to _SMALL_KEYS candidates is selected by one program, in registers;
- a larger group takes four launches over many programs: keys with the first digit's
  histogram; the second digit's histogram inside the chosen bucket; collection of the
  candidates above and inside the 22-bit threshold bucket; and a finalize that ranks them,
  writes the outputs and remaps the beams' decode KV slots.

Block shapes depend on the codebook size only, never on a beam width, so a server compiles
each kernel once per trie depth whatever widths its requests ask for.
"""

from __future__ import annotations

from typing import Optional, Sequence

import msgspec
import torch
import triton
import triton.language as tl

# Candidates one program holds in registers: 16 per thread at 8 warps.
_SMALL_KEYS = 4096
_TILE_KEYS = 4096
# 11-bit digits: tl.histogram of 4096 keys into 2048 bins takes ~0.3 us on RTX PRO 6000.
_BINS = 2048
# Threshold-bucket entries the finalize sorts in registers; a fuller bucket takes the
# slow path, which rescans the group's keys in one program.
_BUCKET_KEYS = 256
# Entries one finalize program ranks, and entries it compares them with per step; swept on
# RTX PRO 6000 from 256 to 2,000 beams, one and eight groups.
_RANK_SLICE = 32
_RANK_CHUNK = 512
# Padding entries the depth-0 kernel writes per step for a beam width above the codebook.
_PAD_BLOCK = 256
_INT64_MIN = tl.constexpr(-(2**63))
# Per-group int32 slots next to the histograms: counts of collected entries, programs done
# per pass, and the threshold digits the last program of a pass picks for the next one.
_META = tl.constexpr(8)
_SURE, _BUCKET = tl.constexpr(0), tl.constexpr(1)
_KEYS_DONE, _HIST2_DONE = tl.constexpr(2), tl.constexpr(3)
_DIGIT1, _ABOVE1, _PREFIX22, _ABOVE2 = (
    tl.constexpr(4),
    tl.constexpr(5),
    tl.constexpr(6),
    tl.constexpr(7),
)
# Per-group int64 fields of the descriptor table the host uploads once per call. A group's
# frontier row 0 is row LEADER of the leader tensors and row b > 0 is row TAIL + b - 1 of the
# tail tensors; its rows start at ROW in the per-row tensors and its beams at OUT in the outputs.
_DESC = tl.constexpr(6)
_LEADER, _TAIL, _ROW, _OUT, _WIDTH, _KV = (
    tl.constexpr(0),
    tl.constexpr(1),
    tl.constexpr(2),
    tl.constexpr(3),
    tl.constexpr(4),
    tl.constexpr(5),
)
# Many pointers are views whose 16-byte alignment changes from step to step (a group's slice
# of the previous step's outputs, the tail of a batch's logits), and the integers vary with
# the batch; specializing on either would compile while serving. Measured on RTX PRO 6000:
# aligned specializations of these kernels are not faster.
_POINTERS = [
    "leader_logits",
    "tail_logits",
    "leader_norm",
    "tail_norm",
    "desc",
    "cum",
    "prefix",
    "masks",
    "out_tokens",
    "out_parents",
    "out_scores",
    "out_prefix",
    "scratch",
    "fallback",
    "old_kv",
    "new_kv",
    "req_to_token",
    "kv_rows",
    "keys",
    "hist1",
    "hist2",
    "meta",
    "sure",
    "bucket",
    "spill",
]
_INTEGERS = ["s_ll", "s_tl", "s_ln", "s_tn", "s_rt", "C", "token_start"]


class RadixSelection(msgspec.Struct):
    """One step's selections. Group g owns [offsets[g], offsets[g + 1]) of the per-beam
    tensors, best candidate first, and [row_offsets[g], row_offsets[g + 1]) of old_kv."""

    tokens: torch.Tensor  # [sum of widths] int64
    parent_idx: torch.Tensor  # [sum of widths] int64, frontier row within the group
    cum_logprobs: torch.Tensor  # [sum of widths] float32
    # [sum of widths, depth + 1] int64; None on a final step.
    prefix_codes: Optional[torch.Tensor]
    # Decode slots before the remap [sum of rows, kv_len] and after it
    # [sum of widths, kv_len]; None on a step that does not remap.
    old_kv: Optional[torch.Tensor]
    new_kv: Optional[torch.Tensor]
    offsets: list[int]
    row_offsets: list[int]


@triton.jit
def _score_key(score):
    bits = score.to(tl.uint32, bitcast=True)
    return tl.where((bits >> 31) != 0, ~bits, bits | 0x80000000)


@triton.jit
def _key_score(key):
    bits = tl.where((key >> 31) != 0, key & 0x7FFFFFFF, ~key)
    return bits.to(tl.float32, bitcast=True)


@triton.jit
def _sort_key(key, flat):
    # Signed int64 that orders by key, then by the lower flat index.
    high = (key ^ 0x80000000).to(tl.int32, bitcast=True).to(tl.int64)
    return (high << 32) | (4294967295 - flat.to(tl.int64))


@triton.jit
def _unpack(sk, C):
    key = (sk >> 32).to(tl.int32).to(tl.uint32, bitcast=True) ^ 0x80000000
    flat = 4294967295 - (sk & 4294967295)
    return key, flat // C, flat % C


@triton.jit
def _pick(hist, need, BINS: tl.constexpr):
    """The largest bin b with count(>= b) >= need, and the count above b."""
    idx = tl.arange(0, BINS)
    at_least = tl.sum(hist, axis=0) - tl.cumsum(hist, axis=0) + hist
    bucket = tl.max(tl.where(at_least >= need, idx, -1), axis=0)
    above = tl.sum(tl.where(idx == bucket, at_least - hist, 0), axis=0)
    return bucket, above


@triton.jit
def _field(desc, g, field):
    return tl.load(desc + g * _DESC + field)


@triton.jit
def _num_rows(width, DEPTH: tl.constexpr):
    # The first selection reads the leader's row alone; later ones read one row per beam.
    if DEPTH == 0:
        rows = width * 0 + 1
    else:
        rows = width
    return rows


@triton.jit
def _row_base(lead, start, rows, leader, tail, s_lead, s_tail):
    return tl.where(
        rows == 0, leader + lead * s_lead, tail + (start + rows - 1) * s_tail
    )


@triton.jit
def _prefix_flat(row0, rows, live, C, prefix, DEPTH: tl.constexpr):
    flat = tl.zeros_like(rows)
    for i in tl.static_range(DEPTH):
        code = tl.load(prefix + (row0 + rows) * DEPTH + i, mask=live, other=0)
        flat = flat * C + code
    return flat


@triton.jit
def _tile_keys(
    desc,
    g,
    rows,
    codes,
    C,
    leader_logits,
    tail_logits,
    leader_norm,
    tail_norm,
    s_ll,
    s_tl,
    s_ln,
    s_tn,
    cum,
    prefix,
    masks,
    DEPTH: tl.constexpr,
):
    """Keys [rows, codes] (0 never competes), the in-bounds mask and each row's fallback."""
    lead, start = _field(desc, g, _LEADER), _field(desc, g, _TAIL)
    row0, width = _field(desc, g, _ROW), _field(desc, g, _WIDTH)
    in_r = rows < _num_rows(width, DEPTH)
    inb = in_r & (codes < C)
    lp = _row_base(lead, start, rows, leader_logits, tail_logits, s_ll, s_tl)
    logit = tl.load(lp + codes, mask=inb, other=0.0).to(tl.float32)
    npr = _row_base(lead, start, rows, leader_norm, tail_norm, s_ln, s_tn)
    row_max = tl.load(npr, mask=in_r, other=0.0).to(tl.float32)
    row_lse = tl.load(npr + 1, mask=in_r, other=0.0).to(tl.float32)
    cum_r = tl.load(cum + row0 + rows, mask=in_r, other=0.0)
    flat = _prefix_flat(row0, rows, in_r, C, prefix, DEPTH)
    valid = (tl.load(masks + flat * C + codes, mask=inb, other=0) != 0) & inb
    num_valid = tl.sum(valid.to(tl.int32), axis=1, keep_dims=True)
    dead = (inb & ~valid).to(tl.int32)
    # int32 keeps the comparison below on the tile's int32 counts.
    row_slots = tl.minimum(width, C).to(tl.int32)
    competes = (dead != 0) & ((tl.cumsum(dead, axis=1) - dead) < row_slots - num_valid)
    first = tl.min(tl.where(valid, codes, C), axis=1, keep_dims=True)
    fallback = tl.where(num_valid > 0, first, 0)
    score = cum_r + ((logit - row_max) - row_lse)
    dead_key = tl.where(competes, 0x007FFFFF, 0).to(tl.uint32)
    return tl.where(valid, _score_key(score), dead_key), inb, fallback


@triton.jit
def _copy_old_kv(
    desc, g, rows, live, old_kv, req_to_token, kv_rows, s_rt, L: tl.constexpr
):
    row0, base = _field(desc, g, _ROW), _field(desc, g, _KV)
    slot_row = tl.load(kv_rows + row0 + rows, mask=live, other=0)
    for i in tl.static_range(L):
        slot = tl.load(req_to_token + slot_row * s_rt + base + i, mask=live, other=0)
        tl.store(old_kv + (row0 + rows) * L + i, slot, mask=live)


@triton.jit
def _write(
    desc,
    g,
    pos,
    sk,
    live,
    C,
    token_start,
    prefix,
    masks,
    fallback,
    out_tokens,
    out_parents,
    out_scores,
    out_prefix,
    old_kv,
    new_kv,
    req_to_token,
    kv_rows,
    s_rt,
    DEPTH: tl.constexpr,
    FINAL: tl.constexpr,
    L: tl.constexpr,
):
    row0, out0 = _field(desc, g, _ROW), _field(desc, g, _OUT)
    key, row, code = _unpack(sk, C)
    # A finite score means an allowed code; only a -inf one can be a forbidden code,
    # which carries its row's fallback code.
    neg = live & (key == 0x007FFFFF)
    flat = _prefix_flat(row0, row, neg, C, prefix, DEPTH)
    forbidden = neg & (tl.load(masks + flat * C + code, mask=neg, other=1) == 0)
    code = tl.where(
        forbidden, tl.load(fallback + row0 + row, mask=forbidden, other=0), code
    )
    out = out0 + pos
    tl.store(out_tokens + out, code + token_start, mask=live)
    tl.store(out_parents + out, row, mask=live)
    tl.store(out_scores + out, _key_score(key), mask=live)
    if not FINAL:
        for i in tl.static_range(DEPTH):
            parent_code = tl.load(prefix + (row0 + row) * DEPTH + i, mask=live, other=0)
            tl.store(out_prefix + out * (DEPTH + 1) + i, parent_code, mask=live)
        tl.store(out_prefix + out * (DEPTH + 1) + DEPTH, code, mask=live)
    if L > 0:
        # A remapping step has one frontier row per beam, so beam `pos` owns row `pos`.
        base = _field(desc, g, _KV)
        slot_row = tl.load(kv_rows + row0 + pos, mask=live, other=0)
        for i in tl.static_range(L):
            slot = tl.load(old_kv + (row0 + row) * L + i, mask=live, other=0)
            tl.store(new_kv + out * L + i, slot, mask=live)
            tl.store(req_to_token + slot_row * s_rt + base + i, slot, mask=live)


@triton.jit(do_not_specialize=_INTEGERS, do_not_specialize_on_alignment=_POINTERS)
def _select_small_kernel(
    leader_logits,
    tail_logits,
    leader_norm,
    tail_norm,
    desc,
    s_ll,
    s_tl,
    s_ln,
    s_tn,
    cum,
    prefix,
    masks,
    out_tokens,
    out_parents,
    out_scores,
    out_prefix,
    scratch,
    fallback,
    old_kv,
    new_kv,
    req_to_token,
    kv_rows,
    s_rt,
    C,
    token_start,
    DEPTH: tl.constexpr,
    FINAL: tl.constexpr,
    L: tl.constexpr,
    RB: tl.constexpr,
    CB: tl.constexpr,
    SB: tl.constexpr,
    PB: tl.constexpr,
    BINS: tl.constexpr,
):
    g = tl.program_id(0).to(tl.int64)
    row0, out0, width = (
        _field(desc, g, _ROW),
        _field(desc, g, _OUT),
        _field(desc, g, _WIDTH),
    )
    num_rows = _num_rows(width, DEPTH)
    target = tl.minimum(width, num_rows * C).to(tl.int32)
    rows = tl.arange(0, RB).to(tl.int64)[:, None]
    codes = tl.arange(0, CB).to(tl.int64)[None, :]
    key, inb, fb = _tile_keys(
        desc,
        g,
        rows,
        codes,
        C,
        leader_logits,
        tail_logits,
        leader_norm,
        tail_norm,
        s_ll,
        s_tl,
        s_ln,
        s_tn,
        cum,
        prefix,
        masks,
        DEPTH,
    )
    rows1 = tl.arange(0, RB).to(tl.int64)
    tl.store(fallback + row0 + rows1, tl.reshape(fb, [RB]), mask=rows1 < num_rows)
    if L > 0:
        _copy_old_kv(
            desc, g, rows1, rows1 < num_rows, old_kv, req_to_token, kv_rows, s_rt, L
        )

    k = tl.reshape(key, [RB * CB])
    live = tl.reshape(inb, [RB * CB])
    flat = tl.reshape(rows * C + codes, [RB * CB])
    d1 = (k >> 21).to(tl.int32)
    b1, a1 = _pick(tl.histogram(d1, BINS, mask=live), target, BINS)
    in1 = live & (d1 == b1)
    d2 = ((k >> 10) & 0x7FF).to(tl.int32)
    b2, a2 = _pick(tl.histogram(d2, BINS, mask=in1), target - a1, BINS)
    in2 = in1 & (d2 == b2)
    d3 = (k & 0x3FF).to(tl.int32)
    b3, a3 = _pick(tl.histogram(d3, BINS, mask=in2), target - a1 - a2, BINS)
    thresh = (b1.to(tl.uint32) << 21) | (b2.to(tl.uint32) << 10) | b3.to(tl.uint32)
    tie = (live & (k == thresh)).to(tl.int32)
    chosen = live & (
        (k > thresh)
        | ((tie != 0) & (tl.cumsum(tie, axis=0) - tie < target - a1 - a2 - a3))
    )
    chosen_i = chosen.to(tl.int32)
    pos = tl.cumsum(chosen_i, axis=0) - chosen_i
    tl.store(scratch + g * SB + pos, _sort_key(k, flat), mask=chosen)
    tl.debug_barrier()

    j = tl.arange(0, SB)
    sk = tl.load(scratch + g * SB + j, mask=j < target, other=_INT64_MIN)
    sk = tl.sort(sk, descending=True)
    _write(
        desc,
        g,
        j,
        sk,
        j < target,
        C,
        token_start,
        prefix,
        masks,
        fallback,
        out_tokens,
        out_parents,
        out_scores,
        out_prefix,
        old_kv,
        new_kv,
        req_to_token,
        kv_rows,
        s_rt,
        DEPTH,
        FINAL,
        L,
    )
    if DEPTH == 0:
        if target < width:
            # The torch path pads a one-row frontier with -inf copies of its best token.
            tl.debug_barrier()
            best = tl.load(out_tokens + out0)
            for start in range(target, width, PB):
                p = start + tl.arange(0, PB)
                pad = p < width
                tl.store(
                    out_tokens + out0 + p, best + tl.zeros([PB], tl.int64), mask=pad
                )
                tl.store(out_parents + out0 + p, tl.zeros([PB], tl.int64), mask=pad)
                tl.store(
                    out_scores + out0 + p,
                    tl.full([PB], float("-inf"), tl.float32),
                    mask=pad,
                )
                if not FINAL:
                    code = best - token_start + tl.zeros([PB], tl.int64)
                    tl.store(out_prefix + out0 + p, code, mask=pad)


@triton.jit
def _last_program(meta, g, slot):
    """True in the group's last program to get here; it then sees every program's atomics."""
    tl.debug_barrier()
    ticket = tl.atomic_add(meta + g * _META + slot, 1, sem="acq_rel")
    last = ticket == tl.num_programs(1) - 1
    tl.debug_barrier()
    return last


@triton.jit(do_not_specialize=_INTEGERS, do_not_specialize_on_alignment=_POINTERS)
def _keys_kernel(
    leader_logits,
    tail_logits,
    leader_norm,
    tail_norm,
    desc,
    s_ll,
    s_tl,
    s_ln,
    s_tn,
    cum,
    prefix,
    masks,
    keys,
    fallback,
    hist1,
    meta,
    old_kv,
    req_to_token,
    kv_rows,
    s_rt,
    C,
    DEPTH: tl.constexpr,
    L: tl.constexpr,
    TR: tl.constexpr,
    CB: tl.constexpr,
    BINS: tl.constexpr,
):
    # Programs past a narrower group's rows only take their ticket.
    g = tl.program_id(0).to(tl.int64)
    row0, width = _field(desc, g, _ROW), _field(desc, g, _WIDTH)
    rows1 = tl.program_id(1).to(tl.int64) * TR + tl.arange(0, TR).to(tl.int64)
    rows = rows1[:, None]
    codes = tl.arange(0, CB).to(tl.int64)[None, :]
    key, inb, fb = _tile_keys(
        desc,
        g,
        rows,
        codes,
        C,
        leader_logits,
        tail_logits,
        leader_norm,
        tail_norm,
        s_ll,
        s_tl,
        s_ln,
        s_tn,
        cum,
        prefix,
        masks,
        DEPTH,
    )
    in_r = rows1 < width
    tl.store(keys + (row0 + rows) * C + codes, key.to(tl.int32, bitcast=True), mask=inb)
    tl.store(fallback + row0 + rows1, tl.reshape(fb, [TR]), mask=in_r)
    if L > 0:
        _copy_old_kv(desc, g, rows1, in_r, old_kv, req_to_token, kv_rows, s_rt, L)
    hist = tl.histogram(
        tl.reshape((key >> 21).to(tl.int32), [TR * CB]),
        BINS,
        mask=tl.reshape(inb, [TR * CB]),
    )
    bins = tl.arange(0, BINS)
    tl.atomic_add(hist1 + g * BINS + bins, hist, mask=hist > 0, sem="relaxed")
    if _last_program(meta, g, _KEYS_DONE):
        b1, above = _pick(
            tl.load(hist1 + g * BINS + bins, volatile=True), width.to(tl.int32), BINS
        )
        tl.store(meta + g * _META + _DIGIT1, b1)
        tl.store(meta + g * _META + _ABOVE1, above)


@triton.jit(do_not_specialize=_INTEGERS, do_not_specialize_on_alignment=_POINTERS)
def _hist2_kernel(keys, hist2, meta, desc, C, TILE: tl.constexpr, BINS: tl.constexpr):
    g = tl.program_id(0).to(tl.int64)
    row0, width = _field(desc, g, _ROW), _field(desc, g, _WIDTH)
    b1 = tl.load(meta + g * _META + _DIGIT1)
    idx = tl.program_id(1).to(tl.int64) * TILE + tl.arange(0, TILE)
    live = idx < width * C
    k = tl.load(keys + row0 * C + idx, mask=live, other=0).to(tl.uint32, bitcast=True)
    inb = live & ((k >> 21).to(tl.int32) == b1)
    hist = tl.histogram(((k >> 10) & 0x7FF).to(tl.int32), BINS, mask=inb)
    bins = tl.arange(0, BINS)
    tl.atomic_add(hist2 + g * BINS + bins, hist, mask=hist > 0, sem="relaxed")
    if _last_program(meta, g, _HIST2_DONE):
        above1 = tl.load(meta + g * _META + _ABOVE1)
        b2, above = _pick(
            tl.load(hist2 + g * BINS + bins, volatile=True),
            width.to(tl.int32) - above1,
            BINS,
        )
        tl.store(meta + g * _META + _PREFIX22, (b1 << 11) | b2)
        tl.store(meta + g * _META + _ABOVE2, above1 + above)


@triton.jit(do_not_specialize=_INTEGERS, do_not_specialize_on_alignment=_POINTERS)
def _collect_kernel(
    keys, meta, sure, bucket, desc, C, TILE: tl.constexpr, BK: tl.constexpr
):
    g = tl.program_id(0).to(tl.int64)
    row0, out0, width = (
        _field(desc, g, _ROW),
        _field(desc, g, _OUT),
        _field(desc, g, _WIDTH),
    )
    prefix22 = tl.load(meta + g * _META + _PREFIX22).to(tl.uint32)
    idx = tl.program_id(1).to(tl.int64) * TILE + tl.arange(0, TILE)
    live = idx < width * C
    k = tl.load(keys + row0 * C + idx, mask=live, other=0).to(tl.uint32, bitcast=True)
    sk = _sort_key(k, idx)
    above = (live & ((k >> 10) > prefix22)).to(tl.int32)
    n_above = tl.sum(above, axis=0)
    if n_above > 0:
        base = tl.atomic_add(meta + g * _META + _SURE, n_above, sem="relaxed")
        tl.store(
            sure + out0 + base + tl.cumsum(above, axis=0) - above, sk, mask=above != 0
        )
    inside = (live & ((k >> 10) == prefix22)).to(tl.int32)
    n_inside = tl.sum(inside, axis=0)
    if n_inside > 0:
        base = tl.atomic_add(meta + g * _META + _BUCKET, n_inside, sem="relaxed")
        pos = base + tl.cumsum(inside, axis=0) - inside
        tl.store(bucket + g * BK + pos, sk, mask=(inside != 0) & (pos < BK))


@triton.jit
def _rank_and_write(
    desc,
    g,
    start,
    n,
    src,
    out_base,
    C,
    token_start,
    prefix,
    masks,
    fallback,
    out_tokens,
    out_parents,
    out_scores,
    out_prefix,
    old_kv,
    new_kv,
    req_to_token,
    kv_rows,
    s_rt,
    DEPTH: tl.constexpr,
    FINAL: tl.constexpr,
    L: tl.constexpr,
    SL: tl.constexpr,
    CH: tl.constexpr,
):
    """Writes src[start : start + SL] at out_base + its rank among src[:n] (keys unique)."""
    i = start + tl.arange(0, SL)
    live = i < n
    ski = tl.load(src + i, mask=live, other=_INT64_MIN)
    rank = tl.zeros([SL], tl.int32)
    for c0 in range(0, n, CH):
        j = c0 + tl.arange(0, CH)
        skj = tl.load(src + j, mask=j < n, other=_INT64_MIN)
        rank += tl.sum((skj[None, :] > ski[:, None]).to(tl.int32), axis=1)
    _write(
        desc,
        g,
        out_base + rank,
        ski,
        live,
        C,
        token_start,
        prefix,
        masks,
        fallback,
        out_tokens,
        out_parents,
        out_scores,
        out_prefix,
        old_kv,
        new_kv,
        req_to_token,
        kv_rows,
        s_rt,
        DEPTH,
        FINAL,
        L,
    )


@triton.jit(do_not_specialize=_INTEGERS, do_not_specialize_on_alignment=_POINTERS)
def _finalize_kernel(
    keys,
    meta,
    sure,
    bucket,
    spill,
    desc,
    C,
    token_start,
    prefix,
    masks,
    fallback,
    out_tokens,
    out_parents,
    out_scores,
    out_prefix,
    old_kv,
    new_kv,
    req_to_token,
    kv_rows,
    s_rt,
    DEPTH: tl.constexpr,
    FINAL: tl.constexpr,
    L: tl.constexpr,
    TILE: tl.constexpr,
    BINS: tl.constexpr,
    BK: tl.constexpr,
    SL: tl.constexpr,
    CH: tl.constexpr,
):
    # Programs [0, n) rank slices of the entries above the threshold bucket, which take
    # the first output positions; the last program takes the bucket's winners after them.
    g = tl.program_id(0).to(tl.int64)
    out0, width = _field(desc, g, _OUT), _field(desc, g, _WIDTH)
    pid = tl.program_id(1)
    n_sure = tl.load(meta + g * _META + _SURE)
    if pid < tl.num_programs(1) - 1:
        if pid * SL < n_sure:
            _rank_and_write(
                desc,
                g,
                pid * SL,
                n_sure,
                sure + out0,
                0,
                C,
                token_start,
                prefix,
                masks,
                fallback,
                out_tokens,
                out_parents,
                out_scores,
                out_prefix,
                old_kv,
                new_kv,
                req_to_token,
                kv_rows,
                s_rt,
                DEPTH,
                FINAL,
                L,
                SL,
                CH,
            )
    else:
        need = width.to(tl.int32) - n_sure
        n_bucket = tl.load(meta + g * _META + _BUCKET)
        if n_bucket <= BK:
            j = tl.arange(0, BK)
            sk = tl.load(bucket + g * BK + j, mask=j < n_bucket, other=_INT64_MIN)
            sk = tl.sort(sk, descending=True)
            _write(
                desc,
                g,
                n_sure + j,
                sk,
                j < need,
                C,
                token_start,
                prefix,
                masks,
                fallback,
                out_tokens,
                out_parents,
                out_scores,
                out_prefix,
                old_kv,
                new_kv,
                req_to_token,
                kv_rows,
                s_rt,
                DEPTH,
                FINAL,
                L,
            )
        else:
            _bucket_slow_path(
                desc,
                g,
                keys,
                meta,
                spill,
                n_sure,
                need,
                C,
                token_start,
                prefix,
                masks,
                fallback,
                out_tokens,
                out_parents,
                out_scores,
                out_prefix,
                old_kv,
                new_kv,
                req_to_token,
                kv_rows,
                s_rt,
                DEPTH,
                FINAL,
                L,
                TILE,
                BINS,
                SL,
                CH,
            )


@triton.jit
def _bucket_slow_path(
    desc,
    g,
    keys,
    meta,
    spill,
    n_sure,
    need,
    C,
    token_start,
    prefix,
    masks,
    fallback,
    out_tokens,
    out_parents,
    out_scores,
    out_prefix,
    old_kv,
    new_kv,
    req_to_token,
    kv_rows,
    s_rt,
    DEPTH: tl.constexpr,
    FINAL: tl.constexpr,
    L: tl.constexpr,
    TILE: tl.constexpr,
    BINS: tl.constexpr,
    SL: tl.constexpr,
    CH: tl.constexpr,
):
    # The bucket overflowed (ties at the threshold, or many dead candidates): one program
    # rescans the group's keys for the last digit, then takes ties in flat-index order.
    row0, out0, width = (
        _field(desc, g, _ROW),
        _field(desc, g, _OUT),
        _field(desc, g, _WIDTH),
    )
    total = width * C
    prefix22 = tl.load(meta + g * _META + _PREFIX22).to(tl.uint32)
    hist = tl.zeros([BINS], tl.int32)
    for start in range(0, total, TILE):
        idx = start + tl.arange(0, TILE)
        live = idx < total
        k = tl.load(keys + row0 * C + idx, mask=live, other=0).to(
            tl.uint32, bitcast=True
        )
        hist += tl.histogram(
            (k & 0x3FF).to(tl.int32), BINS, mask=live & ((k >> 10) == prefix22)
        )
    b3, n_above = _pick(hist, need, BINS)
    thresh = (prefix22 << 10) | b3.to(tl.uint32)
    n_gt = 0
    n_tie = 0
    for start in range(0, total, TILE):
        idx = start + tl.arange(0, TILE)
        live = idx < total
        k = tl.load(keys + row0 * C + idx, mask=live, other=0).to(
            tl.uint32, bitcast=True
        )
        sk = _sort_key(k, idx)
        gt = (live & ((k >> 10) == prefix22) & (k > thresh)).to(tl.int32)
        tl.store(spill + out0 + n_gt + tl.cumsum(gt, axis=0) - gt, sk, mask=gt != 0)
        tie = (live & (k == thresh)).to(tl.int32)
        tie_rank = n_tie + tl.cumsum(tie, axis=0) - tie
        _write(
            desc,
            g,
            n_sure + n_above + tie_rank,
            sk,
            (tie != 0) & (tie_rank < need - n_above),
            C,
            token_start,
            prefix,
            masks,
            fallback,
            out_tokens,
            out_parents,
            out_scores,
            out_prefix,
            old_kv,
            new_kv,
            req_to_token,
            kv_rows,
            s_rt,
            DEPTH,
            FINAL,
            L,
        )
        n_gt += tl.sum(gt, axis=0)
        n_tie += tl.sum(tie, axis=0)
    tl.debug_barrier()
    for start in range(0, n_above, SL):
        _rank_and_write(
            desc,
            g,
            start,
            n_above,
            spill + out0,
            n_sure,
            C,
            token_start,
            prefix,
            masks,
            fallback,
            out_tokens,
            out_parents,
            out_scores,
            out_prefix,
            old_kv,
            new_kv,
            req_to_token,
            kv_rows,
            s_rt,
            DEPTH,
            FINAL,
            L,
            SL,
            CH,
        )


def _upload(values: Sequence[int], device) -> torch.Tensor:
    if torch.device(device).type == "cuda":
        return torch.tensor(values, dtype=torch.int64, pin_memory=True).to(
            device, non_blocking=True
        )
    return torch.tensor(values, dtype=torch.int64, device=device)


def _prefix_sums(counts: Sequence[int]) -> list[int]:
    sums = [0]
    for count in counts:
        sums.append(sums[-1] + count)
    return sums


def _declared(kernel, args: dict) -> dict:
    """The entries of `args` that `kernel` takes."""
    return {name: args[name] for name in kernel.arg_names if name in args}


def _launch_small(args: dict, desc: torch.Tensor, *, constants: dict, cb: int) -> None:
    """One program per group: the groups whose candidates fit one program's registers."""
    small_rows = max(1, _SMALL_KEYS // cb)
    # One sort covers a first step's codebook and a later step's beams.
    sb = max(cb, small_rows)
    num_groups = desc.shape[0]
    scratch = torch.empty((num_groups, sb), dtype=torch.int64, device=desc.device)
    _select_small_kernel[(num_groups,)](
        **args,
        desc=desc,
        scratch=scratch,
        **constants,
        RB=small_rows,
        CB=cb,
        SB=sb,
        PB=_PAD_BLOCK,
        BINS=_BINS,
        num_warps=8,
    )


def _launch_large(
    args: dict,
    desc: torch.Tensor,
    *,
    constants: dict,
    cb: int,
    widest: int,
    total_rows: int,
    total_width: int,
) -> None:
    """Four launches over many programs per group; `widest` sizes the grids, and a
    narrower group's spare programs return at once."""
    num_groups, device = desc.shape[0], desc.device
    size = args["C"]
    work = torch.zeros(
        num_groups * (2 * _BINS + _META.value), dtype=torch.int32, device=device
    )
    hist1 = work[: num_groups * _BINS]
    hist2 = work[num_groups * _BINS : 2 * num_groups * _BINS]
    meta = work[2 * num_groups * _BINS :]
    keys = torch.empty(total_rows * size, dtype=torch.int32, device=device)
    sure = torch.empty(total_width, dtype=torch.int64, device=device)
    bucket = torch.empty((num_groups, _BUCKET_KEYS), dtype=torch.int64, device=device)
    spill = torch.empty(total_width, dtype=torch.int64, device=device)
    tile_rows = max(1, _TILE_KEYS // cb)
    tiles = triton.cdiv(widest * size, _TILE_KEYS)
    work_args = dict(
        args,
        **constants,
        desc=desc,
        keys=keys,
        hist1=hist1,
        meta=meta,
        sure=sure,
        bucket=bucket,
        spill=spill,
    )
    _keys_kernel[(num_groups, triton.cdiv(widest, tile_rows))](
        **_declared(_keys_kernel, work_args),
        TR=tile_rows,
        CB=cb,
        BINS=_BINS,
        num_warps=8,
    )
    _hist2_kernel[(num_groups, tiles)](
        keys, hist2, meta, desc, size, TILE=_TILE_KEYS, BINS=_BINS, num_warps=8
    )
    _collect_kernel[(num_groups, tiles)](
        keys,
        meta,
        sure,
        bucket,
        desc,
        size,
        TILE=_TILE_KEYS,
        BK=_BUCKET_KEYS,
        num_warps=8,
    )
    _finalize_kernel[(num_groups, triton.cdiv(widest, _RANK_SLICE) + 1)](
        **_declared(_finalize_kernel, work_args),
        TILE=_TILE_KEYS,
        BINS=_BINS,
        BK=_BUCKET_KEYS,
        SL=_RANK_SLICE,
        CH=_RANK_CHUNK,
        num_warps=8,
    )


def _descriptors(
    order: Sequence[int],
    out: RadixSelection,
    *,
    leader_rows: Sequence[int],
    tail_starts: Sequence[int],
    kv_starts: Optional[Sequence[int]],
) -> torch.Tensor:
    """The descriptor table, one upload: the groups' rows in `order`."""
    fields = []
    for g in order:
        fields += [
            leader_rows[g],
            tail_starts[g],
            out.row_offsets[g],
            out.offsets[g],
            out.offsets[g + 1] - out.offsets[g],
            kv_starts[g] if kv_starts is not None else 0,
        ]
    return _upload(fields, out.tokens.device).view(len(order), _DESC.value)


def _allocate(
    beam_widths: Sequence[int],
    *,
    depth: int,
    final: bool,
    remap: int,
    slot_dtype: Optional[torch.dtype],
    device,
) -> RadixSelection:
    """The step's outputs, sized for every group back to back."""
    offsets = _prefix_sums(beam_widths)
    row_offsets = _prefix_sums([1 if depth == 0 else width for width in beam_widths])
    total_width, total_rows = offsets[-1], row_offsets[-1]
    tokens = torch.empty(total_width, dtype=torch.int64, device=device)
    prefix_codes = old_kv = new_kv = None
    if not final:
        prefix_codes = torch.empty(
            (total_width, depth + 1), dtype=torch.int64, device=device
        )
    if remap:
        old_kv = torch.empty((total_rows, remap), dtype=slot_dtype, device=device)
        new_kv = torch.empty((total_width, remap), dtype=slot_dtype, device=device)
    return RadixSelection(
        tokens=tokens,
        parent_idx=torch.empty_like(tokens),
        cum_logprobs=torch.empty(total_width, dtype=torch.float32, device=device),
        prefix_codes=prefix_codes,
        old_kv=old_kv,
        new_kv=new_kv,
        offsets=offsets,
        row_offsets=row_offsets,
    )


def radix_select_trie_beams(
    *,
    leader_logits: torch.Tensor,
    leader_normalizer: torch.Tensor,
    leader_rows: Sequence[int],
    tail_logits: Optional[torch.Tensor],
    tail_normalizer: Optional[torch.Tensor],
    tail_starts: Optional[Sequence[int]],
    beam_widths: Sequence[int],
    cum_logprobs: torch.Tensor,
    prefix_codes: torch.Tensor,
    child_masks: torch.Tensor,
    codebook_size: int,
    token_start: int,
    final: bool,
    req_to_token: Optional[torch.Tensor] = None,
    kv_rows: Optional[torch.Tensor] = None,
    kv_starts: Optional[Sequence[int]] = None,
    kv_len: int = 0,
) -> RadixSelection:
    """One selection step for groups of one trie depth, as the module docstring describes.

    Covers any beam widths, mixed freely, on CUDA; codebook_size <= 4096; child masks as a
    dense table. It rejects (asserts) a larger codebook and a table of another shape.

    leader_logits/tail_logits: logits [rows, >= codebook_size], unit stride along the codes;
        group g's frontier row 0 is leader row leader_rows[g], and row b > 0 is tail row
        tail_starts[g] + b - 1. tail_* are None at depth 0, where a group has one row.
    leader_normalizer/tail_normalizer: [rows, 2] (row max, log-sum-exp after the max).
    beam_widths: each group's beam width, in group order.
    cum_logprobs: [rows of all groups] float32 frontier scores, groups back to back;
        prefix_codes: [rows of all groups, depth] int64.
    child_masks: [codebook_size**depth, codebook_size] bool valid children of a prefix,
        indexed by its codes read as a base-codebook_size number.
    req_to_token/kv_rows/kv_starts/kv_len: with kv_len > 0 on a non-final step, each beam's
        decode slots req_to_token[kv_rows[row], kv_starts[g] : kv_starts[g] + kv_len] become
        its parent's, as remap_kv_mapping does.
    """
    depth, size = prefix_codes.shape[1], codebook_size
    assert child_masks.shape == (size**depth, size), (
        f"depth {depth} needs a dense child-mask table [{size**depth}, {size}], "
        f"got {tuple(child_masks.shape)}"
    )
    assert size <= _TILE_KEYS and cum_logprobs.dtype == torch.float32
    assert leader_logits.stride(1) == 1 and leader_normalizer.stride(1) == 1
    if tail_logits is None:
        assert depth == 0, "frontier rows past the leader need the tail logits"
        tail_logits, tail_normalizer = leader_logits, leader_normalizer
        tail_starts = leader_rows
    remap = kv_len if req_to_token is not None and not final else 0
    cb = triton.next_power_of_2(size)
    small_rows = max(1, _SMALL_KEYS // cb)
    num_groups = len(beam_widths)
    small = [g for g in range(num_groups) if depth == 0 or beam_widths[g] <= small_rows]
    large = [g for g in range(num_groups) if depth > 0 and beam_widths[g] > small_rows]
    out = _allocate(
        beam_widths,
        depth=depth,
        final=final,
        remap=remap,
        slot_dtype=req_to_token.dtype if remap else None,
        device=cum_logprobs.device,
    )
    assert cum_logprobs.shape == (out.row_offsets[-1],)
    assert prefix_codes.shape[0] == out.row_offsets[-1]
    desc = _descriptors(
        small + large,
        out,
        leader_rows=leader_rows,
        tail_starts=tail_starts,
        kv_starts=kv_starts if remap else None,
    )
    # Unused pointer arguments still need a device tensor; an empty one may have none.
    unused = out.tokens
    args = dict(
        leader_logits=leader_logits,
        tail_logits=tail_logits,
        leader_norm=leader_normalizer,
        tail_norm=tail_normalizer,
        s_ll=leader_logits.stride(0),
        s_tl=tail_logits.stride(0),
        s_ln=leader_normalizer.stride(0),
        s_tn=tail_normalizer.stride(0),
        cum=cum_logprobs,
        prefix=prefix_codes if depth else unused,
        masks=child_masks.view(torch.uint8),
        fallback=torch.empty_like(out.tokens[: out.row_offsets[-1]]),
        out_tokens=out.tokens,
        out_parents=out.parent_idx,
        out_scores=out.cum_logprobs,
        out_prefix=unused if final else out.prefix_codes,
        old_kv=out.old_kv if remap else unused,
        new_kv=out.new_kv if remap else unused,
        req_to_token=req_to_token if remap else unused,
        kv_rows=kv_rows if remap else unused,
        s_rt=req_to_token.stride(0) if remap else 0,
        C=size,
        token_start=token_start,
    )
    constants = dict(DEPTH=depth, FINAL=final, L=remap)
    if small:
        _launch_small(args, desc[: len(small)], constants=constants, cb=cb)
    if large:
        _launch_large(
            args,
            desc[len(small) :],
            constants=constants,
            cb=cb,
            widest=max(beam_widths[g] for g in large),
            total_rows=out.row_offsets[-1],
            total_width=out.offsets[-1],
        )
    return out
