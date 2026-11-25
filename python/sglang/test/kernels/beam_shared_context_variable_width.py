"""A decode batch for the variable-width shared-context attention kernels, with its
FP32 reference: per row, plain softmax attention over the row's own KV slots."""

import math
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention import beam_shared_context_variable_width_stream_k
from sglang.srt.beam_search import shared_context_attention
from sglang.srt.beam_search.batch_tail import BeamTailEntry
from sglang.srt.beam_search.shared_context_variable_width import (
    StreamKGrid,
    build_variable_width_metadata,
)

# Split costs that force the stream-K plan's choice, whatever the batch: whole
# tasks per CTA, or equal runs of units, which split tasks.
NEVER_SPLIT = 1 << 20
ALWAYS_SPLIT = -(1 << 20)


def stream_k_split_cost(split_cost_units):
    """Build stream-K plans with this split cost instead of the measured one."""
    return patch.object(
        beam_shared_context_variable_width_stream_k,
        "_SPLIT_COST_UNITS",
        split_cost_units,
    )


def variable_width_case(
    widths,
    prompt_lens,
    decode_lens,
    *,
    q_heads=4,
    kv_heads=2,
    head_dim=64,
    num_slots=1 << 15,
    first_slot=0,
    max_context=1024,
    seed=0,
):
    """A decode batch of beam requests, one per entry of `widths`.

    Rows are laid out as the scheduler does: one leader row per group, then each
    group's member rows. A member's req_to_token row repeats the leader's prompt
    slots and holds its own decode slots; every row of a group has the leader's
    length. The KV slots are `num_slots` slots of the cache from `first_slot` on.
    Returns the batch view the metadata builder reads, the kernel inputs and the
    FP32 reference.
    """
    torch.manual_seed(seed)
    device = torch.device("cuda")
    num_groups = len(widths)
    num_rows = sum(widths)
    # Each row's pool row: scattered, and with spare pool rows left unused.
    pool_rows = torch.randperm(num_rows + 16, device=device)[:num_rows]
    req_to_token = torch.zeros(
        (num_rows + 16, max_context), dtype=torch.int32, device=device
    )
    kv_slots = first_slot + torch.randperm(num_slots, device=device).to(torch.int32)
    seq_lens = torch.zeros(num_rows, dtype=torch.int64, device=device)
    entries = []
    next_member_row, next_slot = num_groups, 0
    for leader_row, (width, prompt_len, decode_len) in enumerate(
        zip(widths, prompt_lens, decode_lens, strict=True)
    ):
        member_rows = range(next_member_row, next_member_row + width - 1)
        prompt_slots = kv_slots[next_slot : next_slot + prompt_len]
        next_slot += prompt_len
        for row in (leader_row, *member_rows):
            req_to_token[pool_rows[row], :prompt_len] = prompt_slots
            req_to_token[pool_rows[row], prompt_len : prompt_len + decode_len] = (
                kv_slots[next_slot : next_slot + decode_len]
            )
            next_slot += decode_len
            seq_lens[row] = prompt_len + decode_len
        # Member rows are counted from the first row after the leaders.
        entries.append(
            BeamTailEntry(
                group=SimpleNamespace(prompt_len=prompt_len),
                leader_idx=leader_row,
                start=member_rows.start - num_groups,
                end=member_rows.stop - num_groups,
            )
        )
        next_member_row = member_rows.stop

    q = torch.randn(num_rows, q_heads, head_dim, dtype=torch.bfloat16, device=device)
    used_slots = torch.randn(
        num_slots, kv_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    if first_slot:
        # One tensor serves as both K and V, to halve what the unused slots take.
        k_buffer = torch.empty(
            (first_slot + num_slots, kv_heads, head_dim),
            dtype=torch.bfloat16,
            device=device,
        )
        k_buffer[first_slot:] = used_slots
        v_buffer = k_buffer
    else:
        k_buffer = used_slots
        v_buffer = torch.randn_like(k_buffer)
    softmax_scale = 1.0 / math.sqrt(head_dim)
    q_per_kv = q_heads // kv_heads
    reference = torch.empty(q.shape, dtype=torch.float32, device=device)
    for row in range(num_rows):
        row_slots = req_to_token[pool_rows[row], : int(seq_lens[row])].long()
        k = k_buffer[row_slots].repeat_interleave(q_per_kv, dim=1).float()
        v = v_buffer[row_slots].repeat_interleave(q_per_kv, dim=1).float()
        scores = torch.einsum("hd,thd->ht", q[row].float(), k) * softmax_scale
        reference[row] = torch.einsum("ht,thd->hd", scores.softmax(dim=-1), v)

    batch = SimpleNamespace(
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens.cpu(),
        beam_tail=SimpleNamespace(num_base_rows=num_groups, entries=entries),
        device=device,
    )
    return SimpleNamespace(
        batch=batch,
        q=q,
        k_buffer=k_buffer,
        v_buffer=v_buffer,
        req_to_token=req_to_token,
        req_pool_indices=pool_rows.to(torch.int64),
        seq_lens=seq_lens,
        softmax_scale=softmax_scale,
        reference=reference,
    )


def build_case_metadata(case, *, stream_k, num_ctas=None):
    """A case's layout from the host's builder, as serving builds it. With
    `stream_k` the layout carries the stream-K plan, for a grid of `num_ctas` CTAs
    (default: the device's)."""
    grid = None
    if stream_k:
        grid = StreamKGrid(
            q_heads=case.q.shape[1],
            num_ctas=num_ctas
            or beam_shared_context_variable_width_stream_k.default_num_ctas(
                case.q.device.index
            ),
        )
    return build_variable_width_metadata(
        batch=case.batch, req_to_token=case.req_to_token, stream_k_grid=grid
    )


def forward_variable_width_attention(tensors, metadata):
    """One layer's shared-context attention through serving's entry point, on a
    case's tensors (or on a graph's own, which carry the same fields). Returns
    [rows, q_heads, head_dim]."""
    num_rows, q_heads, head_dim = tensors.q.shape
    layer = SimpleNamespace(
        layer_id=0,
        tp_q_head_num=q_heads,
        head_dim=head_dim,
        scaling=tensors.softmax_scale,
        logit_cap=0,
    )
    kv_layers = SimpleNamespace(
        read=lambda layer_id: SimpleNamespace(
            kv=lambda: (tensors.k_buffer, tensors.v_buffer)
        )
    )
    forward_batch = SimpleNamespace(
        beam=SimpleNamespace(attention_metadata=metadata),
        req_pool_indices=tensors.req_pool_indices,
        seq_lens=tensors.seq_lens,
    )
    output = shared_context_attention.forward(
        tensors.q.view(num_rows, q_heads * head_dim), layer, forward_batch, kv_layers
    )
    return output.view(num_rows, q_heads, head_dim)


def assert_rows_match_reference(output, reference, max_row_error=0.01):
    """Every row's error against the FP32 reference stays below max_row_error of
    the row, both taken as the RMS over the row's heads and dims.

    Measured on the batches of these tests (prompts of 1 to 5,000 tokens, both
    kernels): 0.16% to 0.27% of a row, from the kernels' bf16 arithmetic. A
    row's magnitude falls with its prompt's length, which an absolute tolerance
    cannot follow.
    """
    error = (output.float() - reference).flatten(1).square().mean(dim=1).sqrt()
    scale = reference.flatten(1).square().mean(dim=1).sqrt()
    worst = int((error / scale).argmax())
    if error[worst] > max_row_error * scale[worst]:
        raise AssertionError(
            f"Row {worst}: RMS error {float(error[worst]):.3e} against a reference "
            f"of RMS {float(scale[worst]):.3e}, more than {max_row_error:.0%}"
        )
