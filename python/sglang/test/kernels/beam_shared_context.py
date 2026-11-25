"""A decode batch for the variable-width shared-context attention kernels, with its
FP32 reference: per row, plain softmax attention over the row's own KV slots."""

import math
from types import SimpleNamespace

import torch

from sglang.srt.beam_search.batch_tail import BeamTailEntry
from sglang.srt.beam_search.shared_context_variable_width import (
    BEAMS_PER_TILE,
    StreamKGrid,
    attention_kernel,
    build_variable_width_metadata,
    device_layout,
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
    max_context=1024,
    seed=0,
):
    """A decode batch of beam requests, one per entry of `widths`.

    Rows are laid out as the scheduler does: one leader row per group, then each
    group's member rows. A member's req_to_token row repeats the leader's prompt
    slots and holds its own decode slots; every row of a group has the leader's
    length. Returns the batch view the metadata builder reads, the kernel inputs
    and the FP32 reference.
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
    kv_slots = torch.randperm(num_slots, device=device).to(torch.int32)
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
    k_buffer = torch.randn(
        num_slots, kv_heads, head_dim, dtype=torch.bfloat16, device=device
    )
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


def build_case_metadata(case, *, stream_k, num_ctas=None, split_cost_units=None):
    """A case's layout from the host's builder, as serving builds it. With
    `stream_k`, `num_ctas` and `split_cost_units` force the grid size of the
    kernel and its plan's choice between whole tasks and equal runs."""
    grid = None
    if stream_k:
        from sglang.kernels.ops.attention.beam_shared_context_variable_width_stream_k import (
            default_num_ctas,
        )

        grid = StreamKGrid(
            q_heads=case.q.shape[1],
            num_ctas=num_ctas or default_num_ctas(case.q.device.index),
            split_cost_units=split_cost_units,
        )
    return build_variable_width_metadata(
        batch=case.batch, req_to_token=case.req_to_token, stream_k=grid
    )


def run_variable_width_attention(case, metadata):
    """Launch the kernel a layout selects, as serving does, on a case's tensors
    (or on a graph's own, which carry the same fields)."""
    grid = metadata.stream_k
    attention = attention_kernel(stream_k=grid is not None)
    grid_options = {} if grid is None else {"num_ctas": grid.num_ctas}
    return attention(
        q=case.q,
        k_buffer=case.k_buffer,
        v_buffer=case.v_buffer,
        req_to_token=metadata.req_to_token,
        req_pool_indices=case.req_pool_indices,
        seq_lens=case.seq_lens,
        layout=device_layout(metadata, device=case.q.device),
        block_m=BEAMS_PER_TILE,
        softmax_scale=case.softmax_scale,
        **grid_options,
    )
