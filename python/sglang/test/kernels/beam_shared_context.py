"""A decode batch for the width-free shared-context attention kernels, with its
FP32 reference: per row, plain softmax attention over the row's own KV slots."""

import math
from types import SimpleNamespace

import torch

from sglang.srt.beam_search.batch_tail import BeamTailEntry


def ragged_case(
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
    """A decode batch's rows as the scheduler lays them out: one leader row per
    group, then each group's member rows. Members alias the leader's prompt slots
    in their own req_to_token rows and own their decode slots; every row of a
    group has the leader's length. A width of 1 is a regular request: one base
    row, outside the beam tail. Returns the batch view the metadata builder reads,
    the kernel inputs and the FP32 reference."""
    torch.manual_seed(seed)
    device = torch.device("cuda")
    num_groups = len(widths)
    num_rows = sum(widths)
    pools = torch.randperm(num_rows + 16, device=device)[:num_rows]
    req_to_token = torch.zeros(
        (num_rows + 16, max_context), dtype=torch.int32, device=device
    )
    slots = torch.randperm(num_slots, device=device).to(torch.int32)
    seq_lens = torch.zeros(num_rows, dtype=torch.int64, device=device)
    entries, group_rows = [], []
    next_member, next_slot = num_groups, 0
    for group, (width, prompt_len, decode_len) in enumerate(
        zip(widths, prompt_lens, decode_lens, strict=True)
    ):
        rows = [group, *range(next_member, next_member + width - 1)]
        prompt = slots[next_slot : next_slot + prompt_len]
        next_slot += prompt_len
        for row in rows:
            req_to_token[pools[row], :prompt_len] = prompt
            req_to_token[pools[row], prompt_len : prompt_len + decode_len] = slots[
                next_slot : next_slot + decode_len
            ]
            next_slot += decode_len
            seq_lens[row] = prompt_len + decode_len
        if width > 1:
            entries.append(
                BeamTailEntry(
                    SimpleNamespace(prompt_len=prompt_len),
                    group,
                    next_member - num_groups,
                    next_member - num_groups + width - 1,
                )
            )
        group_rows.append(rows)
        next_member += width - 1

    q = torch.randn(num_rows, q_heads, head_dim, dtype=torch.bfloat16, device=device)
    k_buffer = torch.randn(
        num_slots, kv_heads, head_dim, dtype=torch.bfloat16, device=device
    )
    v_buffer = torch.randn_like(k_buffer)
    softmax_scale = 1.0 / math.sqrt(head_dim)
    q_per_kv = q_heads // kv_heads
    reference = torch.empty(q.shape, dtype=torch.float32, device=device)
    for rows in group_rows:
        for row in rows:
            keys = req_to_token[pools[row], : int(seq_lens[row])].long()
            k = k_buffer[keys].repeat_interleave(q_per_kv, dim=1).float()
            v = v_buffer[keys].repeat_interleave(q_per_kv, dim=1).float()
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
        req_pool_indices=pools.to(torch.int64),
        seq_lens=seq_lens,
        softmax_scale=softmax_scale,
        reference=reference,
    )


def run_ragged(case, metadata, *, block_m, **stream_k):
    """Launch the kernel a layout selects on a case's tensors (or a graph's own,
    which carry the same fields)."""
    if metadata.stream_k:
        from sglang.kernels.ops.attention.beam_shared_context_ragged_stream_k import (
            shared_context_attention_ragged_stream_k as attention,
        )
    else:
        from sglang.kernels.ops.attention.beam_shared_context_ragged import (
            shared_context_attention_ragged as attention,
        )
    return attention(
        q=case.q,
        k_buffer=case.k_buffer,
        v_buffer=case.v_buffer,
        req_to_token=metadata.req_to_token,
        req_pool_indices=case.req_pool_indices,
        seq_lens=case.seq_lens,
        layout=metadata.layout,
        block_m=block_m,
        softmax_scale=case.softmax_scale,
        **stream_k,
    )
