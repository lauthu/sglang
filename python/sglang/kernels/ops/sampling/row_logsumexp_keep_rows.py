"""Row logsumexp that also keeps the leading input rows, counted on the GPU.

The trie output head streams the full vocabulary through ``row_logsumexp`` for
its normalizer. Ordinary requests decoding next to beam requests need those
full-vocabulary logits for sampling, so this variant also writes the first
``kept_count[0]`` rows out as fp32 in the same pass. The count is read on the
GPU, so one CUDA graph keeps any number of rows, and none in a beam-only step.

The ``(max, log_sum)`` outputs are those of ``row_logsumexp``: same block size,
same per-block accumulation.
"""

import torch
import triton
import triton.language as tl

from sglang.srt.layers.logsumexp import _accumulate_block_lse


@triton.jit
def _row_logsumexp_keep_rows_kernel(
    x_ptr,
    out_max_ptr,
    out_log_sum_ptr,
    kept_ptr,
    kept_count_ptr,
    row_stride,
    col_stride,
    kept_row_stride,
    # int rather than tl.constexpr so triton does not unroll the loop
    num_cols,
    kept_capacity,
    BLOCK_N: tl.constexpr,
):
    # One program per row.
    row = tl.program_id(0).to(tl.int64)
    row_ptr = x_ptr + row * row_stride
    kept_row_ptr = kept_ptr + row * kept_row_stride
    keep = row < tl.minimum(tl.load(kept_count_ptr), kept_capacity)

    m_i = float("-inf")
    l_i = 0.0
    for start in range(0, num_cols, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        in_row = offs < num_cols
        x = tl.load(row_ptr + offs * col_stride, mask=in_row, other=float("-inf")).to(
            tl.float32
        )
        tl.store(kept_row_ptr + offs, x, mask=in_row & keep)
        m_i, l_i, _ = _accumulate_block_lse(x, m_i, l_i)

    tl.store(out_max_ptr + row, m_i)
    tl.store(out_log_sum_ptr + row, tl.log(l_i))


def row_logsumexp_keep_rows(
    x: torch.Tensor, kept: torch.Tensor, kept_count: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """``row_logsumexp(x)``; also writes ``x[row]`` as fp32 into ``kept[row]`` for
    every row below both ``kept_count[0]`` and ``kept``'s row count.

    ``kept`` is ``[capacity, x.shape[1]]`` fp32 with a unit column stride, e.g. a
    column slice of a full-vocabulary buffer; its other rows are left untouched.
    """
    assert x.ndim == 2 and kept.ndim == 2
    assert x.dtype in (torch.float16, torch.bfloat16, torch.float32)
    assert kept.dtype == torch.float32 and kept.stride(1) == 1
    assert kept.shape[1] == x.shape[1]
    assert kept_count.dtype == torch.int32 and kept_count.numel() == 1

    num_rows, num_cols = x.shape
    out_max = torch.empty(num_rows, device=x.device, dtype=torch.float32)
    out_log_sum = torch.empty(num_rows, device=x.device, dtype=torch.float32)
    if num_rows == 0:
        return out_max, out_log_sum
    if num_cols == 0:
        return out_max.fill_(float("-inf")), out_log_sum.zero_()

    # Same block size and warps as row_logsumexp, so the outputs match it.
    BLOCK_N = triton.next_power_of_2(min(num_cols, 16384))
    _row_logsumexp_keep_rows_kernel[(num_rows,)](
        x,
        out_max,
        out_log_sum,
        kept,
        kept_count,
        x.stride(0),
        x.stride(1),
        kept.stride(0),
        num_cols,
        kept.shape[0],
        BLOCK_N=BLOCK_N,
        num_warps=8,
    )
    return out_max, out_log_sum
