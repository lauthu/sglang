"""row_logsumexp_keep_rows, and the trie head's use of it.

The trie output head swaps it in for row_logsumexp when ordinary requests share a
beam decode step, so beam scores must not move by a bit, and a CUDA graph must
keep exactly the rows its device-side count names on each replay.
"""

import pytest
import torch

from sglang.kernels.ops.sampling.row_logsumexp_keep_rows import (
    row_logsumexp_keep_rows,
)
from sglang.srt.layers.logsumexp import row_logsumexp
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=15, stage="base-b-kernel-unit", runner_config="1-gpu-small")

_SENTINEL = -12345.0


def _kept_buffer(capacity, total_cols):
    return torch.full(
        (capacity, total_cols), _SENTINEL, dtype=torch.float32, device="cuda"
    )


@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize("num_cols", [8192, 20000])
@pytest.mark.parametrize("count", [0, 3, 64, 1000])
def test_matches_row_logsumexp_and_keeps_counted_rows(dtype, num_cols, count):
    """The normalizer is bitwise row_logsumexp's; rows below min(count, capacity)
    land as fp32 in the destination's column slice, and nothing else is written.
    20,000 columns span two 16,384-column blocks."""
    torch.manual_seed(num_cols + count)
    num_rows, capacity, col_offset = 300, 64, 7
    x = torch.randn(num_rows, num_cols, device="cuda").to(dtype) * 8
    buffer = _kept_buffer(capacity, num_cols + 2 * col_offset)
    kept_count = torch.tensor([count], dtype=torch.int32, device="cuda")

    out_max, out_log_sum = row_logsumexp_keep_rows(
        x, buffer[:, col_offset : col_offset + num_cols], kept_count
    )

    ref_max, ref_log_sum = row_logsumexp(x)
    assert torch.equal(out_max, ref_max)
    assert torch.equal(out_log_sum, ref_log_sum)
    kept_rows = min(count, capacity)
    window = buffer[:, col_offset : col_offset + num_cols]
    assert torch.equal(window[:kept_rows], x[:kept_rows].float())
    assert bool((window[kept_rows:] == _SENTINEL).all())
    assert bool((buffer[:, :col_offset] == _SENTINEL).all())
    assert bool((buffer[:, col_offset + num_cols :] == _SENTINEL).all())


def test_graph_keeps_the_rows_counted_at_replay():
    """A graph captured with count 0 keeps whatever count is staged before each
    replay: the count is read on the GPU, not baked in at capture."""
    torch.manual_seed(0)
    x = torch.randn(16, 4096, device="cuda", dtype=torch.bfloat16)
    buffer = _kept_buffer(8, 4096)
    kept_count = torch.zeros(1, dtype=torch.int32, device="cuda")

    row_logsumexp_keep_rows(x, buffer, kept_count)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        row_logsumexp_keep_rows(x, buffer, kept_count)
    for count in (5, 0, 8):
        buffer.fill_(_SENTINEL)
        kept_count.fill_(count)
        graph.replay()
        torch.cuda.synchronize()
        assert torch.equal(buffer[:count], x[:count].float())
        assert bool((buffer[count:] == _SENTINEL).all())


def test_trie_head_keeps_every_chunk_in_its_columns():
    """On CUDA the trie head writes each vocabulary chunk's transformed logits into
    that chunk's columns of the kept rows, and keeping leaves its normalizer as is."""
    from sglang.srt.beam_search.mixed_requests import BeamDenseRows
    from sglang.srt.beam_search.trie_config import TrieOutputHeadConfig
    from sglang.srt.beam_search.trie_output_head import BeamTrieOutputHead

    torch.manual_seed(0)
    config = TrieOutputHeadConfig(
        tensor_path="/unused/trie_output_head.safetensors",
        token_start=8,
        codebook_size=4,
        num_codebooks=2,
    )
    vocab_size, chunk_size = 2500, 1000
    hidden_states = torch.randn(6, 64, device="cuda", dtype=torch.bfloat16)
    weight = torch.randn(2560, 64, device="cuda", dtype=torch.bfloat16)
    depths = torch.zeros(6, dtype=torch.int64, device="cuda")

    def project(states, selected_weight):
        return torch.matmul(states, selected_weight.T)

    def transform(logits):
        return logits * 2

    head = BeamTrieOutputHead(config, vocab_size=vocab_size, chunk_size=chunk_size)
    plain = head.project_mixed(hidden_states, weight, depths, project, transform)
    kept = head.project_mixed(
        hidden_states,
        weight,
        depths,
        project,
        transform,
        dense_rows=BeamDenseRows(num_rows=4),
    )
    chunks = [
        transform(
            project(hidden_states, weight[start : min(start + chunk_size, vocab_size)])
        )
        for start in range(0, vocab_size, chunk_size)
    ]
    reference = torch.cat(chunks, dim=1).float()
    assert torch.equal(kept.dense_logits, reference[:4])
    assert torch.equal(kept.normalizer, plain.normalizer)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
