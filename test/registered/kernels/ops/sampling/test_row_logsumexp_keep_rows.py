"""row_logsumexp_keep_rows.

The trie output head swaps it in for row_logsumexp when ordinary requests share a
beam decode step, so beam scores must not move by a bit, and a CUDA graph must
keep exactly the rows its device-side count names on each replay.
"""

import itertools
import unittest

import torch

from sglang.kernels.ops.sampling.row_logsumexp_keep_rows import (
    row_logsumexp_keep_rows,
)
from sglang.srt.layers.logsumexp import row_logsumexp
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=10, stage="base-b-kernel-unit", runner_config="1-gpu-small")

_SENTINEL = -12345.0


def _kept_buffer(capacity, total_cols):
    return torch.full(
        (capacity, total_cols), _SENTINEL, dtype=torch.float32, device="cuda"
    )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestRowLogsumexpKeepRows(CustomTestCase):
    def test_matches_row_logsumexp_and_keeps_counted_rows(self):
        """The normalizer is bitwise row_logsumexp's; rows below min(count, capacity)
        land as fp32 in the destination's column slice, and nothing else is written.
        20,000 columns span two 16,384-column blocks."""
        num_rows, capacity, col_offset = 300, 64, 7
        for dtype, num_cols, count in itertools.product(
            (torch.bfloat16, torch.float32), (8192, 20000), (0, 3, 64, 1000)
        ):
            with self.subTest(dtype=dtype, num_cols=num_cols, count=count):
                torch.manual_seed(num_cols + count)
                x = torch.randn(num_rows, num_cols, device="cuda").to(dtype) * 8
                buffer = _kept_buffer(capacity, num_cols + 2 * col_offset)
                kept_count = torch.tensor([count], dtype=torch.int32, device="cuda")

                out_max, out_log_sum = row_logsumexp_keep_rows(
                    x, buffer[:, col_offset : col_offset + num_cols], kept_count
                )

                ref_max, ref_log_sum = row_logsumexp(x)
                self.assertTrue(torch.equal(out_max, ref_max))
                self.assertTrue(torch.equal(out_log_sum, ref_log_sum))
                kept_rows = min(count, capacity)
                window = buffer[:, col_offset : col_offset + num_cols]
                self.assertTrue(torch.equal(window[:kept_rows], x[:kept_rows].float()))
                self.assertTrue(bool((window[kept_rows:] == _SENTINEL).all()))
                self.assertTrue(bool((buffer[:, :col_offset] == _SENTINEL).all()))
                self.assertTrue(
                    bool((buffer[:, col_offset + num_cols :] == _SENTINEL).all())
                )

    def test_graph_keeps_the_rows_counted_at_replay(self):
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
            with self.subTest(count=count):
                buffer.fill_(_SENTINEL)
                kept_count.fill_(count)
                graph.replay()
                torch.cuda.synchronize()
                self.assertTrue(torch.equal(buffer[:count], x[:count].float()))
                self.assertTrue(bool((buffer[count:] == _SENTINEL).all()))


if __name__ == "__main__":
    unittest.main()
