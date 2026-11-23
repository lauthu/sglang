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
"""Shared-context attention for any split of a decode step's rows into groups.

The kernels read the layout the host builds each step, so every case goes
through the real builder: a layout the two sides read differently fails against
the FP32 reference.
"""

import unittest

import torch

from sglang.srt.beam_search.shared_context_ragged import (
    RAGGED_BLOCK_M,
    build_ragged_metadata,
)
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.beam_shared_context import ragged_case, run_ragged
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=40, stage="base-b-kernel-unit", runner_config="1-gpu-small")


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestRaggedSharedContextAttention(CustomTestCase):
    def _assert_matches_reference(self, case, *, one_row_groups, stream_k, **forced):
        metadata = build_ragged_metadata(
            batch=case.batch,
            req_to_token=case.req_to_token,
            one_row_groups=one_row_groups,
            stream_k=stream_k,
        )
        output = run_ragged(case, metadata, block_m=RAGGED_BLOCK_M, **forced)
        torch.testing.assert_close(output.float(), case.reference, rtol=1e-2, atol=1e-2)

    def test_any_split_matches_fp32_reference(self):
        """Mixed widths across the 128-beam tile edge, each group with its own prompt
        and decode length, leaders apart from their member rows."""
        case = ragged_case(
            widths=(2, 3, 127, 128, 129, 256, 300),
            prompt_lens=(1, 700, 64, 65, 300, 129, 513),
            decode_lens=(1, 4, 2, 3, 1, 2, 4),
        )
        for stream_k in (False, True):
            with self.subTest(stream_k=stream_k):
                self._assert_matches_reference(
                    case, one_row_groups=False, stream_k=stream_k
                )

    def test_ordinary_rows_as_one_row_groups_match_fp32_reference(self):
        """Ordinary requests among beam groups attend over their whole KV as one-row
        groups, up to 5,000 tokens, in the same launch as the beams."""
        case = ragged_case(
            widths=(1, 129, 1, 2, 1, 300),
            prompt_lens=(1, 700, 2048, 64, 5000, 513),
            decode_lens=(5, 1, 3, 2, 0, 4),
            max_context=8192,
        )
        for stream_k in (False, True):
            with self.subTest(stream_k=stream_k):
                self._assert_matches_reference(
                    case, one_row_groups=True, stream_k=stream_k
                )

    def test_stream_k_any_cut_of_the_units_matches_fp32_reference(self):
        """Wherever the CTA runs cut the work (inside a prompt, at a decode step,
        across heads, tiles and groups, CTAs without units), the partial segments
        merge into the same attention; so do whole tiles per CTA."""
        never, always = 1 << 20, -(1 << 20)
        mixed = ragged_case(
            widths=(1, 129, 2, 300, 1, 7),
            prompt_lens=(1, 700, 64, 513, 2048, 130),
            decode_lens=(5, 1, 2, 4, 0, 3),
            max_context=4096,
        )
        tiny = ragged_case(widths=(2,), prompt_lens=(1,), decode_lens=(1,))
        runs = (
            (mixed, 1, always),
            (mixed, 3, always),
            (mixed, 7, always),
            (mixed, 188, always),
            (mixed, 188, never),
            (tiny, 64, always),
        )
        for index, (case, num_ctas, split_cost_units) in enumerate(runs):
            with self.subTest(run=index, num_ctas=num_ctas, whole=split_cost_units > 0):
                self._assert_matches_reference(
                    case,
                    one_row_groups=True,
                    stream_k=True,
                    num_ctas=num_ctas,
                    split_cost_units=split_cost_units,
                )


if __name__ == "__main__":
    unittest.main()
