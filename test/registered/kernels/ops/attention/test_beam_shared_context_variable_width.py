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
"""Shared-context attention for beam requests of any widths in one batch.

The kernels read the layout that the host builds for each batch, so every case
goes through the real builder: if the two sides read a layout differently, the
output fails against the FP32 reference.
"""

import unittest

import torch

from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.beam_shared_context import (
    build_case_metadata,
    run_variable_width_attention,
    variable_width_case,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=40, stage="base-b-kernel-unit", runner_config="1-gpu-small")

# split_cost_units that make the stream-K plan give each CTA whole tasks, or equal
# runs of units that split tasks, whatever the batch.
NEVER_SPLIT = 1 << 20
ALWAYS_SPLIT = -(1 << 20)
# The SMs of an RTX PRO 6000, the grid the stream-K kernel was tuned with.
TUNED_NUM_CTAS = 188


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestVariableWidthSharedContextAttention(CustomTestCase):
    def _assert_matches_reference(self, case, *, stream_k, **stream_k_options):
        metadata = build_case_metadata(case, stream_k=stream_k, **stream_k_options)
        output = run_variable_width_attention(case, metadata)
        torch.testing.assert_close(output.float(), case.reference, rtol=1e-2, atol=1e-2)

    def test_any_widths_match_fp32_reference(self):
        """Beam widths on both sides of the 128-beam tile size in one batch, each
        request with its own prompt (up to 5,000 tokens) and generated tokens."""
        case = variable_width_case(
            widths=(2, 3, 127, 128, 129, 256, 300),
            prompt_lens=(1, 5000, 64, 65, 300, 129, 513),
            decode_lens=(1, 4, 2, 3, 1, 2, 4),
            max_context=8192,
        )
        for stream_k in (False, True):
            with self.subTest(stream_k=stream_k):
                self._assert_matches_reference(case, stream_k=stream_k)

    def test_stream_k_any_cut_of_the_units_matches_fp32_reference(self):
        """Wherever the CTAs' runs cut the work (inside a prompt, at a decode step,
        between heads, tiles and groups, with CTAs left without units), the
        segments merge into the same attention; so do whole tasks per CTA."""
        mixed = variable_width_case(
            widths=(2, 129, 2, 300, 3, 7),
            prompt_lens=(1, 700, 64, 513, 2048, 130),
            decode_lens=(5, 1, 2, 4, 1, 3),
            max_context=4096,
        )
        # 4 tasks of 3 units each: most of the 64 CTAs get no unit.
        tiny = variable_width_case(widths=(2,), prompt_lens=(1,), decode_lens=(1,))
        runs = (
            (mixed, 1, ALWAYS_SPLIT),
            (mixed, 3, ALWAYS_SPLIT),
            (mixed, 7, ALWAYS_SPLIT),
            (mixed, TUNED_NUM_CTAS, ALWAYS_SPLIT),
            (mixed, TUNED_NUM_CTAS, NEVER_SPLIT),
            (tiny, 64, ALWAYS_SPLIT),
        )
        for index, (case, num_ctas, split_cost_units) in enumerate(runs):
            with self.subTest(
                run=index,
                num_ctas=num_ctas,
                whole_tasks=split_cost_units == NEVER_SPLIT,
            ):
                self._assert_matches_reference(
                    case,
                    stream_k=True,
                    num_ctas=num_ctas,
                    split_cost_units=split_cost_units,
                )


if __name__ == "__main__":
    unittest.main()
