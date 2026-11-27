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
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.kernels.ops.attention import beam_shared_context_variable_width_stream_k
from sglang.kernels.ops.attention.beam_shared_context_variable_width_stream_k import (
    shared_context_attention_variable_width_stream_k,
)
from sglang.srt.beam_search import shared_context_attention
from sglang.srt.beam_search.shared_context_attention_variable_width import (
    VariableWidthBeamAttentionMetadata,
)
from sglang.srt.environ import envs
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.beam_shared_context_variable_width import (
    ALWAYS_SPLIT,
    NEVER_SPLIT,
    assert_rows_match_reference,
    build_case_metadata,
    forward_variable_width_attention,
    stream_k_split_cost,
    variable_width_case,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=60, stage="base-b-kernel-unit", runner_config="1-gpu-small")

# The SMs of an RTX PRO 6000, the grid the stream-K kernel was tuned with.
TUNED_NUM_CTAS = 188


class _ModelRunner:
    """The model runner fields that pick a server's variable-width kernel."""

    def __init__(self, *, q_heads, kv_heads, head_dim, req_to_token):
        self.model = torch.nn.Sequential(
            RadixAttention(
                num_heads=q_heads,
                head_dim=head_dim,
                scaling=head_dim**-0.5,
                num_kv_heads=kv_heads,
                layer_id=0,
            )
        )
        self.req_to_token_pool = SimpleNamespace(req_to_token=req_to_token)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestVariableWidthSharedContextAttention(CustomTestCase):
    def _assert_matches_reference(self, case, metadata):
        output = forward_variable_width_attention(case, metadata)
        assert_rows_match_reference(output, case.reference)

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
                self._assert_matches_reference(
                    case, build_case_metadata(case, stream_k=stream_k)
                )

    def test_stream_k_ranges_cut_anywhere_match_fp32_reference(self):
        """Wherever the CTAs' ranges cut the work (inside a prompt, at a decode step,
        between heads, tiles and groups, with CTAs left an empty range), the parts
        merge into the same attention; so do ranges that end only between (tile,
        head) pairs."""
        mixed = variable_width_case(
            widths=(2, 129, 2, 300, 3, 7),
            prompt_lens=(1, 700, 64, 513, 2048, 130),
            decode_lens=(5, 1, 2, 4, 1, 3),
            max_context=4096,
        )
        # 4 heads of one tile of 2 work units: most of the 64 CTAs get an empty range.
        tiny = variable_width_case(widths=(2,), prompt_lens=(1,), decode_lens=(1,))
        launches = (
            (mixed, 1, ALWAYS_SPLIT),
            (mixed, 3, ALWAYS_SPLIT),
            (mixed, 7, ALWAYS_SPLIT),
            (mixed, TUNED_NUM_CTAS, ALWAYS_SPLIT),
            (mixed, TUNED_NUM_CTAS, NEVER_SPLIT),
            (tiny, 64, ALWAYS_SPLIT),
        )
        for index, (case, num_ctas, split_cost_units) in enumerate(launches):
            with (
                self.subTest(
                    launch=index,
                    num_ctas=num_ctas,
                    never_split=split_cost_units == NEVER_SPLIT,
                ),
                stream_k_split_cost(split_cost_units),
            ):
                self._assert_matches_reference(
                    case, build_case_metadata(case, stream_k=True, num_ctas=num_ctas)
                )

    def test_kv_cache_past_2_31_elements(self):
        """K/V offsets past 2^31 elements must not wrap. The default kernel always
        computes them in 64 bits; stream-K switches to 64 bits for such a cache."""
        if torch.cuda.mem_get_info()[0] < 6 << 30:
            self.skipTest("needs 6 GB of free GPU memory")
        kv_heads, head_dim = 2, 64
        # Every slot of the case sits past 2^31 elements of the cache.
        case = variable_width_case(
            widths=(2, 129),
            prompt_lens=(70, 300),
            decode_lens=(2, 1),
            kv_heads=kv_heads,
            head_dim=head_dim,
            num_slots=1 << 12,
            first_slot=(1 << 31) // (kv_heads * head_dim) + 16,
        )
        for stream_k in (False, True):
            with self.subTest(stream_k=stream_k):
                self._assert_matches_reference(
                    case, build_case_metadata(case, stream_k=stream_k)
                )

    def test_server_runs_the_selected_kernel(self):
        """SGLANG_BEAM_SHARED_CONTEXT_KERNEL picks the kernel of a server whose beam
        decode graphs take any width. Stream-K's plan is built for the model's query
        heads and one CTA per SM of the GPU. A batch's layout is uploaded by the
        first layer that runs it and kept for the step's other layers."""
        case = variable_width_case(
            widths=(2, 129, 16),
            prompt_lens=(700, 64, 2048),
            decode_lens=(1, 2, 3),
            max_context=4096,
        )
        _, q_heads, head_dim = case.q.shape
        model_runner = _ModelRunner(
            q_heads=q_heads,
            kv_heads=case.k_buffer.shape[1],
            head_dim=head_dim,
            req_to_token=case.req_to_token,
        )
        case.batch.forward_mode = SimpleNamespace(is_decode=lambda: True)
        for kernel in ("default", "stream_k"):
            with (
                self.subTest(kernel=kernel),
                envs.SGLANG_ENABLE_BEAM_SHARED_CONTEXT_ATTENTION.override(True),
                envs.SGLANG_ENABLE_BEAM_DECODE_GRAPH.override(True),
                envs.SGLANG_BEAM_DECODE_GRAPH_FIXED_WIDTH.override(None),
                envs.SGLANG_BEAM_SHARED_CONTEXT_KERNEL.override(kernel),
                patch.object(
                    shared_context_attention,
                    "supports_shared_context",
                    return_value=True,
                ),
                patch.object(
                    beam_shared_context_variable_width_stream_k,
                    "shared_context_attention_variable_width_stream_k",
                    wraps=shared_context_attention_variable_width_stream_k,
                ) as stream_k,
            ):
                metadata = shared_context_attention.build_metadata(
                    case.batch, model_runner
                )
                self.assertIsInstance(metadata, VariableWidthBeamAttentionMetadata)
                self.assertIsNone(metadata.layout)
                self._assert_matches_reference(case, metadata)
                self.assertEqual(metadata.layout.device, case.q.device)
                self.assertEqual(stream_k.call_count, kernel == "stream_k")
                if kernel == "stream_k":
                    self.assertEqual(
                        stream_k.call_args.kwargs["num_ctas"],
                        torch.cuda.get_device_properties(
                            case.q.device
                        ).multi_processor_count,
                    )
                    self.assertEqual(metadata.stream_k_grid.q_heads, q_heads)


if __name__ == "__main__":
    unittest.main()
