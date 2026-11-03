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

import itertools
import math
import os
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from sglang.srt.beam_search import shared_context_split_kv_kernel
from sglang.srt.beam_search.shared_context_attention import (
    BeamAttentionMetadata,
    build_graph_capture_metadata,
    build_metadata,
    forward,
    graph_compatible,
    stage_graph_metadata,
)
from sglang.srt.beam_search.shared_context_attention_kernel import (
    shared_context_attention,
)
from sglang.srt.beam_search.shared_context_split_kv_kernel import (
    shared_context_attention_split_kv,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


def _scheduler_layout_case(context_lens, decode_len=2, decode_capacity=3):
    """Random beam rows in the scheduler's row layout and their fp32 reference.

    Prompts and decode tokens sit at scattered K/V slots, and the decode-slot
    capacity runs past the decode length.
    """
    torch.manual_seed(0)
    device = torch.device("cuda")
    beam_width = 4
    q_heads = 4
    kv_heads = 2
    head_dim = 64
    num_groups = len(context_lens)
    num_slots = 1024

    q = torch.randn(
        num_groups * beam_width,
        q_heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    k_buffer = torch.randn(
        num_slots,
        kv_heads,
        head_dim,
        dtype=torch.bfloat16,
        device=device,
    )
    v_buffer = torch.randn_like(k_buffer)
    slots = torch.randperm(num_slots, device=device).to(torch.int32)
    context_slots = slots[: sum(context_lens)]
    context_cu_seqlens = torch.tensor(
        [0, *itertools.accumulate(context_lens)],
        dtype=torch.int32,
        device=device,
    )
    decode_slots = slots[
        sum(context_lens) : sum(context_lens)
        + num_groups * decode_capacity * beam_width
    ].view(num_groups, decode_capacity * beam_width)
    decode_lens = torch.full(
        (num_groups,),
        decode_len,
        dtype=torch.int32,
        device=device,
    )
    softmax_scale = 1.0 / math.sqrt(head_dim)

    reference = torch.empty(q.shape, dtype=torch.float32, device=device)
    for group, (start, end) in enumerate(
        itertools.pairwise(context_cu_seqlens.tolist())
    ):
        for beam_index in range(beam_width):
            # Scheduler rows: group leaders first, then the groups' other beams.
            row = (
                group
                if beam_index == 0
                else num_groups + group * (beam_width - 1) + beam_index - 1
            )
            slot_ids = torch.cat(
                (
                    context_slots[start:end],
                    decode_slots[group, beam_index::beam_width][:decode_len],
                )
            ).long()
            for q_head in range(q_heads):
                kv_head = q_head // (q_heads // kv_heads)
                scores = k_buffer[slot_ids, kv_head].float() @ q[row, q_head].float()
                probabilities = torch.softmax(scores * softmax_scale, dim=0)
                reference[row, q_head] = (
                    probabilities[:, None] * v_buffer[slot_ids, kv_head].float()
                ).sum(dim=0)

    metadata = BeamAttentionMetadata(
        num_groups=num_groups,
        context_slots=context_slots,
        context_cu_seqlens=context_cu_seqlens,
        decode_slots=decode_slots,
        beam_width=beam_width,
        decode_len=decode_len,
        decode_lens=decode_lens,
    )
    return SimpleNamespace(
        q=q,
        k_buffer=k_buffer,
        v_buffer=v_buffer,
        metadata=metadata,
        softmax_scale=softmax_scale,
        reference=reference,
    )


_GRAPH_ENV = {
    "SGLANG_BEAM_SHARED_CONTEXT_ATTENTION": "true",
    "SGLANG_BEAM_SHARED_CONTEXT_WIDTH": "4",
    "SGLANG_BEAM_SHARED_CONTEXT_GRAPH_MAX_CONTEXT": "4",
    "SGLANG_BEAM_SHARED_CONTEXT_GRAPH_MAX_DECODE": "3",
}

_LONG_CONTEXT_GRAPH_ENV = {
    **_GRAPH_ENV,
    "SGLANG_BEAM_SHARED_CONTEXT_GRAPH_MAX_CONTEXT": "32768",
}


class TestSharedContextAttentionMetadata(CustomTestCase):
    def test_missing_metadata_preserves_normal_attention_fallback(self):
        self.assertIsNone(
            forward(
                torch.empty(0),
                SimpleNamespace(),
                SimpleNamespace(beam_attention_metadata=None),
                SimpleNamespace(),
            )
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_native_kernel_matches_fp32_reference(self):
        torch.manual_seed(0)
        device = torch.device("cuda")
        beam_width = 8
        q_heads = 4
        kv_heads = 2
        head_dim = 64
        context_len = 5
        decode_len = 2
        num_slots = context_len + decode_len * beam_width

        q = torch.randn(
            1,
            beam_width,
            q_heads,
            head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        k_buffer = torch.randn(
            num_slots,
            kv_heads,
            head_dim,
            dtype=torch.bfloat16,
            device=device,
        )
        v_buffer = torch.randn_like(k_buffer)
        context_slots = torch.arange(
            context_len,
            dtype=torch.int32,
            device=device,
        )
        context_cu_seqlens = torch.tensor(
            [0, context_len],
            dtype=torch.int32,
            device=device,
        )
        decode_slots = torch.arange(
            context_len,
            num_slots,
            dtype=torch.int32,
            device=device,
        ).view(1, decode_len * beam_width)
        decode_lens = torch.tensor([decode_len], dtype=torch.int32, device=device)
        softmax_scale = 1.0 / math.sqrt(head_dim)

        output = shared_context_attention(
            q,
            k_buffer,
            v_buffer,
            context_slots,
            context_cu_seqlens,
            decode_slots,
            decode_lens,
            softmax_scale=softmax_scale,
        )

        reference = torch.empty_like(output)
        for beam_index in range(beam_width):
            slot_ids = torch.cat(
                (
                    context_slots,
                    decode_slots[
                        0,
                        beam_index::beam_width,
                    ],
                )
            ).long()
            for q_head in range(q_heads):
                kv_head = q_head // (q_heads // kv_heads)
                keys = k_buffer[slot_ids, kv_head].float()
                values = v_buffer[slot_ids, kv_head].float()
                scores = keys @ q[0, beam_index, q_head].float()
                probabilities = torch.softmax(scores * softmax_scale, dim=0)
                reference[0, beam_index, q_head] = (
                    probabilities[:, None] * values
                ).sum(dim=0)

        torch.testing.assert_close(
            output.float(),
            reference.float(),
            rtol=1e-2,
            atol=1e-2,
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_split_kv_matches_fp32_reference(self):
        """Every split count gives the unsplit result, also when a short prompt
        leaves splits empty; decode slots past a group's decode length are unread."""
        case = _scheduler_layout_case([1, 70, 300])
        metadata = case.metadata
        for num_splits in (1, 3, 8):
            with self.subTest(num_splits=num_splits):
                output = shared_context_attention_split_kv(
                    case.q,
                    case.k_buffer,
                    case.v_buffer,
                    metadata.context_slots,
                    metadata.context_cu_seqlens,
                    metadata.decode_slots,
                    metadata.decode_lens,
                    beam_width=metadata.beam_width,
                    softmax_scale=case.softmax_scale,
                    num_splits=num_splits,
                )
                torch.testing.assert_close(
                    output.float(),
                    case.reference,
                    rtol=1e-2,
                    atol=1e-2,
                )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_forward_runs_split_kv_only_when_enabled(self):
        """Both paths match the reference; the flag decides which one runs."""
        case = _scheduler_layout_case([1, 70, 300])
        num_rows, q_heads, head_dim = case.q.shape
        layer = SimpleNamespace(
            layer_id=0,
            tp_q_head_num=q_heads,
            head_dim=head_dim,
            scaling=case.softmax_scale,
        )
        forward_batch = SimpleNamespace(beam_attention_metadata=case.metadata)
        token_to_kv_pool = SimpleNamespace(
            get_kv_buffer=lambda layer_id: (case.k_buffer, case.v_buffer)
        )
        for split_kv in (False, True):
            env = {"SGLANG_BEAM_SHARED_CONTEXT_SPLIT_KV": str(split_kv).lower()}
            with (
                self.subTest(split_kv=split_kv),
                patch.dict(os.environ, env),
                patch.object(
                    shared_context_split_kv_kernel,
                    "shared_context_attention_split_kv",
                    wraps=shared_context_split_kv_kernel.shared_context_attention_split_kv,
                ) as split_kv_path,
            ):
                output = forward(
                    case.q.view(num_rows, q_heads * head_dim),
                    layer,
                    forward_batch,
                    token_to_kv_pool,
                )
                self.assertEqual(split_kv_path.called, split_kv)
                torch.testing.assert_close(
                    output.view(num_rows, q_heads, head_dim).float(),
                    case.reference,
                    rtol=1e-2,
                    atol=1e-2,
                )

    @patch.dict(os.environ, _LONG_CONTEXT_GRAPH_ENV)
    def test_graph_metadata_supports_32k_context_capacity(self):
        target = build_graph_capture_metadata(
            num_rows=8,
            device=torch.device("cpu"),
        )

        self.assertIsNotNone(target)
        self.assertEqual(target.context_slots.numel(), 2 * 32768)
        self.assertEqual(target.context_slots.dtype, torch.int32)
        self.assertEqual(target.decode_slots.dtype, torch.int32)
        self.assertEqual(target.context_cu_seqlens.tolist(), [0, 1, 2])

    @patch.dict(os.environ, _GRAPH_ENV)
    def test_graph_metadata_staging(self):
        target = build_graph_capture_metadata(
            num_rows=8,
            device=torch.device("cpu"),
        )
        self.assertIsNotNone(target)
        source = BeamAttentionMetadata(
            num_groups=2,
            context_slots=torch.tensor([11, 12, 13, 14, 15]),
            context_cu_seqlens=torch.tensor([0, 2, 5], dtype=torch.int32),
            decode_slots=torch.arange(16, 24).view(2, 4),
            beam_width=4,
            decode_len=1,
            decode_lens=torch.ones(2, dtype=torch.int32),
        )

        self.assertTrue(graph_compatible(source, num_rows=8, captured_rows=8))
        stage_graph_metadata(target, source)

        self.assertEqual(target.context_slots.tolist(), [11, 12, 13, 14, 15, 0, 0, 0])
        self.assertEqual(target.context_cu_seqlens.tolist(), [0, 2, 5])
        self.assertEqual(
            target.decode_slots.tolist(),
            [
                [16, 17, 18, 19, 0, 0, 0, 0, 0, 0, 0, 0],
                [20, 21, 22, 23, 0, 0, 0, 0, 0, 0, 0, 0],
            ],
        )
        self.assertEqual(target.decode_lens.tolist(), [1, 1])

    @patch.dict(os.environ, _GRAPH_ENV)
    def test_graph_metadata_rejects_capacity_overflow(self):
        source = BeamAttentionMetadata(
            num_groups=2,
            context_slots=torch.arange(9),
            context_cu_seqlens=torch.tensor([0, 4, 9], dtype=torch.int32),
            decode_slots=torch.arange(8).view(2, 4),
            beam_width=4,
            decode_len=1,
            decode_lens=torch.ones(2, dtype=torch.int32),
        )
        self.assertFalse(graph_compatible(source, num_rows=8, captured_rows=8))

    @patch.dict(os.environ, _GRAPH_ENV)
    def test_mixed_batch_falls_back_to_normal_attention(self):
        entry = SimpleNamespace(start=0, end=3)
        batch = SimpleNamespace(
            beam_tail=SimpleNamespace(entries=[entry]),
            seq_lens=torch.ones(5, dtype=torch.int32),
            seq_lens_cpu=torch.ones(5, dtype=torch.int32),
        )

        self.assertIsNone(build_metadata(batch, SimpleNamespace()))


if __name__ == "__main__":
    unittest.main()
