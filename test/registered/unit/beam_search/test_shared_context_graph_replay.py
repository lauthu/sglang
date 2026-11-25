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
"""Shared-context beam graphs replay the batch loaded into their shared buffers."""

import unittest
from types import SimpleNamespace

import torch

from sglang.kernels.ops.attention.beam_shared_context import shared_context_attention
from sglang.srt.beam_search.beam_decode_cuda_graph_runner import (
    BeamGraphInputs,
    FixedWidthCapacity,
    FixedWidthSharedContextInput,
    TrieLevelsInput,
    VariableWidthSharedContextInput,
)
from sglang.srt.beam_search.forward_inputs import BeamForwardInputs
from sglang.srt.beam_search.shared_context_attention import (
    BeamAttentionBucket,
    BeamAttentionMetadata,
)
from sglang.srt.beam_search.shared_context_variable_width import build_variable_width_metadata
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.beam_shared_context import (
    run_variable_width_attention,
    variable_width_case,
)
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=40, stage="base-b", runner_config="1-gpu-small")

WIDTH = 8
CAPACITY = FixedWidthCapacity(beam_width=WIDTH, max_context_len=8, max_decode_len=3)
# A KV slot no loaded batch uses: reading it changes the result.
STALE_SLOT = 63


def _attention(q, k, v, bucket):
    return shared_context_attention(
        q,
        k,
        v,
        bucket.row_indices,
        bucket.context_slots,
        bucket.context_cu_seqlens,
        bucket.decode_slots,
        bucket.decode_lens,
        softmax_scale=0.125,
    )


def _batch(groups, context, prompt_offsets, decode_lens, *, flip):
    """A batch of `groups` groups whose decode slots start at KV slot 8."""
    rows = torch.arange(groups * WIDTH, dtype=torch.int32, device="cuda").view(
        groups, WIDTH
    )
    decode = 8 + torch.arange(
        groups * WIDTH * max(decode_lens), dtype=torch.int32, device="cuda"
    ).view(groups, -1)
    bucket = BeamAttentionBucket(
        row_indices=rows.flip((0, 1)) if flip else rows,
        context_slots=torch.tensor(context, dtype=torch.int32, device="cuda"),
        context_cu_seqlens=torch.tensor(
            prompt_offsets, dtype=torch.int32, device="cuda"
        ),
        decode_slots=decode.flip(1) if flip else decode,
        beam_width=WIDTH,
        decode_lens=torch.tensor(decode_lens, dtype=torch.int32, device="cuda"),
    )
    return BeamAttentionMetadata(buckets=[bucket], num_rows=groups * WIDTH)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestSharedContextGraphReplay(CustomTestCase):
    def test_graphs_of_two_sizes_replay_the_loaded_batch(self):
        """Graphs of every size read one set of buffers. Each replay must follow
        the lengths, slots and row mapping of the batch loaded for it, and read
        no slot past them, where another batch's values remain."""
        torch.manual_seed(17)
        q = torch.randn((2 * WIDTH, 4, 64), dtype=torch.bfloat16, device="cuda")
        k = torch.randn((64, 2, 64), dtype=torch.bfloat16, device="cuda")
        v = torch.randn_like(k)
        inputs = BeamGraphInputs(
            trie_levels_input=None,
            shared_context_input=FixedWidthSharedContextInput(
                max_rows=2 * WIDTH, device="cuda", capacity=CAPACITY
            ),
        )

        graphs = {}
        for rows in (2 * WIDTH, WIDTH):
            captured = inputs.prepare_capture_inputs(rows)
            graph_bucket = captured.attention_metadata.buckets[0]
            _attention(q[:rows], k, v, graph_bucket)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output = _attention(q[:rows], k, v, graph_bucket)
            graphs[rows] = (graph, output, graph_bucket)

        batches = (
            _batch(2, list(range(8)), [0, 5, 8], [1, 2], flip=False),
            _batch(1, [5, 4, 3], [0, 3], [3], flip=True),
            _batch(2, [7, 6, 5], [0, 2, 3], [2, 1], flip=True),
            _batch(1, [1, 2], [0, 2], [1], flip=False),
        )
        for metadata in batches:
            rows = metadata.num_rows
            graph, output, graph_bucket = graphs[rows]
            graph_bucket.context_slots.fill_(STALE_SLOT)
            graph_bucket.decode_slots.fill_(STALE_SLOT)
            inputs.load_replay_inputs(BeamForwardInputs(None, metadata))
            graph.replay()
            torch.testing.assert_close(
                output.float(),
                _attention(q[:rows], k, v, metadata.buckets[0]).float(),
                rtol=1e-2,
                atol=1e-2,
            )


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestVariableWidthGraphReplay(CustomTestCase):
    def test_graph_captured_once_replays_any_widths(self):
        """Capture records no beam width, group count or length: one beam graph
        replays every batch loaded into its layout, and the padding rows after
        a batch's read 0. With stream-K, the batches alternate between splitting tasks and
        whole tasks per CTA (the batch of 40 groups)."""
        batches = (
            dict(widths=(256, 256), prompt_lens=(2048, 7), decode_lens=(2, 1)),
            dict(
                widths=(2, 129, 300), prompt_lens=(5, 1000, 64), decode_lens=(3, 1, 2)
            ),
            dict(widths=(2,) * 40, prompt_lens=(64,) * 40, decode_lens=(1,) * 40),
            dict(widths=(7,), prompt_lens=(33,), decode_lens=(4,)),
            dict(
                widths=(2, 256, 3, 2),
                prompt_lens=(900, 2048, 3000, 1),
                decode_lens=(2, 1, 1, 6),
            ),
        )
        cases = tuple(
            variable_width_case(**batch, max_context=4096, seed=seed)
            for seed, batch in enumerate(batches)
        )
        for stream_k in (False, True):
            with self.subTest(stream_k=stream_k):
                self._replay_batches_in_one_graph(cases, stream_k=stream_k)

    def _replay_batches_in_one_graph(self, cases, *, stream_k):
        graph_size = 512
        first = cases[0]
        # The tensors the graph is captured on; each replay copies a case into them.
        with torch.device("cuda"):
            graph_tensors = SimpleNamespace(
                q=torch.zeros(graph_size, *first.q.shape[1:], dtype=torch.bfloat16),
                k_buffer=torch.zeros_like(first.k_buffer),
                v_buffer=torch.zeros_like(first.v_buffer),
                req_pool_indices=torch.zeros(graph_size, dtype=torch.int64),
                seq_lens=torch.ones(graph_size, dtype=torch.int64),
                softmax_scale=first.softmax_scale,
            )
            req_to_token = torch.zeros(
                (len(first.req_to_token) * 2, first.req_to_token.shape[1]),
                dtype=torch.int32,
            )
        beam_graph_inputs = BeamGraphInputs(
            trie_levels_input=TrieLevelsInput(max_rows=graph_size, device="cuda"),
            shared_context_input=VariableWidthSharedContextInput(
                max_rows=graph_size,
                device=req_to_token.device,
                req_to_token=req_to_token,
                stream_k=stream_k,
            ),
        )
        graph_metadata = beam_graph_inputs.prepare_capture_inputs(
            graph_size
        ).attention_metadata
        # Compile the kernels outside the capture.
        run_variable_width_attention(graph_tensors, graph_metadata)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = run_variable_width_attention(graph_tensors, graph_metadata)

        for index, case in enumerate(cases):
            with self.subTest(batch=index):
                num_rows = case.q.shape[0]
                graph_tensors.q[:num_rows].copy_(case.q)
                graph_tensors.k_buffer.copy_(case.k_buffer)
                graph_tensors.v_buffer.copy_(case.v_buffer)
                graph_tensors.req_pool_indices[:num_rows].copy_(case.req_pool_indices)
                graph_tensors.seq_lens[:num_rows].copy_(case.seq_lens)
                req_to_token[: len(case.req_to_token)].copy_(case.req_to_token)
                batch_inputs = BeamForwardInputs(
                    trie_levels=torch.zeros(num_rows, dtype=torch.int64, device="cuda"),
                    attention_metadata=build_variable_width_metadata(
                        batch=case.batch,
                        req_to_token=case.req_to_token,
                        stream_k=stream_k,
                    ),
                )
                self.assertTrue(
                    beam_graph_inputs.can_replay(batch_inputs, graph_size=graph_size)
                )
                beam_graph_inputs.load_replay_inputs(batch_inputs)
                graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(
                    output[:num_rows].float(), case.reference, rtol=1e-2, atol=1e-2
                )
                padding_rows = output[num_rows:]
                self.assertEqual(int(padding_rows.abs().sum()), 0)


if __name__ == "__main__":
    unittest.main()
