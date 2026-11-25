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
"""Shared-context beam graphs replay the batch loaded into their shared buffers,
and the trie head keeps regular requests' full-vocabulary rows on the way."""

import unittest
from types import SimpleNamespace

import torch

from sglang.kernels.ops.attention.beam_shared_context import shared_context_attention
from sglang.srt.beam_search.beam_decode_cuda_graph_runner import (
    BeamGraphInputs,
    FixedWidthCapacity,
    FixedWidthSharedContextInput,
    TrieLevelsInput,
    WidthFreeSharedContextInput,
)
from sglang.srt.beam_search.forward_inputs import BeamForwardInputs
from sglang.srt.beam_search.mixed_batch import BeamDenseRows
from sglang.srt.beam_search.shared_context_attention import (
    BeamAttentionBucket,
    BeamAttentionMetadata,
)
from sglang.srt.beam_search.shared_context_ragged import (
    RAGGED_BLOCK_M,
    build_ragged_metadata,
)
from sglang.srt.beam_search.trie_config import TrieOutputHeadConfig
from sglang.srt.beam_search.trie_output_head import BeamTrieOutputHead
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.kernels.beam_shared_context import ragged_case, run_ragged
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
class TestWidthFreeGraphReplay(CustomTestCase):
    def test_graph_captured_once_replays_any_split(self):
        """Nothing about widths, group count or lengths is recorded at capture: one
        beam graph replays every split loaded into its layout, regular requests as
        one-row groups included, and padded rows read 0. With stream-K, the batches
        alternate between splitting the units and whole tiles per CTA (40 tiles)."""
        cases = tuple(
            ragged_case(*split, max_context=4096, seed=seed)
            for seed, split in enumerate(
                (
                    ((256, 256), (2048, 7), (2, 1)),
                    ((2, 129, 300), (5, 1000, 64), (3, 1, 2)),
                    ((2,) * 40, (64,) * 40, (1,) * 40),
                    ((7,), (33,), (4,)),
                    ((1, 256, 1, 1), (900, 2048, 3000, 1), (2, 1, 0, 6)),
                )
            )
        )
        for stream_k in (False, True):
            with self.subTest(stream_k=stream_k):
                self._replay_splits_in_one_graph(cases, stream_k=stream_k)

    def _replay_splits_in_one_graph(self, cases, *, stream_k):
        graph_size = 512
        first = cases[0]
        # The graph's own inputs; each replay stages a case into them.
        with torch.device("cuda"):
            graph_inputs = SimpleNamespace(
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
        inputs = BeamGraphInputs(
            trie_levels_input=TrieLevelsInput(max_rows=graph_size, device="cuda"),
            shared_context_input=WidthFreeSharedContextInput(
                max_rows=graph_size,
                device=req_to_token.device,
                req_to_token=req_to_token,
                allow_one_row_groups=True,
                stream_k=stream_k,
            ),
        )
        captured = inputs.prepare_capture_inputs(graph_size).attention_metadata
        run_ragged(graph_inputs, captured, block_m=RAGGED_BLOCK_M)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = run_ragged(graph_inputs, captured, block_m=RAGGED_BLOCK_M)
        for index, case in enumerate(cases):
            with self.subTest(split=index):
                rows = case.q.shape[0]
                graph_inputs.q[:rows].copy_(case.q)
                graph_inputs.k_buffer.copy_(case.k_buffer)
                graph_inputs.v_buffer.copy_(case.v_buffer)
                graph_inputs.req_pool_indices[:rows].copy_(case.req_pool_indices)
                graph_inputs.seq_lens[:rows].copy_(case.seq_lens)
                req_to_token[: len(case.req_to_token)].copy_(case.req_to_token)
                batch_inputs = BeamForwardInputs(
                    trie_levels=torch.zeros(rows, dtype=torch.int64, device="cuda"),
                    attention_metadata=build_ragged_metadata(
                        batch=case.batch,
                        req_to_token=case.req_to_token,
                        allow_one_row_groups=True,
                        stream_k=stream_k,
                    ),
                )
                self.assertTrue(inputs.can_replay(batch_inputs, graph_size=graph_size))
                inputs.load_replay_inputs(batch_inputs)
                graph.replay()
                torch.cuda.synchronize()
                torch.testing.assert_close(
                    output[:rows].float(), case.reference, rtol=1e-2, atol=1e-2
                )
                self.assertEqual(int(output[rows:].abs().sum()), 0)


@unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
class TestTrieHeadKeptRows(CustomTestCase):
    def test_trie_head_keeps_every_chunk_in_its_columns(self):
        """On CUDA the trie head writes each vocabulary chunk's transformed logits
        into that chunk's columns of the kept rows, and keeping leaves its
        normalizer as is."""
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
                project(
                    hidden_states, weight[start : min(start + chunk_size, vocab_size)]
                )
            )
            for start in range(0, vocab_size, chunk_size)
        ]
        reference = torch.cat(chunks, dim=1).float()
        self.assertTrue(torch.equal(kept.dense_logits, reference[:4]))
        self.assertTrue(torch.equal(kept.normalizer, plain.normalizer))


if __name__ == "__main__":
    unittest.main()
