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
"""What a beam decode graph replays: the batch loaded into its buffers, and only
a batch that fits them."""

import contextlib
import itertools
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import msgspec
import torch

from sglang.srt.beam_search import (
    beam_decode_cuda_graph_runner,
    decode_graph_settings,
    mixed_batch,
    shared_context_attention,
    trie_output_head,
)
from sglang.srt.beam_search.batch_tail import BeamTailEntry
from sglang.srt.beam_search.beam_decode_cuda_graph_runner import (
    BeamGraphInputs,
    DenseRowsInput,
    FixedWidthCapacity,
    FixedWidthSharedContextInput,
    TrieLevelsInput,
    WidthFreeSharedContextInput,
    _create_beam_graph_inputs,
)
from sglang.srt.beam_search.forward_inputs import (
    BeamForwardInputs,
    build_beam_forward_inputs,
)
from sglang.srt.beam_search.logits_capture import capture_pre_sample_logits
from sglang.srt.beam_search.mixed_batch import BeamDenseRows
from sglang.srt.beam_search.shared_context_attention import (
    BeamAttentionBucket,
    BeamAttentionMetadata,
)
from sglang.srt.beam_search.shared_context_ragged import RAGGED_BLOCK_M
from sglang.srt.beam_search.trie_config import TrieOutputHeadConfig
from sglang.srt.beam_search.trie_output_head import BeamTrieOutputHead
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

CAPACITY = FixedWidthCapacity(beam_width=2, max_context_len=4, max_decode_len=3)


def _inputs(max_rows, *, trie_head=True, shared_context=True) -> BeamGraphInputs:
    return BeamGraphInputs(
        trie_levels_input=(
            TrieLevelsInput(max_rows=max_rows, device="cpu") if trie_head else None
        ),
        shared_context_input=(
            FixedWidthSharedContextInput(
                max_rows=max_rows, device="cpu", capacity=CAPACITY
            )
            if shared_context
            else None
        ),
    )


def _metadata(prompt_lens, decode_lens, width=2) -> BeamAttentionMetadata:
    """One bucket as build_metadata lays it out, with distinct slot numbers."""
    groups, steps = len(prompt_lens), max(decode_lens)
    return BeamAttentionMetadata(
        num_rows=groups * width,
        buckets=[
            BeamAttentionBucket(
                row_indices=torch.arange(groups * width, dtype=torch.int32)
                .view(groups, width)
                .flip(1),
                context_slots=100 + torch.arange(sum(prompt_lens), dtype=torch.int32),
                context_cu_seqlens=torch.tensor(
                    [0, *itertools.accumulate(prompt_lens)], dtype=torch.int32
                ),
                decode_slots=200
                + torch.arange(groups * width * steps, dtype=torch.int32).view(
                    groups, width * steps
                ),
                beam_width=width,
                decode_lens=torch.tensor(decode_lens, dtype=torch.int32),
            )
        ],
    )


class TestBeamGraphInputs(CustomTestCase):
    def test_loading_refreshes_every_tensor_the_graph_reads(self):
        """A graph reads its inputs by address: a batch tensor that loading
        skips leaves the previous batch's values in the replay. Under the dense
        LM head a batch carries no trie levels and replays all the same."""
        for trie_head in (True, False):
            with self.subTest(trie_head=trie_head):
                inputs = _inputs(8, trie_head=trie_head)
                captured = inputs.prepare_capture_inputs(4)
                batch = BeamForwardInputs(
                    torch.tensor([1, 2, 1, 2]) if trie_head else None,
                    _metadata(prompt_lens=[1, 2], decode_lens=[1, 2]),
                )
                self.assertTrue(inputs.can_replay(batch, graph_size=4))
                inputs.load_replay_inputs(batch)

                if trie_head:
                    self.assertEqual(captured.trie_levels.tolist(), [1, 2, 1, 2])
                captured_bucket = captured.attention_metadata.buckets[0]
                batch_bucket = batch.attention_metadata.buckets[0]
                tensors = 0
                for field in msgspec.structs.fields(BeamAttentionBucket):
                    value = getattr(batch_bucket, field.name)
                    if not isinstance(value, torch.Tensor):
                        continue
                    tensors += 1
                    loaded = getattr(captured_bucket, field.name)[
                        tuple(map(slice, value.shape))
                    ]
                    self.assertEqual(loaded.tolist(), value.tolist(), field.name)
                self.assertEqual(tensors, 5)

    def test_only_a_batch_that_fits_the_graph_replays(self):
        """Replaying a beam graph for a batch that does not fit its buffers
        returns the attention of other slots; such a batch runs eagerly."""
        inputs = _inputs(8)
        inputs.prepare_capture_inputs(4)
        levels = torch.zeros(4, dtype=torch.int64)
        at_capacity = _metadata(
            prompt_lens=[CAPACITY.max_context_len] * 2,
            decode_lens=[CAPACITY.max_decode_len, 1],
        )
        self.assertTrue(
            inputs.can_replay(BeamForwardInputs(levels, at_capacity), graph_size=4)
        )

        two_widths = _metadata(prompt_lens=[1, 1], decode_lens=[1, 1])
        two_widths.buckets.append(two_widths.buckets[0])
        rejected = {
            "no graph of this size": (
                torch.zeros(8, dtype=torch.int64),
                _metadata(prompt_lens=[1] * 4, decode_lens=[1] * 4),
                8,
            ),
            "fewer rows than the graph": (
                levels,
                _metadata(prompt_lens=[1], decode_lens=[1]),
                4,
            ),
            "decode steps past the capacity": (
                levels,
                _metadata(
                    prompt_lens=[1, 1], decode_lens=[CAPACITY.max_decode_len + 1, 1]
                ),
                4,
            ),
            "another beam width": (
                levels,
                _metadata(prompt_lens=[1], decode_lens=[1], width=4),
                4,
            ),
            "two widths in the batch": (levels, two_widths, 4),
            "ordinary attention": (levels, None, 4),
        }
        for name, (trie_levels, metadata, graph_size) in rejected.items():
            with self.subTest(name):
                self.assertFalse(
                    inputs.can_replay(
                        BeamForwardInputs(trie_levels, metadata),
                        graph_size=graph_size,
                    )
                )

    def test_ordinary_attention_graph_takes_no_shared_context_batch(self):
        """Without shared-context attention a beam graph holds FlashInfer
        attention, which a batch with shared-context metadata does not run."""
        inputs = _inputs(4, shared_context=False)
        inputs.prepare_capture_inputs(4)
        levels = torch.zeros(3, dtype=torch.int64)
        self.assertTrue(
            inputs.can_replay(BeamForwardInputs(levels, None), graph_size=4)
        )
        shared = _metadata(prompt_lens=[1, 1], decode_lens=[1, 1])
        self.assertFalse(
            inputs.can_replay(BeamForwardInputs(levels, shared), graph_size=4)
        )


class TestFixedWidthSharedContextInput(CustomTestCase):
    def test_a_smaller_graph_captures_on_its_own_values(self):
        """Graphs of every size share the buffers, and a graph of fewer groups
        reads them at other offsets. What a larger graph's batch left there is
        no valid input: a length past the capacity reads outside the buffers."""
        shared_context_input = FixedWidthSharedContextInput(
            max_rows=8, device="cpu", capacity=CAPACITY
        )
        shared_context_input.prepare_capture_inputs(8)
        shared_context_input.load_replay_inputs(
            _metadata(
                prompt_lens=[CAPACITY.max_context_len] * 4,
                decode_lens=[CAPACITY.max_decode_len] * 4,
            )
        )
        bucket = shared_context_input.prepare_capture_inputs(4).buckets[0]
        self.assertEqual(sorted(bucket.row_indices.flatten().tolist()), [0, 1, 2, 3])
        prompt_offsets = bucket.context_cu_seqlens.tolist()
        self.assertEqual(prompt_offsets[0], 0)
        self.assertEqual(prompt_offsets, sorted(prompt_offsets))
        self.assertLessEqual(prompt_offsets[-1], bucket.context_slots.numel())
        for decode_len in bucket.decode_lens.tolist():
            self.assertIn(decode_len, range(1, CAPACITY.max_decode_len + 1))

    def test_graphs_need_a_width_and_a_token_budget(self):
        """The graph is captured for one beam width and a fixed number of decode
        steps. Nothing but the operator knows the width, and without a trie
        nothing bounds the steps, so a server that sets neither captures no graph.
        A step attends the tokens generated before it: one fewer than the budget."""
        runner = SimpleNamespace(
            lora_manager=None,
            device="cpu",
            model_config=SimpleNamespace(context_len=16),
        )
        cases = (
            # (width, max new tokens, trie codebooks) -> max_decode_len, or None for no graph
            ((None, 4, None), None),
            ((2, None, None), None),
            ((2, 4, None), 3),
            ((2, None, 3), 2),
            ((2, 8, 3), 2),
        )
        for (width, max_new_tokens, codebooks), max_decode_len in cases:
            trie_config = (
                None if codebooks is None else SimpleNamespace(num_codebooks=codebooks)
            )
            with (
                self.subTest(
                    width=width, max_new_tokens=max_new_tokens, codebooks=codebooks
                ),
                patch.object(
                    beam_decode_cuda_graph_runner,
                    "get_beam_trie_config",
                    return_value=trie_config,
                ),
                envs.SGLANG_BEAM_DECODE_GRAPH_FIXED_WIDTH.override(width),
                envs.SGLANG_BEAM_DECODE_GRAPH_FIXED_WIDTH_MAX_NEW_TOKENS.override(
                    max_new_tokens
                ),
            ):
                attention = FixedWidthSharedContextInput.create(runner, max_rows=8)
                if max_decode_len is None:
                    self.assertIsNone(attention)
                else:
                    self.assertEqual(
                        attention.capacity,
                        FixedWidthCapacity(
                            beam_width=2,
                            max_context_len=16,
                            max_decode_len=max_decode_len,
                        ),
                    )


_TRIE_CONFIG = TrieOutputHeadConfig(
    tensor_path="/unused/trie_output_head.safetensors",
    token_start=8,
    codebook_size=4,
    num_codebooks=2,
)


def _model_runner(compact_head=True, lora=False):
    """The model runner fields the forward-input builders and the beam graph
    inputs read."""
    return SimpleNamespace(
        beam_trie_compact_enabled=compact_head,
        lora_manager=object() if lora else None,
        model_config=SimpleNamespace(context_len=8, vocab_size=16),
        req_to_token_pool=SimpleNamespace(
            req_to_token=torch.zeros(4, 8, dtype=torch.int32)
        ),
        device=torch.device("cpu"),
        max_decode_logits_rows=lambda: 600,
    )


@contextlib.contextmanager
def _beam_decode_graphs(*, fixed_width=None, mixed_batches=False, **env):
    """A server with the beam decode graphs on, and shared-context attention on
    and supported. Without a fixed width its graphs are the width-free ones."""
    with contextlib.ExitStack() as stack:
        overrides = {
            "SGLANG_ENABLE_BEAM_SHARED_CONTEXT_ATTENTION": True,
            "SGLANG_ENABLE_BEAM_DECODE_GRAPH": True,
            "SGLANG_BEAM_DECODE_GRAPH_FIXED_WIDTH": fixed_width,
            "SGLANG_ENABLE_BEAM_DECODE_GRAPH_MIXED_BATCH": mixed_batches,
            **env,
        }
        for name, value in overrides.items():
            stack.enter_context(getattr(envs, name).override(value))
        stack.enter_context(
            patch.object(
                shared_context_attention, "supports_shared_context", return_value=True
            )
        )
        stack.enter_context(
            patch.object(
                trie_output_head, "get_beam_trie_config", return_value=_TRIE_CONFIG
            )
        )
        yield


def _decode_batch(widths, prompt_len=7, decode_len=2):
    """A decode batch with one base row per request, in order, then each live beam
    group's member rows. A width of 1 is a regular request with its own KV; 0 is
    a finished group's leader, which has no member rows left."""
    reqs, entries, base_lens, member_lens = [], [], [], []
    for row, width in enumerate(widths):
        if width == 1:
            reqs.append(SimpleNamespace(beam_group=None))
            base_lens.append(40 + row)
            continue
        group = SimpleNamespace(prompt_len=prompt_len, num_generated=decode_len - 1)
        reqs.append(SimpleNamespace(beam_group=group))
        base_lens.append(prompt_len + decode_len)
        if width > 1:
            start = len(member_lens)
            entries.append(BeamTailEntry(group, row, start, start + width - 1))
            member_lens.extend([prompt_len + decode_len] * (width - 1))
    seq_lens = torch.tensor(base_lens + member_lens, dtype=torch.int64)
    return SimpleNamespace(
        reqs=reqs,
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens,
        beam_tail=SimpleNamespace(num_base_rows=len(reqs), entries=entries),
        forward_mode=SimpleNamespace(is_decode=lambda: True),
        device=torch.device("cpu"),
    )


def _width_free_inputs(max_rows, *, mixed_batches=False, trie_head=True):
    keeps_dense_rows = trie_head and mixed_batches
    return BeamGraphInputs(
        trie_levels_input=(
            TrieLevelsInput(max_rows=max_rows, device="cpu") if trie_head else None
        ),
        shared_context_input=WidthFreeSharedContextInput(
            max_rows=max_rows,
            device="cpu",
            req_to_token=torch.zeros(4, 8, dtype=torch.int32),
            allow_one_row_groups=keeps_dense_rows,
            stream_k=False,
        ),
        dense_rows_input=(
            DenseRowsInput(max_rows=max_rows, vocab_size=16, device="cpu")
            if keeps_dense_rows
            else None
        ),
    )


def _tables(layout):
    """A layout buffer's counts and its loaded group and tile tables."""
    num_groups, num_tiles, num_rows = layout[:3].tolist()
    groups = layout[3 : 3 + 4 * num_groups].view(-1, 4).tolist()
    tiles = layout[3 + 4 * num_groups :][: 2 * num_tiles].view(-1, 2).tolist()
    return num_rows, groups, tiles


def _rows_in_tiles(layout):
    """Every row the loaded tiles cover, tile by tile."""
    _, groups, tiles = _tables(layout)
    rows = []
    for group, first_beam in tiles:
        leader, first_member, members, _ = groups[group]
        for beam in range(first_beam, min(first_beam + RAGGED_BLOCK_M, members + 1)):
            rows.append(leader if beam == 0 else first_member + beam - 1)
    return rows


class TestWidthFreeBeamGraphInputs(CustomTestCase):
    def test_settings_decide_what_the_graphs_hold(self):
        """Graphs hold the fixed-width shared-context input, which limits them to whole
        groups of that width, only when a width is set; otherwise they hold the
        width-free one and take every size. Rows are kept only for mixed batches
        and where the compact head fills them, and LoRA, whose graph key slot the
        beam label takes, leaves a shared-context server without beam graphs."""
        # (fixed width, mixed batches, compact head, LoRA):
        # (shared-context input, kept rows), None for no graphs
        cases = {
            (2, False, True, False): (FixedWidthSharedContextInput, False),
            (None, False, True, False): (WidthFreeSharedContextInput, False),
            (None, True, True, False): (WidthFreeSharedContextInput, True),
            (None, True, False, False): (WidthFreeSharedContextInput, False),
            (None, False, True, True): None,
        }
        for (fixed_width, mixed, compact_head, lora), holds in cases.items():
            with self.subTest(
                fixed_width=fixed_width,
                mixed_batches=mixed,
                compact_head=compact_head,
                lora=lora,
            ):
                with (
                    _beam_decode_graphs(fixed_width=fixed_width, mixed_batches=mixed),
                    patch.object(
                        beam_decode_cuda_graph_runner,
                        "get_beam_trie_config",
                        return_value=_TRIE_CONFIG,
                    ),
                ):
                    inputs = _create_beam_graph_inputs(
                        _model_runner(compact_head=compact_head, lora=lora)
                    )
                if holds is None:
                    self.assertIsNone(inputs)
                    continue
                attention, keeps_rows = holds
                self.assertIs(type(inputs.shared_context_input), attention)
                self.assertEqual(inputs.dense_rows_input is not None, keeps_rows)
                self.assertEqual(inputs.trie_levels_input is not None, compact_head)
                self.assertEqual(inputs.can_capture(3), fixed_width is None)

    def test_only_graphs_without_a_fixed_width_run_the_width_free_layout(self):
        """A server keeps the per-width buckets, and with them its kernel, unless
        its beam decode graphs take any width: with the graphs off, or with a
        fixed width, a batch must not get the width-free layout."""
        batch, runner = _decode_batch((4, 4)), _model_runner()
        buckets = object()
        cases = (
            ({"SGLANG_ENABLE_BEAM_DECODE_GRAPH": False}, False),
            ({"fixed_width": 4}, False),
            ({}, True),
        )
        for settings, width_free in cases:
            with (
                self.subTest(settings=settings),
                _beam_decode_graphs(**settings),
                patch.object(
                    shared_context_attention,
                    "_build_width_buckets",
                    return_value=buckets,
                ),
            ):
                metadata = shared_context_attention.build_metadata(batch, runner)
                self.assertEqual(metadata is not buckets, width_free)

    def test_any_split_loads_into_a_larger_graph(self):
        """Width-free graphs take a batch of any widths and fewer rows than they
        hold: over a larger earlier batch's tables, the loaded tiles must cover
        each of its rows exactly once."""
        with _beam_decode_graphs():
            inputs = _width_free_inputs(max_rows=600)
            self.assertTrue(inputs.can_capture(431))
            captured = inputs.prepare_capture_inputs(512)
            runner = _model_runner()
            inputs.load_replay_inputs(
                build_beam_forward_inputs(_decode_batch((300, 300)), runner)
            )
            beam = build_beam_forward_inputs(_decode_batch((2, 129, 300)), runner)
            self.assertEqual(beam.trie_levels.shape[0], 431)
            self.assertIsNone(beam.dense_rows)
            self.assertTrue(inputs.can_replay(beam, graph_size=512))
            inputs.load_replay_inputs(beam)

            layout = captured.attention_metadata.layout
            num_rows, groups, _ = _tables(layout)
            self.assertEqual((num_rows, len(groups)), (431, 3))
            self.assertEqual(sorted(_rows_in_tiles(layout)), list(range(431)))

    def test_dense_head_batch_replays_without_trie_levels(self):
        """With the compact trie head off a beam batch carries its width-free
        layout and no trie levels; a graph captured without levels replays it."""
        with _beam_decode_graphs():
            inputs = _width_free_inputs(max_rows=600, trie_head=False)
            captured = inputs.prepare_capture_inputs(512)
            self.assertIsNone(captured.trie_levels)
            beam = build_beam_forward_inputs(
                _decode_batch((2, 129, 300)), _model_runner(compact_head=False)
            )
            self.assertIsNone(beam.trie_levels)
            self.assertTrue(inputs.can_replay(beam, graph_size=512))
            inputs.load_replay_inputs(beam)
            self.assertEqual(
                sorted(_rows_in_tiles(captured.attention_metadata.layout)),
                list(range(431)),
            )

    def test_width_free_graph_takes_no_ordinary_attention_batch(self):
        """A batch the layout builder turned away (here a finished group's leader
        still in it) carries no tables; replaying it would attend the previous
        batch's groups."""
        with _beam_decode_graphs():
            inputs = _width_free_inputs(max_rows=600)
            inputs.prepare_capture_inputs(512)
            beam = build_beam_forward_inputs(
                _decode_batch((129, 0, 300)), _model_runner()
            )
            self.assertIsNotNone(beam.trie_levels)
            self.assertIsNone(beam.attention_metadata)
            self.assertFalse(inputs.can_replay(beam, graph_size=512))

    def test_startup_rejects_settings_that_cannot_run_together(self):
        """The stream-K kernel and mixed batches exist only on the width-free
        graphs, and a mixed batch cannot keep deterministic inference: none may
        silently run something else."""
        check = decode_graph_settings.check_settings
        with _beam_decode_graphs(SGLANG_BEAM_SHARED_CONTEXT_KERNEL="stream_k"):
            check(deterministic_inference=True)
        for without_width_free_graphs in (
            {"fixed_width": 2},
            {"SGLANG_ENABLE_BEAM_DECODE_GRAPH": False},
        ):
            with _beam_decode_graphs(
                SGLANG_BEAM_SHARED_CONTEXT_KERNEL="stream_k",
                **without_width_free_graphs,
            ):
                with self.assertRaisesRegex(ValueError, "FIXED_WIDTH unset"):
                    check(deterministic_inference=False)
            with _beam_decode_graphs(mixed_batches=True, **without_width_free_graphs):
                with self.assertRaisesRegex(ValueError, "FIXED_WIDTH unset"):
                    check(deterministic_inference=False)
        with _beam_decode_graphs(SGLANG_BEAM_SHARED_CONTEXT_KERNEL="split_kv"):
            with self.assertRaisesRegex(ValueError, "not one of"):
                check(deterministic_inference=False)
        # A mixed batch moves a regular request's logits with the batch it decodes in.
        with _beam_decode_graphs(mixed_batches=True):
            check(deterministic_inference=False)
            with self.assertRaisesRegex(ValueError, "deterministic"):
                check(deterministic_inference=True)


class TestMixedBatches(CustomTestCase):
    """SGLANG_ENABLE_BEAM_DECODE_GRAPH_MIXED_BATCH: regular requests join beam
    decode batches."""

    def test_regular_requests_join_only_with_mixed_batches_on(self):
        """With the switch off, or without the compact head that keeps its logits,
        a batch with a regular request gets no beam inputs, so it runs the
        ordinary decode path, uncounted. With it on, its regular rows and a
        finished group's leader are one-row groups reading their whole KV as the
        prompt, the tiles cover every row once, and the graph keeps one
        full-vocabulary row per request, and none in a following beam-only batch."""
        runner = _model_runner()
        batch = _decode_batch((1, 129, 0, 1, 300))
        rows = len(batch.seq_lens)
        with _beam_decode_graphs():
            oversized = mixed_batch.num_oversized_batches()
            self.assertIsNone(build_beam_forward_inputs(batch, runner))
            self.assertEqual(mixed_batch.num_oversized_batches(), oversized)
        with _beam_decode_graphs(mixed_batches=True):
            self.assertIsNone(
                build_beam_forward_inputs(batch, _model_runner(compact_head=False))
            )
            inputs = _width_free_inputs(max_rows=600, mixed_batches=True)
            captured = inputs.prepare_capture_inputs(512)

            beam = build_beam_forward_inputs(batch, runner)
            self.assertEqual(beam.trie_levels[:5].tolist(), [0, 1, 1, 0, 1])
            self.assertTrue(inputs.can_replay(beam, graph_size=512))
            inputs.load_replay_inputs(beam)

            layout = captured.attention_metadata.layout
            num_rows, groups, _ = _tables(layout)
            self.assertEqual((num_rows, len(groups)), (rows, 5))
            for leader, _, members, prompt_len in groups:
                if members == 0:
                    self.assertEqual(prompt_len, int(batch.seq_lens[leader]))
            self.assertEqual(sorted(_rows_in_tiles(layout)), list(range(rows)))
            self.assertEqual(int(captured.dense_rows.count), 5)

            # The kept-rows part exists, the batch carries no rows to keep: it replays.
            beam_only = build_beam_forward_inputs(_decode_batch((129, 300)), runner)
            self.assertIsNone(beam_only.dense_rows)
            self.assertTrue(inputs.can_replay(beam_only, graph_size=512))
            inputs.load_replay_inputs(beam_only)
            self.assertEqual(int(captured.dense_rows.count), 0)

    def test_too_many_requests_take_the_ordinary_path(self):
        """Above SGLANG_BEAM_DECODE_GRAPH_MIXED_BATCH_MAX_REQUESTS requests, a batch
        with a regular request gets no beam inputs, so it runs the ordinary decode
        path, and is counted. At the cap it runs the beam path; beam-only batches
        never count against the cap."""
        runner = _model_runner()
        with _beam_decode_graphs(
            mixed_batches=True, SGLANG_BEAM_DECODE_GRAPH_MIXED_BATCH_MAX_REQUESTS=3
        ):
            at_cap = build_beam_forward_inputs(_decode_batch((1, 4, 1)), runner)
            self.assertEqual(at_cap.trie_levels[:3].tolist(), [0, 1, 0])
            self.assertEqual(len(_tables(at_cap.attention_metadata.layout)[1]), 3)
            self.assertEqual(at_cap.dense_rows.num_rows, 3)

            oversized = mixed_batch.num_oversized_batches()
            self.assertIsNone(
                build_beam_forward_inputs(_decode_batch((1, 4, 1, 4)), runner)
            )
            self.assertEqual(mixed_batch.num_oversized_batches(), oversized + 1)

            beam_only = build_beam_forward_inputs(_decode_batch((4, 4, 4, 4)), runner)
            self.assertEqual(len(_tables(beam_only.attention_metadata.layout)[1]), 4)
            self.assertIsNone(beam_only.dense_rows)
            self.assertEqual(mixed_batch.num_oversized_batches(), oversized + 1)

    def test_sampler_reads_kept_rows_and_selection_reads_compact_rows(self):
        """With regular requests in a beam batch, the sampler must get the kept
        full-vocabulary rows, one per request, while beam selection keeps the
        leaders' and members' compact rows. A beam-only batch replays the same
        graph and must not sample from the kept-row buffer it returns, and a head
        that fell back to the dense LM head keeps no rows to sample from."""
        batch = _decode_batch((1, 3, 1))
        compact = torch.randn(5, 4)
        dense = torch.randn(64, 16)

        def capture(dense_rows, kept_logits=dense):
            output = SimpleNamespace(
                next_token_logits=compact,
                beam_dense_logits=kept_logits,
                beam_normalizer=torch.randn(5, 2),
                hidden_states=None,
                beam=None,
            )
            forward = SimpleNamespace(
                beam=BeamForwardInputs(torch.zeros(5), None, dense_rows),
                positions=torch.arange(5),
            )
            capture_pre_sample_logits(batch, forward, output)
            return output

        output = capture(BeamDenseRows(num_rows=3))
        self.assertTrue(torch.equal(output.next_token_logits, dense[:3]))
        self.assertTrue(torch.equal(output.beam.leader_logits, compact[1:2]))
        self.assertTrue(torch.equal(output.beam.tail_logits, compact[3:]))
        self.assertTrue(torch.equal(capture(None).next_token_logits, compact[:3]))
        fallback = capture(BeamDenseRows(num_rows=3), kept_logits=None)
        self.assertTrue(torch.equal(fallback.next_token_logits, compact[:3]))

    def test_kept_rows_are_the_transformed_full_vocabulary_logits(self):
        """Regular requests in a beam batch sample from the rows the normalizer
        pass keeps: they must equal the dense LM head after the head's transform,
        across chunk edges, while compact logits and normalizers stay as without
        keeping. A captured buffer keeps only the rows its count names."""
        torch.manual_seed(0)
        hidden_states = torch.randn(4, 5)
        weight = torch.randn(18, 5)
        depths = torch.tensor([0, 1, 0, 1])

        def project(states, selected_weight):
            return torch.matmul(states, selected_weight.T)

        def transform(logits):
            return logits * 0.5

        head = BeamTrieOutputHead(_TRIE_CONFIG, vocab_size=16, chunk_size=3)
        plain = head.project_mixed(hidden_states, weight, depths, project, transform)
        kept = head.project_mixed(
            hidden_states,
            weight,
            depths,
            project,
            transform,
            dense_rows=BeamDenseRows(num_rows=3),
        )

        # Chunk by chunk, as the head projects: one GEMM over all 16 columns
        # rounds differently on CPU.
        dense = torch.cat(
            [
                transform(project(hidden_states, weight[start : min(start + 3, 16)]))
                for start in range(0, 16, 3)
            ],
            dim=1,
        )
        self.assertTrue(torch.equal(kept.dense_logits, dense[:3]))
        self.assertTrue(torch.equal(kept.logits, plain.logits))
        self.assertTrue(torch.equal(kept.normalizer, plain.normalizer))

        captured = BeamDenseRows(
            num_rows=3,
            count=torch.tensor([1], dtype=torch.int32),
            logits=torch.full((3, 16), -7.0),
        )
        head.project_mixed(
            hidden_states, weight, depths, project, transform, dense_rows=captured
        )
        self.assertTrue(torch.equal(captured.logits[:1], dense[:1]))
        self.assertTrue(bool((captured.logits[1:] == -7.0).all()))


if __name__ == "__main__":
    unittest.main()
