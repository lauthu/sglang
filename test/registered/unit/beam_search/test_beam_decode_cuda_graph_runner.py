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
    shared_context_attention,
    trie_output_head,
)
from sglang.srt.beam_search.batch_tail import BeamTailEntry
from sglang.srt.beam_search.beam_decode_cuda_graph_runner import (
    BeamGraphInputs,
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
from sglang.srt.beam_search.shared_context_attention import (
    BeamAttentionBucket,
    BeamAttentionMetadata,
)
from sglang.srt.beam_search.shared_context_width_free import WIDTH_FREE_BLOCK_M
from sglang.srt.beam_search.trie_config import TrieOutputHeadConfig
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
        model_config=SimpleNamespace(context_len=8),
        req_to_token_pool=SimpleNamespace(
            req_to_token=torch.zeros(4, 8, dtype=torch.int32)
        ),
        device=torch.device("cpu"),
        max_decode_logits_rows=lambda: 600,
    )


@contextlib.contextmanager
def _beam_decode_graphs(*, fixed_width=None, **env):
    """A server with the beam decode graphs on, and shared-context attention on
    and supported. Without a fixed width its graphs are the width-free ones."""
    with contextlib.ExitStack() as stack:
        overrides = {
            "SGLANG_ENABLE_BEAM_SHARED_CONTEXT_ATTENTION": True,
            "SGLANG_ENABLE_BEAM_DECODE_GRAPH": True,
            "SGLANG_BEAM_DECODE_GRAPH_FIXED_WIDTH": fixed_width,
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


def _width_free_inputs(max_rows, *, trie_head=True):
    return BeamGraphInputs(
        trie_levels_input=(
            TrieLevelsInput(max_rows=max_rows, device="cpu") if trie_head else None
        ),
        shared_context_input=WidthFreeSharedContextInput(
            max_rows=max_rows,
            device="cpu",
            req_to_token=torch.zeros(4, 8, dtype=torch.int32),
            stream_k=False,
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
        for beam in range(
            first_beam, min(first_beam + WIDTH_FREE_BLOCK_M, members + 1)
        ):
            rows.append(leader if beam == 0 else first_member + beam - 1)
    return rows


class TestWidthFreeBeamGraphInputs(CustomTestCase):
    def test_settings_decide_what_the_graphs_hold(self):
        """Graphs hold the fixed-width shared-context input, which limits them to
        whole groups of that width, only when a width is set; otherwise they hold
        the width-free one and take every size. LoRA, whose graph key slot the beam
        label takes, leaves a shared-context server without beam graphs."""
        # (fixed width, compact head, LoRA): shared-context input, None for no graphs
        cases = {
            (2, True, False): FixedWidthSharedContextInput,
            (None, True, False): WidthFreeSharedContextInput,
            (None, False, False): WidthFreeSharedContextInput,
            (None, True, True): None,
        }
        for (fixed_width, compact_head, lora), shared_context in cases.items():
            with self.subTest(
                fixed_width=fixed_width, compact_head=compact_head, lora=lora
            ):
                with (
                    _beam_decode_graphs(fixed_width=fixed_width),
                    patch.object(
                        beam_decode_cuda_graph_runner,
                        "get_beam_trie_config",
                        return_value=_TRIE_CONFIG,
                    ),
                ):
                    inputs = _create_beam_graph_inputs(
                        _model_runner(compact_head=compact_head, lora=lora)
                    )
                if shared_context is None:
                    self.assertIsNone(inputs)
                    continue
                self.assertIs(type(inputs.shared_context_input), shared_context)
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
        batch's groups. A batch with a regular request carries no beam inputs at
        all, so it decodes on the ordinary path."""
        with _beam_decode_graphs():
            inputs = _width_free_inputs(max_rows=600)
            inputs.prepare_capture_inputs(512)
            beam = build_beam_forward_inputs(
                _decode_batch((129, 0, 300)), _model_runner()
            )
            self.assertIsNotNone(beam.trie_levels)
            self.assertIsNone(beam.attention_metadata)
            self.assertFalse(inputs.can_replay(beam, graph_size=512))
            self.assertIsNone(
                build_beam_forward_inputs(_decode_batch((129, 1, 300)), _model_runner())
            )

    def test_startup_rejects_settings_that_cannot_run_together(self):
        """The stream-K kernel exists only on the width-free graphs, and a kernel
        name they do not have must not silently run another kernel."""
        check = decode_graph_settings.check_settings
        with _beam_decode_graphs(SGLANG_BEAM_SHARED_CONTEXT_KERNEL="stream_k"):
            check()
        for without_width_free_graphs in (
            {"fixed_width": 2},
            {"SGLANG_ENABLE_BEAM_DECODE_GRAPH": False},
        ):
            with _beam_decode_graphs(
                SGLANG_BEAM_SHARED_CONTEXT_KERNEL="stream_k",
                **without_width_free_graphs,
            ):
                with self.assertRaisesRegex(ValueError, "FIXED_WIDTH unset"):
                    check()
        with _beam_decode_graphs(SGLANG_BEAM_SHARED_CONTEXT_KERNEL="split_kv"):
            with self.assertRaisesRegex(ValueError, "not one of"):
                check()


if __name__ == "__main__":
    unittest.main()
