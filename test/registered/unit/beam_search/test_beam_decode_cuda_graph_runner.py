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
a batch that fits them. Also the stream-K plan that the host adds to a
variable-width batch's layout."""

import collections
import contextlib
import itertools
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import msgspec
import torch

from sglang.kernels.ops.attention.beam_shared_context_variable_width_stream_k import (
    build_stream_k_plan,
    task_units,
)
from sglang.srt.beam_search import (
    beam_decode_cuda_graph_runner,
    shared_context_attention,
    trie_output_head,
)
from sglang.srt.beam_search.batch_tail import BeamTailEntry
from sglang.srt.beam_search.beam_decode_cuda_graph_runner import (
    BeamGraphInputs,
    FixedWidthCapacity,
    FixedWidthSharedContextInput,
    TrieLevelsInput,
    VariableWidthSharedContextInput,
    _create_beam_graph_inputs,
)
from sglang.srt.beam_search.forward_inputs import (
    BeamForwardInputs,
    build_beam_forward_inputs,
)
from sglang.srt.beam_search.shared_context_fixed_width import (
    BeamAttentionBucket,
    BeamAttentionMetadata,
)
from sglang.srt.beam_search.shared_context_variable_width import (
    BEAMS_PER_TILE,
    StreamKGrid,
    VariableWidthBeamAttentionMetadata,
    build_variable_width_metadata,
)
from sglang.srt.beam_search.trie_config import TrieOutputHeadConfig
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.kernels.beam_shared_context_variable_width import (
    ALWAYS_SPLIT,
    NEVER_SPLIT,
    stream_k_split_cost,
)
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

    def test_graphs_need_a_token_budget(self):
        """The graph is captured for a fixed number of decode steps. Without a trie
        nothing bounds the steps, so a server that sets no budget captures no
        graph. A step attends the tokens generated before it: one fewer than the
        budget."""
        runner = SimpleNamespace(
            lora_manager=None,
            device="cpu",
            model_config=SimpleNamespace(context_len=16),
        )
        cases = (
            # (width, max new tokens, trie codebooks) -> max_decode_len, or None for no graph
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
def _shared_context_server(*, decode_graphs=True, fixed_width=None, kernel="default"):
    """The settings of a server with shared-context attention on and supported.
    With the beam decode graphs on and no fixed width, its graphs are the
    variable-width ones."""
    with contextlib.ExitStack() as stack:
        overrides = (
            (envs.SGLANG_ENABLE_BEAM_SHARED_CONTEXT_ATTENTION, True),
            (envs.SGLANG_ENABLE_BEAM_DECODE_GRAPH, decode_graphs),
            (envs.SGLANG_BEAM_DECODE_GRAPH_FIXED_WIDTH, fixed_width),
            (envs.SGLANG_BEAM_SHARED_CONTEXT_KERNEL, kernel),
        )
        for setting, value in overrides:
            stack.enter_context(setting.override(value))
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


# Entries of _decode_batch's `widths` that are not a live beam group.
REGULAR_REQUEST = 1
FINISHED_GROUP = 0


def _decode_batch(widths, prompt_len=7, decode_len=2):
    """A decode batch of one request per entry of `widths`: a beam width, or
    REGULAR_REQUEST (a request with its own KV), or FINISHED_GROUP (the leader of
    a beam group that has no member rows left).

    Rows are laid out as the scheduler does: one row per request, in order, then
    each live beam group's member rows.
    """
    reqs, entries, leader_lens, member_lens = [], [], [], []
    for row, width in enumerate(widths):
        if width == REGULAR_REQUEST:
            reqs.append(SimpleNamespace(beam_group=None))
            leader_lens.append(40 + row)
            continue
        group = SimpleNamespace(prompt_len=prompt_len, num_generated=decode_len - 1)
        reqs.append(SimpleNamespace(beam_group=group))
        leader_lens.append(prompt_len + decode_len)
        if width != FINISHED_GROUP:
            num_members = width - 1
            start = len(member_lens)
            entries.append(
                BeamTailEntry(
                    group=group, leader_idx=row, start=start, end=start + num_members
                )
            )
            member_lens.extend([prompt_len + decode_len] * num_members)
    seq_lens = torch.tensor(leader_lens + member_lens, dtype=torch.int64)
    return SimpleNamespace(
        reqs=reqs,
        seq_lens=seq_lens,
        seq_lens_cpu=seq_lens,
        beam_tail=SimpleNamespace(num_base_rows=len(reqs), entries=entries),
        forward_mode=SimpleNamespace(is_decode=lambda: True),
        device=torch.device("cpu"),
    )


def _variable_width_graph_inputs(max_rows, *, trie_head=True):
    return BeamGraphInputs(
        trie_levels_input=(
            TrieLevelsInput(max_rows=max_rows, device="cpu") if trie_head else None
        ),
        shared_context_input=VariableWidthSharedContextInput(
            max_rows=max_rows,
            device="cpu",
            req_to_token=torch.zeros(4, 8, dtype=torch.int32),
            stream_k_grid=None,
        ),
    )


def _read_layout(layout):
    """What a layout buffer holds: (real rows, tiles). The format the kernels
    read is a header of [tiles, real rows], then per tile (leader row, first
    member row, members, prompt length, first beam)."""
    header_len, tile_fields = 2, 5
    num_tiles, num_real_rows = layout[:header_len].tolist()
    tiles = layout[header_len : header_len + tile_fields * num_tiles]
    return num_real_rows, tiles.view(num_tiles, tile_fields).tolist()


def _rows_in_tiles(layout):
    """The row of every beam of every tile of a layout, tile by tile."""
    _, tiles = _read_layout(layout)
    rows = []
    for leader_row, first_member_row, num_members, _, first_beam in tiles:
        beam_width = num_members + 1
        for beam in range(first_beam, min(first_beam + BEAMS_PER_TILE, beam_width)):
            rows.append(leader_row if beam == 0 else first_member_row + beam - 1)
    return rows


class TestVariableWidthBeamGraphInputs(CustomTestCase):
    def test_settings_decide_what_the_graphs_hold(self):
        """With a fixed width set, graphs hold the fixed-width shared-context
        input, which limits them to whole groups of that width. Without one they
        hold the variable-width input and are captured at every size. A LoRA server
        gets no beam graphs: the beam label takes LoRA's slot of the graph key."""
        # (fixed width, compact head, LoRA) -> the shared-context input's class,
        # or None for a server without beam graphs.
        cases = {
            (2, True, False): FixedWidthSharedContextInput,
            (None, True, False): VariableWidthSharedContextInput,
            (None, False, False): VariableWidthSharedContextInput,
            (None, True, True): None,
        }
        for (fixed_width, compact_head, lora), input_class in cases.items():
            with self.subTest(
                fixed_width=fixed_width, compact_head=compact_head, lora=lora
            ):
                with (
                    _shared_context_server(fixed_width=fixed_width),
                    patch.object(
                        beam_decode_cuda_graph_runner,
                        "get_beam_trie_config",
                        return_value=_TRIE_CONFIG,
                    ),
                ):
                    inputs = _create_beam_graph_inputs(
                        _model_runner(compact_head=compact_head, lora=lora)
                    )
                if input_class is None:
                    self.assertIsNone(inputs)
                    continue
                self.assertIsInstance(inputs.shared_context_input, input_class)
                self.assertEqual(inputs.trie_levels_input is not None, compact_head)
                # 3 rows are not whole groups of 2 beams.
                self.assertEqual(inputs.can_capture(3), fixed_width is None)

    def test_only_graphs_without_a_fixed_width_run_the_variable_width_layout(self):
        """A server keeps the per-width buckets, and with them its kernel, unless
        its beam decode graphs take any width: with the graphs off, or with a
        fixed width, a batch must not get the variable-width layout."""
        batch, runner = _decode_batch((4, 4)), _model_runner()
        width_buckets = object()
        # Server settings -> whether a batch gets the variable-width layout. The
        # per-width buckets have a split-KV kernel.
        cases = (
            ({"decode_graphs": False}, False),
            ({"decode_graphs": False, "kernel": "split_kv"}, False),
            ({"fixed_width": 4}, False),
            ({"fixed_width": 4, "kernel": "split_kv"}, False),
            ({}, True),
        )
        for settings, expects_variable_width in cases:
            with (
                self.subTest(settings=settings),
                _shared_context_server(**settings),
                patch.object(
                    shared_context_attention,
                    "build_fixed_width_metadata",
                    return_value=width_buckets,
                ),
            ):
                metadata = shared_context_attention.build_metadata(batch, runner)
                self.assertEqual(
                    isinstance(metadata, VariableWidthBeamAttentionMetadata),
                    expects_variable_width,
                )

    def test_any_widths_load_into_a_larger_graph(self):
        """Variable-width graphs take a batch of any widths and of fewer rows than
        they hold. Loaded over a larger earlier batch's layout, the batch's tiles
        must cover each of its rows exactly once."""
        with _shared_context_server():
            inputs = _variable_width_graph_inputs(max_rows=600)
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
            num_real_rows, tiles = _read_layout(layout)
            leader_rows = {leader_row for leader_row, _, _, _, _ in tiles}
            self.assertEqual((num_real_rows, len(leader_rows)), (431, 3))
            self.assertEqual(sorted(_rows_in_tiles(layout)), list(range(431)))

    def test_dense_head_batch_replays_without_trie_levels(self):
        """With the compact trie head off, a beam batch carries its variable-width
        layout and no trie levels; a graph captured without levels replays it."""
        with _shared_context_server():
            inputs = _variable_width_graph_inputs(max_rows=600, trie_head=False)
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

    def test_variable_width_graph_takes_no_ordinary_attention_batch(self):
        """A batch that the layout builder turned away (here: a finished group's
        leader is still in it) carries no layout; replaying it would attend the
        previous batch's groups. A batch with a regular request carries no beam
        inputs at all, so it decodes on the ordinary path."""
        with _shared_context_server():
            inputs = _variable_width_graph_inputs(max_rows=600)
            inputs.prepare_capture_inputs(512)
            beam = build_beam_forward_inputs(
                _decode_batch((129, FINISHED_GROUP, 300)), _model_runner()
            )
            self.assertIsNotNone(beam.trie_levels)
            self.assertIsNone(beam.attention_metadata)
            self.assertFalse(inputs.can_replay(beam, graph_size=512))
            with_regular_request = _decode_batch((129, REGULAR_REQUEST, 300))
            self.assertIsNone(
                build_beam_forward_inputs(with_regular_request, _model_runner())
            )

    def test_variable_width_graphs_refuse_a_kernel_they_lack(self):
        """The variable-width layout has no split-KV kernel, and selecting it must
        not silently run another one: the server fails when it creates its beam
        graphs, and a batch fails on a server that captures none."""
        runner = _model_runner()
        with _shared_context_server(kernel="split_kv"):
            with self.assertRaisesRegex(ValueError, "'split_kv', not one of"):
                _create_beam_graph_inputs(runner)
            with self.assertRaisesRegex(ValueError, "'split_kv', not one of"):
                shared_context_attention.build_metadata(_decode_batch((4, 4)), runner)

    def test_layout_buffer_holds_the_most_tiles_of_the_largest_graph(self):
        """A graph's rows hold the most tiles when every group has two beams. The
        layout of that batch, with a stream-K plan after its tiles, fills the
        buffer that the graphs captured."""
        max_rows = 64
        grid = StreamKGrid(q_heads=1, num_ctas=7)
        req_to_token = torch.zeros(4, 8, dtype=torch.int32)
        shared_context_input = VariableWidthSharedContextInput(
            max_rows=max_rows,
            device="cpu",
            req_to_token=req_to_token,
            stream_k_grid=grid,
        )
        graph_layout = shared_context_input.prepare_capture_inputs(max_rows).layout
        metadata = build_variable_width_metadata(
            batch=_decode_batch((2,) * (max_rows // 2)),
            req_to_token=req_to_token,
            stream_k_grid=grid,
        )
        self.assertEqual(len(metadata.host_layout), len(graph_layout))
        shared_context_input.load_replay_inputs(metadata)
        self.assertEqual(graph_layout.tolist(), metadata.host_layout.tolist())


# Tokens of a prompt block in the stream-K kernel.
PROMPT_BLOCK = 64


class TestStreamKPlan(CustomTestCase):
    def test_plan_attends_every_work_unit_once_and_lists_every_split_task(self):
        """What the stream-K kernels rely on, for any batch, grid and choice of
        splitting: the segments of a task cover its work units once and in order;
        a task in one piece is written directly; a split task is listed once, by
        the CTA with its first segment, up to the CTA with its last, and its later
        segments are where the merge kernel reads them."""
        # (prompt blocks, decode steps) of each tile's tasks.
        batches = {
            "one request": [(32, 2)] * 2,
            "eight requests": [(32, 2)] * 16,
            "uneven tiles": [(1, 1), (11, 3), (1, 5), (32, 1), (8, 2), (3, 1)],
            "one tiny tile": [(1, 1)],
        }
        # None: the plan chooses between whole tasks and equal runs, as in serving.
        for (name, tiles), q_heads, num_ctas, split_cost_units in itertools.product(
            batches.items(),
            (1, 16),
            (1, 3, 7, 64, 188),
            (None, NEVER_SPLIT, ALWAYS_SPLIT),
        ):
            with contextlib.ExitStack() as stack:
                stack.enter_context(
                    self.subTest(
                        batch=name,
                        q_heads=q_heads,
                        num_ctas=num_ctas,
                        split=split_cost_units,
                    )
                )
                if split_cost_units is not None:
                    stack.enter_context(stream_k_split_cost(split_cost_units))
                split_tasks = self._check_plan(
                    tiles, q_heads=q_heads, num_ctas=num_ctas
                )
                if split_cost_units == NEVER_SPLIT:
                    self.assertEqual(split_tasks, 0)
                if split_cost_units == ALWAYS_SPLIT and num_ctas == 188:
                    self.assertGreater(split_tasks, 0)

    def _check_plan(self, tiles, *, q_heads, num_ctas):
        """Check one plan as the kernels read it; returns its number of split tasks."""
        work_units = [blocks + steps for blocks, steps in tiles for _ in range(q_heads)]
        plan = build_stream_k_plan(
            task_units_per_tile=[
                task_units(prompt_len=PROMPT_BLOCK * blocks, decode_len=steps)
                for blocks, steps in tiles
            ],
            q_heads=q_heads,
            num_ctas=num_ctas,
        ).tolist()
        self.assertEqual(len(plan), num_ctas)

        # Per task, its segments in CTA order: (cta, work start, work end, partial, slot).
        segments = collections.defaultdict(list)
        # Per listed task: (listing cta, last cta, first slot).
        listed = {}
        for cta, row in enumerate(plan):
            first_task, end_task, first_start, last_end, first_partial, last_partial = (
                row[:6]
            )
            for task in range(first_task, end_task):
                is_first, is_last = task == first_task, task == end_task - 1
                segments[task].append(
                    (
                        cta,
                        first_start if is_first else 0,
                        last_end if is_last else work_units[task],
                        bool(
                            (is_first and first_partial) or (is_last and last_partial)
                        ),
                        2 * cta + (0 if is_first else 1),
                    )
                )
            tile_plus_1, q_head, last_cta, first_slot = row[6:]
            if tile_plus_1:
                task = (tile_plus_1 - 1) * q_heads + q_head
                self.assertNotIn(task, listed)
                listed[task] = (cta, last_cta, first_slot)

        for task, num_units in enumerate(work_units):
            parts = segments[task]
            starts = [start for _, start, _, _, _ in parts]
            ends = [end for _, _, end, _, _ in parts]
            self.assertEqual(starts, [0, *ends[:-1]])
            self.assertEqual(ends[-1], num_units)
            self.assertTrue(all(start < end for start, end in zip(starts, ends)))
            if len(parts) == 1:
                self.assertFalse(parts[0][3])
                self.assertNotIn(task, listed)
                continue
            self.assertTrue(all(partial for _, _, _, partial, _ in parts))
            segment_ctas = [cta for cta, _, _, _, _ in parts]
            self.assertEqual(
                listed[task], (segment_ctas[0], segment_ctas[-1], parts[0][4])
            )
            # The merge kernel reads slot 2 * c of every later CTA c up to the
            # last; one without a segment has no task and marked that slot empty.
            self.assertEqual(
                [slot for _, _, _, _, slot in parts[1:]],
                [2 * cta for cta in segment_ctas[1:]],
            )
            for cta in range(segment_ctas[0] + 1, segment_ctas[-1]):
                if cta not in segment_ctas:
                    self.assertEqual(plan[cta][0], plan[cta][1])
        return len(listed)


if __name__ == "__main__":
    unittest.main()
