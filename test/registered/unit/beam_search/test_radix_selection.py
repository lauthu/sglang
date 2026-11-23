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
"""The coordinator's selection with SGLANG_ENABLE_BEAM_TRIE_RADIX_SELECTION against its torch path."""

import unittest
from types import SimpleNamespace
from unittest import mock

import torch

from sglang.srt.beam_search.batch_tail import BeamTail, BeamTailEntry
from sglang.srt.beam_search.beam_group import BeamGroup
from sglang.srt.beam_search.coordinator import BeamCoordinator
from sglang.srt.beam_search.joint_select import FinalSelect
from sglang.srt.beam_search.logits_capture import capture_pre_sample_logits
from sglang.srt.beam_search.radix_selection import RadixBeamSelector
from sglang.srt.beam_search.trie_config import TrieOutputHeadConfig
from sglang.srt.beam_search.trie_constraint import BeamTrieConstraint, _SparseLevel2
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.model_executor.forward_mode import ForwardMode
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-small")

TOKEN_START = 1000
CODEBOOK = 64
VOCAB = TOKEN_START + 3 * CODEBOOK + 7
PROMPT_LEN = 5
POOL_ROWS = 512
# One request per width in one batch: 100 beams take the multi-program kernels at 64 codes,
# the others one program; 100 also exceeds the codebook, so its first step pads.
MIXED_WIDTHS = (8, 100, 3, 64)
STOP_TOKEN = 7  # outside the SID codebooks, as admission requires


class _Req(SimpleNamespace):
    """A Req stand-in that BeamGroup can hold as it holds a real Req."""


class _ReqToTokenPool:
    def __init__(self, num_groups):
        self.device = torch.device("cuda")
        self.req_to_token = torch.arange(
            POOL_ROWS * 16, dtype=torch.int32, device="cuda"
        ).view(POOL_ROWS, 16)
        self._next_row = num_groups

    def take_rows(self, n):
        rows = list(range(self._next_row, self._next_row + n))
        self._next_row += n
        return rows

    def available_size(self):
        return POOL_ROWS - self._next_row


class _FutureMap:
    def __init__(self):
        self.tokens = torch.zeros(POOL_ROWS, dtype=torch.int64, device="cuda")

    def stash(self, rows, payload):
        self.tokens[rows] = payload.bonus_tokens


def _trie(sparse_level2=False):
    """A random trie in which every prefix keeps a child, level 2 stored dense or sparse."""
    gen = torch.Generator().manual_seed(0)
    root = torch.rand(CODEBOOK, generator=gen) < 0.7
    level1 = torch.rand((CODEBOOK, CODEBOOK), generator=gen) < 0.5
    level2 = torch.rand((CODEBOOK * CODEBOOK, CODEBOOK), generator=gen) < 0.5
    root[0] = level1[:, 0] = level2[:, 0] = True
    level2_sparse = None
    if sparse_level2:
        counts = level2.sum(dim=1)
        level2_sparse = _SparseLevel2(
            keys=torch.arange(CODEBOOK * CODEBOOK, dtype=torch.int16, device="cuda"),
            offsets=torch.cat([counts.new_zeros(1), counts.cumsum(0)]).to(
                device="cuda", dtype=torch.int32
            ),
            values=level2.nonzero()[:, 1].to(device="cuda", dtype=torch.int16),
            max_children=int(counts.max()),
        )
    return BeamTrieConstraint(
        config=TrieOutputHeadConfig(
            tensor_path="",
            token_start=TOKEN_START,
            codebook_size=CODEBOOK,
            num_codebooks=3,
        ),
        root_mask=root.cuda(),
        level1_masks=level1.cuda(),
        level2_masks=None if sparse_level2 else level2.cuda(),
        level2_sparse=level2_sparse,
        level_cardinalities=(int(root.sum()), int(level1.sum()), int(level2.sum())),
    )


def _setup(trie, widths, *, radix, stop_tokens=()):
    pool = _ReqToTokenPool(len(widths))
    with mock.patch(
        "sglang.srt.beam_search.coordinator.get_beam_trie_config", return_value=None
    ):
        coordinator = BeamCoordinator(
            model_config=None,
            spec_algorithm=None,
            dllm_enabled=False,
            max_req_len=0,
            req_to_token_pool=pool,
            token_to_kv_pool_allocator=None,
            tree_cache=None,
            future_map=_FutureMap(),
            trie_constraint=trie,
        )
    if radix:
        coordinator._radix_selector = RadixBeamSelector(trie)
    reqs = []
    for row, width in enumerate(widths):
        group = BeamGroup(
            beam_width=width,
            max_new_tokens=3,
            stop_token_ids=stop_tokens,
            device="cuda",
            device_history=True,
        )
        group.trie_prefix_codes = torch.empty((1, 0), dtype=torch.int64, device="cuda")
        group.prompt_len = PROMPT_LEN
        req = _Req(
            beam_group=group,
            kv=ReqKvInfo(
                req_pool_idx=row,
                kv_committed_len=PROMPT_LEN,
                kv_allocated_len=PROMPT_LEN,
            ),
            output_ids=[],
        )
        group.leader = req
        reqs.append(req)
    return coordinator, reqs


def _forward(mode, rows, width, seed):
    """Logits of one forward: full-vocabulary rows for a prefill, compact rows with their
    normalizer for a decode. Code logits are distinct per row, so no path meets a tie."""
    gen = torch.Generator(device="cuda").manual_seed(seed)
    logits = torch.randn((rows, width), generator=gen, device="cuda")
    codes = (
        slice(0, CODEBOOK)
        if width == CODEBOOK
        else slice(TOKEN_START, TOKEN_START + CODEBOOK)
    )
    order = torch.rand((rows, CODEBOOK), generator=gen, device="cuda").argsort(dim=1)
    logits[:, codes] = torch.linspace(-3, 3, CODEBOOK, device="cuda")[order]
    output = LogitsProcessorOutput(next_token_logits=logits)
    if width == CODEBOOK:
        output.beam_normalizer = torch.randn(
            (rows, 2), generator=gen, device="cuda"
        ).abs()
    forward_batch = SimpleNamespace(
        forward_mode=mode, positions=torch.zeros(rows, device="cuda")
    )
    return output, forward_batch


def _run_search(coordinator, reqs):
    """A whole three-token search for every request, all of them in the same batches."""
    num = len(reqs)
    with mock.patch(
        "sglang.srt.beam_search.coordinator.kv_storage_of",
        return_value=coordinator.req_to_token_pool,
    ):
        batch = SimpleNamespace(
            beam_tail=None, forward_mode=ForwardMode.EXTEND, reqs=reqs
        )
        output, forward_batch = _forward(ForwardMode.EXTEND, num, VOCAB, 0)
        capture_pre_sample_logits(batch, forward_batch, output)
        for pos, req in enumerate(reqs):
            coordinator.select_leader_prefill(req, pos, output, tick=0)

        for tick in (1, 2):
            entries, start = [], 0
            for i, req in enumerate(reqs):
                req.kv.kv_committed_len = PROMPT_LEN + tick
                members = req.beam_group.beam_width - 1
                entries.append(BeamTailEntry(req.beam_group, i, start, start + members))
                start += members
            batch = SimpleNamespace(
                beam_tail=BeamTail(num_base_rows=num, entries=entries),
                forward_mode=ForwardMode.DECODE,
                reqs=reqs,
                forward_iter=tick,
            )
            output, forward_batch = _forward(
                ForwardMode.DECODE, num + start, CODEBOOK, tick
            )
            capture_pre_sample_logits(batch, forward_batch, output)
            coordinator.select_and_relay_decode(batch, output)


def _staged(coordinator, reqs):
    """Everything the launch half leaves behind: selections, frontier, KV slots, relays."""
    outputs = {
        "req_to_token": coordinator.req_to_token_pool.req_to_token,
        "relayed": coordinator.future_map.tokens,
    }
    for i, req in enumerate(reqs):
        group = req.beam_group
        outputs[f"group{i} prefix"] = group.trie_prefix_codes
        outputs[f"group{i} frontier"] = group.frontier_cum_logprobs
        outputs[f"group{i} placeholders"] = torch.tensor(len(req.output_ids))
        for step, (tick, sel) in enumerate(group._pending_steps):
            if isinstance(sel, FinalSelect):
                fields = (sel.tokens, sel.parent_idx, sel.cum_logprobs)
            else:
                fields = (
                    sel.next_tokens,
                    sel.parent_idx,
                    sel.new_cum_logprobs,
                    sel.num_survivors,
                    sel.fin_tokens,
                    sel.fin_parent_idx,
                    sel.fin_cum_logprobs,
                    sel.num_finished,
                )
            outputs[f"group{i} step{step} tick"] = torch.tensor(tick)
            for j, value in enumerate(fields):
                outputs[f"group{i} step{step} field{j}"] = value
        for j, orphans in enumerate(group.pending_orphans):
            outputs[f"group{i} orphans{j} tick"] = torch.tensor(orphans.tick)
            # The orphan search reads both mappings as sets of slots.
            outputs[f"group{i} orphans{j} old"] = (
                orphans.old_mapping.flatten().sort().values
            )
            outputs[f"group{i} orphans{j} new"] = (
                orphans.new_mapping.flatten().sort().values
            )
    return outputs


class TestRadixSelection(CustomTestCase):
    def _compare(self, trie, widths, *, radix_groups, stop_tokens=()):
        """Runs the search on both paths; radix_groups is how many group selections the
        radix kernels must have taken."""
        results = {}
        for radix in (False, True):
            coordinator, reqs = _setup(
                trie, widths, radix=radix, stop_tokens=stop_tokens
            )
            taken = []
            if radix:
                select = RadixBeamSelector.select

                def counted(selector, step, groups, **kwargs):
                    taken.append(len(groups))
                    return select(selector, step, groups, **kwargs)

                with mock.patch.object(RadixBeamSelector, "select", counted):
                    _run_search(coordinator, reqs)
                self.assertEqual(sum(taken), radix_groups)
            else:
                _run_search(coordinator, reqs)
            torch.cuda.synchronize()
            for req in reqs:
                self.assertEqual(req.beam_group.num_generated, 3)
            results[radix] = _staged(coordinator, reqs)
        self.assertEqual(results[True].keys(), results[False].keys())
        for name, expected in results[False].items():
            value = results[True][name]
            self.assertEqual(value.dtype, expected.dtype, name)
            self.assertTrue(torch.equal(value, expected), name)

    def test_mixed_widths_stage_what_the_torch_path_stages(self):
        """Requests of different beam widths share every batch: their first selection reads
        full-vocabulary prefill rows, their decode selections one launch set per step."""
        self._compare(_trie(), MIXED_WIDTHS, radix_groups=3 * len(MIXED_WIDTHS))

    def test_stop_tokens_outside_the_codebooks_change_nothing(self):
        """A request without ignore_eos carries its EOS ids as stop tokens. They lie outside
        the SID codebooks, so no selected beam can finish on one, and the kernels must cover
        such a group and stage what joint_select's stop bookkeeping stages."""
        self._compare(_trie(), (8, 100), radix_groups=6, stop_tokens=(STOP_TOKEN,))

    def test_sparse_level2_final_step_keeps_the_torch_path(self):
        """A codebook too large for a dense level-2 table stores it sparse; the kernels read
        dense tables, so the final step of each group must fall back and still agree."""
        self._compare(_trie(sparse_level2=True), (8, 100), radix_groups=4)

    def test_launch_half_never_syncs_the_host(self):
        """The CPU queues the selection behind a forward that is still running; a host
        sync on this path would stall every step for the forward's duration."""
        coordinator, reqs = _setup(_trie(), MIXED_WIDTHS, radix=True)
        _run_search(coordinator, reqs)  # compiles the kernels
        torch.cuda.synchronize()
        coordinator, reqs = _setup(_trie(), MIXED_WIDTHS, radix=True)
        torch.cuda.set_sync_debug_mode("error")
        try:
            _run_search(coordinator, reqs)
        finally:
            torch.cuda.set_sync_debug_mode("default")


if __name__ == "__main__":
    unittest.main()
