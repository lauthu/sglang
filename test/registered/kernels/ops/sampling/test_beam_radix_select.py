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

"""Radix-select trie beam selection against a deterministic reference and the torch path."""

import itertools
import unittest

import torch

from sglang.kernels.ops.sampling.beam_radix_select import radix_select_trie_beams
from sglang.srt.beam_search.fork import remap_kv_mapping
from sglang.srt.beam_search.joint_select import joint_select, select_final_topk
from sglang.srt.beam_search.trie_config import TrieOutputHeadConfig
from sglang.srt.beam_search.trie_constraint import BeamTrieConstraint
from sglang.test.ci.ci_register import register_cuda_ci
from sglang.test.test_utils import CustomTestCase

register_cuda_ci(est_time=90, stage="base-b-kernel-unit", runner_config="1-gpu-small")

TOKEN_START = 1000
KV_ARGS = ("req_to_token", "kv_rows", "kv_starts", "kv_len")


def make_case(
    *,
    widths,
    size,
    depth,
    final,
    seed,
    density=0.5,
    logits_dtype=torch.bfloat16,
    dead_rows=0,
    quantize=None,
    kv_len=0,
    row_width=None,
    device="cuda",
):
    """Random inputs in the coordinator's layout: leaders and members in separate tensors,
    one group per entry of `widths`. row_width > size makes the logits rows wider than the
    codebook, as dense first-step logits are."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    groups = len(widths)
    rows = [1 if depth == 0 else width for width in widths]
    total = sum(rows)
    row_width = row_width or size
    leader_logits = torch.randn(groups + 3, row_width, generator=gen)
    tail_logits = torch.randn(total - groups + 5, row_width, generator=gen)
    if quantize == 0:
        leader_logits, tail_logits = (
            torch.zeros_like(leader_logits),
            torch.zeros_like(tail_logits),
        )
    elif quantize is not None:
        leader_logits = (leader_logits * quantize).round() / quantize
        tail_logits = (tail_logits * quantize).round() / quantize
    leader_rows = torch.randperm(groups + 3, generator=gen)[:groups].tolist()
    # Tail blocks in a shuffled group order, after two unused rows.
    tail_starts, start = [0] * groups, 2
    for g in torch.randperm(groups, generator=gen).tolist():
        tail_starts[g] = start
        start += rows[g] - 1

    def normalizer(logits):
        return torch.stack(
            (
                logits.max(dim=1).values + 0.5,
                torch.rand(logits.shape[0], generator=gen) * 3 + 1,
            ),
            1,
        )

    leader_norm, tail_norm = normalizer(leader_logits), normalizer(tail_logits)
    cum = torch.cat([-(torch.rand(r, generator=gen) * 8).sort().values for r in rows])
    if depth == 0:
        cum.zero_()
    if dead_rows:
        end = 0
        for r in rows:
            end += r
            cum[max(end - dead_rows, end - r + 1) : end] = float("-inf")
    prefix = torch.randint(0, size, (total, depth), generator=gen)
    masks = torch.rand(size**depth, size, generator=gen) < density
    case = dict(
        leader_logits=leader_logits.to(logits_dtype)[:, :size],
        leader_normalizer=leader_norm,
        leader_rows=leader_rows,
        tail_logits=tail_logits.to(logits_dtype)[:, :size] if depth else None,
        tail_normalizer=tail_norm if depth else None,
        tail_starts=tail_starts if depth else None,
        beam_widths=list(widths),
        cum_logprobs=cum,
        prefix_codes=prefix,
        child_masks=masks,
        codebook_size=size,
        token_start=TOKEN_START + depth * size,
        final=final,
    )
    if kv_len:
        pool_rows = total + 7
        case.update(
            req_to_token=torch.randint(
                0, 1 << 20, (pool_rows, 64), generator=gen, dtype=torch.int32
            ),
            kv_rows=torch.randperm(pool_rows, generator=gen)[:total],
            kv_starts=torch.randint(0, 64 - kv_len, (groups,), generator=gen).tolist(),
            kv_len=kv_len,
        )
    return {
        k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in case.items()
    }


def _group_inputs(case, g):
    """Group g's frontier: logits, normalizers, cum, prefix and the row range it owns."""
    depth = case["prefix_codes"].shape[1]
    rows = 1 if depth == 0 else case["beam_widths"][g]
    begin = sum(1 if depth == 0 else w for w in case["beam_widths"][:g])
    logits = [case["leader_logits"][case["leader_rows"][g]][None]]
    norms = [case["leader_normalizer"][case["leader_rows"][g]][None]]
    if rows > 1:
        start = case["tail_starts"][g]
        logits.append(case["tail_logits"][start : start + rows - 1])
        norms.append(case["tail_normalizer"][start : start + rows - 1])
    span = slice(begin, begin + rows)
    return logits, norms, case["cum_logprobs"][span], case["prefix_codes"][span], span


def reference_select(case):
    """Exact selection: ties go to the lower flat index row * codebook_size + code."""
    size, final = case["codebook_size"], case["final"]
    depth = case["prefix_codes"].shape[1]
    table = case["req_to_token"].clone() if case.get("kv_len") else None
    out = dict(
        tokens=[], parent_idx=[], cum_logprobs=[], prefix_codes=[], old_kv=[], new_kv=[]
    )
    for g, width in enumerate(case["beam_widths"]):
        logits, norms, cum, prefix, span = _group_inputs(case, g)
        x, norm = torch.cat(logits).float(), torch.cat(norms).float()
        rows = x.shape[0]
        flat = torch.zeros(rows, dtype=torch.int64, device=cum.device)
        for i in range(depth):
            flat = flat * size + prefix[:, i]
        valid = case["child_masks"][flat]
        score = cum[:, None] + ((x - norm[:, :1]) - norm[:, 1:])
        num_valid = valid.sum(-1, keepdim=True)
        dead = (~valid).int()
        competes = (dead != 0) & (dead.cumsum(-1) - dead < min(width, size) - num_valid)
        fallback = torch.where(num_valid[:, 0] > 0, valid.int().argmax(-1), 0)
        s = torch.where(valid, score, float("-inf"))
        bits = s.view(torch.int32).long() & 0xFFFFFFFF
        key = torch.where(bits >> 31 != 0, ~bits & 0xFFFFFFFF, bits | 0x80000000)
        high = key ^ 0x80000000
        high = high - ((high >> 31) << 32)
        idx = torch.arange(rows * size, device=cum.device)
        sk = (high.view(-1) << 32) | (0xFFFFFFFF - idx)
        sk = torch.where((valid | competes).view(-1), sk, torch.iinfo(torch.int64).min)
        target = min(width, rows * size)
        top = sk.topk(target).values
        f = 0xFFFFFFFF - (top & 0xFFFFFFFF)
        r, c = f // size, f % size
        code = torch.where(valid[r, c], c, fallback[r])
        tokens, parents, scores = code + case["token_start"], r, s[r, c]
        if target < width:
            pad = width - target
            tokens = torch.cat([tokens, tokens[:1].expand(pad)])
            parents = torch.cat([parents, torch.zeros_like(parents[:1]).expand(pad)])
            scores = torch.cat(
                [scores, torch.full_like(scores[:1], float("-inf")).expand(pad)]
            )
            code = torch.cat([code, code[:1].expand(pad)])
        out["tokens"].append(tokens)
        out["parent_idx"].append(parents)
        out["cum_logprobs"].append(scores)
        if not final:
            out["prefix_codes"].append(
                torch.cat([prefix[parents], code[:, None]], dim=1)
            )
        if table is not None and not final:
            start = case["kv_starts"][g]
            old, new = remap_kv_mapping(
                table,
                rows=case["kv_rows"][span],
                parent_idx=parents,
                prefix_len=start,
                seq_len=start + case["kv_len"],
            )
            out["old_kv"].append(old)
            out["new_kv"].append(new)
    result = {k: torch.cat(v) if v else None for k, v in out.items()}
    result["req_to_token"] = table
    return result


def torch_path_select(case):
    """Upstream's per-group torch path (trie top-k, joint select, prefix advance)."""
    size, depth = case["codebook_size"], case["prefix_codes"].shape[1]
    masks = case["child_masks"]
    constraint = BeamTrieConstraint(
        config=TrieOutputHeadConfig(
            tensor_path="", token_start=TOKEN_START, codebook_size=size, num_codebooks=3
        ),
        root_mask=masks[0]
        if depth == 0
        else torch.ones(size, dtype=torch.bool, device=masks.device),
        level1_masks=masks if depth == 1 else None,
        level2_masks=masks if depth == 2 else None,
        level2_sparse=None,
        level_cardinalities=(size, size * size, size**3),
    )
    results = []
    for g, width in enumerate(case["beam_widths"]):
        pieces, normalizers, cum, prefix, _ = _group_inputs(case, g)
        values, tokens = constraint.topk_logprobs(
            pieces, prefix, depth, width, normalizers
        )
        if case["final"]:
            sel = select_final_topk(cum, values, tokens, width)
            results.append((sel.tokens, sel.parent_idx, sel.cum_logprobs, None))
        else:
            sel = joint_select(
                cum,
                values,
                tokens,
                torch.empty(0, dtype=torch.int64, device=cum.device),
                width,
                num_output_candidates=width,
            )
            nxt = constraint.advance_prefixes(
                prefix, sel.parent_idx[:width], sel.next_tokens[:width], depth
            )
            results.append(
                (
                    sel.next_tokens[:width],
                    sel.parent_idx[:width],
                    sel.new_cum_logprobs[:width],
                    nxt,
                )
            )
    return results


def kernel_select(case):
    args = {k: v for k, v in case.items() if k not in KV_ARGS}
    table = None
    if case.get("kv_len"):
        table = case["req_to_token"].clone()
        args.update(
            req_to_token=table,
            kv_rows=case["kv_rows"],
            kv_starts=case["kv_starts"],
            kv_len=case["kv_len"],
        )
    out = radix_select_trie_beams(**args)
    return out, table


class TestBeamRadixSelect(CustomTestCase):
    def assert_same(self, out, table, ref, msg):
        self.assertTrue(torch.equal(out.tokens, ref["tokens"]), f"tokens {msg}")
        self.assertTrue(
            torch.equal(out.parent_idx, ref["parent_idx"]), f"parents {msg}"
        )
        self.assertTrue(
            torch.equal(
                out.cum_logprobs.view(torch.int32),
                ref["cum_logprobs"].view(torch.int32),
            ),
            f"scores {msg}",
        )
        if ref["prefix_codes"] is None:
            self.assertIsNone(out.prefix_codes, msg)
        else:
            self.assertTrue(
                torch.equal(out.prefix_codes, ref["prefix_codes"]), f"prefix {msg}"
            )
        if ref["old_kv"] is not None:
            self.assertTrue(torch.equal(out.old_kv, ref["old_kv"]), f"old kv {msg}")
            self.assertTrue(torch.equal(out.new_kv, ref["new_kv"]), f"new kv {msg}")
            self.assertTrue(
                torch.equal(table, ref["req_to_token"]), f"req_to_token {msg}"
            )

    def check(self, **kw):
        case = make_case(**kw)
        out, table = kernel_select(case)
        self.assert_same(out, table, reference_select(case), str(kw))

    def test_one_width_per_call(self):
        # The one-program path (<= 4096 candidates per group) and the multi-program path.
        for (width, size), depth, groups in itertools.product(
            [
                (1, 256),
                (3, 64),
                (16, 256),
                (17, 256),
                (64, 256),
                (256, 256),
                (300, 256),
                (1000, 256),
                (2000, 256),
                (128, 1024),
                (40, 16),
            ],
            [1, 2],
            [1, 3],
        ):
            self.check(
                widths=[width] * groups,
                size=size,
                depth=depth,
                final=depth == 2,
                seed=width * 7 + depth + groups,
                kv_len=2 - depth,
            )

    def test_widths_mixed_in_one_call(self):
        # Requests of different beam widths decode in one batch; 16 and 17 beams sit on the
        # two sides of the one-program limit at 256 codes.
        for widths, depth in itertools.product(
            [
                [4, 16, 64, 256],
                [2000, 8, 300, 17],
                [1, 2, 3],
                [16, 17],
                [256, 1000, 256, 32, 2000, 5],
            ],
            [1, 2],
        ):
            self.check(
                widths=widths,
                size=256,
                depth=depth,
                final=depth == 2,
                seed=sum(widths) + depth,
                kv_len=2 - depth,
            )

    def test_first_step(self):
        # One frontier row per group; a width above the codebook size pads like the torch path.
        for widths, size, density in [
            ([16, 256, 2000, 8], 256, 0.5),
            ([256] * 4, 256, 0.9),
            ([300, 64], 256, 0.3),
            ([8, 2000], 1024, 0.01),
        ]:
            self.check(
                widths=widths,
                size=size,
                depth=0,
                final=False,
                seed=sum(widths),
                density=density,
            )

    def test_first_step_reads_dense_rows(self):
        # First-step logits are full-vocabulary rows: the codes are a slice of a wider row.
        self.check(
            widths=[256, 16, 2000],
            size=256,
            depth=0,
            final=False,
            seed=5,
            row_width=1500,
            logits_dtype=torch.float32,
        )

    def test_dead_candidates_and_dead_rows(self):
        # Sparse tries leave fewer valid children than beams; -inf rows score -inf everywhere.
        for widths, depth, dead_rows in [
            ([256, 64], 1, 0),
            ([256, 300], 2, 200),
            ([64, 16], 2, 10),
            ([2000, 256], 1, 1900),
        ]:
            self.check(
                widths=widths,
                size=256,
                depth=depth,
                final=False,
                seed=sum(widths) + dead_rows,
                density=0.004,
                dead_rows=dead_rows,
                kv_len=1,
            )

    def test_ties(self):
        # Coarse logits make exact ties at the threshold; all-equal logits overflow the bucket.
        for widths, quantize, depth in [
            ([256, 64], 2, 1),
            ([2000, 300], 2, 2),
            ([256, 16], 0, 1),
            ([2000, 64], 0, 2),
        ]:
            self.check(
                widths=widths,
                size=256,
                depth=depth,
                final=False,
                seed=sum(widths) + quantize,
                quantize=quantize,
                kv_len=1,
            )

    def test_fp32_logits(self):
        self.check(
            widths=[256, 64],
            size=256,
            depth=2,
            final=False,
            seed=3,
            logits_dtype=torch.float32,
        )

    def test_reference_matches_torch_path(self):
        # Without ties, the torch path's selection is deterministic and must equal the reference.
        for widths, depth in [
            ([16, 256], 1),
            ([256, 300, 8], 1),
            ([256, 64], 2),
            ([64, 256, 2000], 0),
        ]:
            case = make_case(
                widths=widths,
                size=256,
                depth=depth,
                final=depth == 2,
                seed=sum(widths) + depth,
                density=0.9,
                logits_dtype=torch.float32,
            )
            ref = reference_select(case)
            begin = 0
            for width, (tokens, parents, scores, nxt) in zip(
                widths, torch_path_select(case)
            ):
                span = slice(begin, begin + width)
                begin += width
                self.assertTrue(
                    torch.equal(tokens, ref["tokens"][span]), f"tokens {widths} {depth}"
                )
                self.assertTrue(
                    torch.equal(parents, ref["parent_idx"][span]),
                    f"parents {widths} {depth}",
                )
                self.assertTrue(
                    torch.equal(
                        scores.view(torch.int32),
                        ref["cum_logprobs"][span].view(torch.int32),
                    )
                )
                if nxt is not None:
                    self.assertTrue(torch.equal(nxt, ref["prefix_codes"][span]))

    def test_ties_keep_the_torch_paths_scores(self):
        """Logits of a bf16 LM head tie often, and torch.topk leaves the order of ties
        unspecified: the two paths may keep different tied candidates. The selected scores
        are still the same list, and a score that occurs once names the same beam."""
        for widths, depth in [
            ([16, 64, 256, 8], 1),
            ([16, 300], 2),
            ([16, 64, 2000], 0),
        ]:
            case = make_case(
                widths=widths,
                size=256,
                depth=depth,
                final=depth == 2,
                seed=sum(widths) + depth,
                density=0.9,
                quantize=8,
                logits_dtype=torch.float32,
            )
            out, _ = kernel_select(case)
            begin, tied = 0, 0
            for width, (tokens, parents, scores, _) in zip(
                widths, torch_path_select(case)
            ):
                span = slice(begin, begin + width)
                begin += width
                mine = out.cum_logprobs[span]
                self.assertTrue(
                    torch.equal(mine.view(torch.int32), scores.view(torch.int32)),
                    f"{widths} {depth}",
                )
                # Not tied with a neighbor, nor with a candidate just below the cut.
                once = torch.ones(width, dtype=torch.bool, device=scores.device)
                once[1:] &= scores[1:] != scores[:-1]
                once[:-1] &= scores[:-1] != scores[1:]
                once &= scores != scores[-1]
                tied += int((~once).sum())
                self.assertTrue(
                    torch.equal(out.tokens[span][once], tokens[once]),
                    f"tokens {widths} {depth}",
                )
                self.assertTrue(
                    torch.equal(out.parent_idx[span][once], parents[once]),
                    f"parents {widths} {depth}",
                )
            self.assertGreater(tied, 0, "the case must contain ties")

    def test_graph_capture(self):
        case = make_case(
            widths=[256, 64, 8], size=256, depth=1, final=False, seed=11, kv_len=1
        )
        ref = reference_select(case)
        table = case["req_to_token"].clone()
        args = {k: v for k, v in case.items() if k not in KV_ARGS}
        args.update(
            req_to_token=table,
            kv_rows=case["kv_rows"],
            kv_starts=case["kv_starts"],
            kv_len=1,
        )
        radix_select_trie_beams(**args)
        table.copy_(case["req_to_token"])
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = radix_select_trie_beams(**args)
        for _ in range(2):
            table.copy_(case["req_to_token"])
            graph.replay()
            self.assert_same(out, table, ref, "graph replay")

    def test_new_widths_and_views_compile_nothing(self):
        """A beam width or an input alignment the server has not seen yet must reuse the
        compiled kernels: a compile while serving stalls a step for seconds."""
        from sglang.kernels.ops.sampling import beam_radix_select as module

        def shifted(t):
            # The same values one element past a fresh allocation: 4 or 8 bytes off 16.
            flat = torch.empty(t.numel() + 1, dtype=t.dtype, device=t.device)
            flat[1:] = t.flatten()
            return flat[1:].view(t.shape)

        kernels = (
            module._select_small_kernel,
            module._keys_kernel,
            module._hist2_kernel,
            module._collect_kernel,
            module._finalize_kernel,
        )
        device = torch.cuda.current_device()

        def compiled():
            return [len(k.device_caches[device][0]) for k in kernels]

        for depth, final, kv_len in ((0, False, 0), (1, False, 1), (2, True, 0)):
            with self.subTest(depth=depth):
                kernel_select(
                    make_case(
                        widths=[8, 300],
                        size=256,
                        depth=depth,
                        final=final,
                        seed=1,
                        kv_len=kv_len,
                    )
                )
                before = compiled()
                for widths in ([1], [16, 17, 2000], [64] * 5, [3, 1000, 256, 31]):
                    case = make_case(
                        widths=widths,
                        size=256,
                        depth=depth,
                        final=final,
                        seed=sum(widths),
                        kv_len=kv_len,
                    )
                    views = dict(case)
                    for name in (
                        "leader_normalizer",
                        "tail_normalizer",
                        "cum_logprobs",
                        "prefix_codes",
                        "kv_rows",
                    ):
                        if (
                            isinstance(case.get(name), torch.Tensor)
                            and case[name].numel()
                        ):
                            views[name] = shifted(case[name])
                    out, table = kernel_select(views)
                    self.assert_same(
                        out, table, reference_select(case), f"{widths} depth {depth}"
                    )
                self.assertEqual(compiled(), before)


if __name__ == "__main__":
    unittest.main()
