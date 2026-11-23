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
"""One trie beam selection step: the radix-select kernels against the per-group torch path.

usage: python bench_beam_radix_select.py [--codebook 256] [--replays 200] [--json OUT]

Per shape it prints the GPU time of one step from CUDA graph replays, and the host time of
one eager call (Python plus kernel launches, without waiting for the GPU).

L2: the inputs stay in L2 between replays. That is the serving pattern for most of what a
step reads: the logits and normalizers were written by the LM head just before it. The
frontier state and the trie mask rows it touches (under 1 MB per step) would be cold in
serving, so read the result next to a served profile's selection time.
"""

import argparse
import json
import time

import torch

from sglang.kernels.ops.sampling.beam_radix_select import radix_select_trie_beams
from sglang.srt.beam_search.fork import remap_kv_mapping
from sglang.srt.beam_search.joint_select import joint_select, select_final_topk
from sglang.srt.beam_search.trie_config import TrieOutputHeadConfig
from sglang.srt.beam_search.trie_constraint import BeamTrieConstraint

TOKEN_START = 1000
VOCAB = 151_936
# (name, depth, beam widths of the groups in the step)
SHAPES = [
    ("first", 0, [16]),
    ("first", 0, [256]),
    ("first", 0, [2000]),
    ("middle", 1, [4]),
    ("middle", 1, [16]),
    ("middle", 1, [64]),
    ("middle", 1, [256]),
    ("middle", 1, [1000]),
    ("middle", 1, [2000]),
    ("middle", 1, [16] * 8),
    ("middle", 1, [64] * 8),
    ("middle", 1, [256] * 8),
    ("middle", 1, [2000] * 8),
    ("middle", 1, [16, 64, 256]),
    ("middle", 1, [16, 64, 256, 16, 64, 256, 16, 64]),
    ("middle", 1, [8, 2000, 300, 17]),
    ("final", 2, [16]),
    ("final", 2, [256]),
    ("final", 2, [2000]),
    ("final", 2, [256] * 8),
    ("final", 2, [16, 64, 256, 16, 64, 256, 16, 64]),
]


def make_step(depth, widths, size, seed=0):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    groups = len(widths)
    rows = [1 if depth == 0 else width for width in widths]
    total = sum(rows)
    if depth == 0:
        leader = torch.randn(groups, VOCAB, generator=gen)
        tail = None
    else:
        leader = torch.randn(groups, size, generator=gen).to(torch.bfloat16).float()
        tail = (
            torch.randn(total - groups, size, generator=gen).to(torch.bfloat16).float()
        )

    def normalizer(logits):
        return torch.stack(
            (logits.max(dim=1).values, torch.rand(logits.shape[0], generator=gen) + 3),
            1,
        )

    masks = torch.rand(size**depth, size, generator=gen) < 0.9
    step = dict(
        depth=depth,
        widths=widths,
        size=size,
        rows=rows,
        leader=leader.cuda(),
        tail=None if tail is None else tail.cuda(),
        leader_norm=None if depth == 0 else normalizer(leader).cuda(),
        tail_norm=None if depth == 0 else normalizer(tail).cuda(),
        tail_starts=[sum(rows[:g]) - g for g in range(groups)],
        cum=[
            (-(torch.rand(r, generator=gen) * 8).sort().values * (depth > 0)).cuda()
            for r in rows
        ],
        prefix=[torch.randint(0, size, (r, depth), generator=gen).cuda() for r in rows],
        masks=masks.cuda(),
        req_to_token=torch.randint(
            0, 1 << 20, (total + 8, 64), generator=gen, dtype=torch.int32
        ).cuda(),
        kv_rows=[t.cuda() for t in torch.randperm(total, generator=gen).split(rows)],
    )
    step["trie"] = BeamTrieConstraint(
        config=TrieOutputHeadConfig(
            tensor_path="", token_start=TOKEN_START, codebook_size=size, num_codebooks=3
        ),
        root_mask=step["masks"][0]
        if depth == 0
        else torch.ones(size, dtype=torch.bool, device="cuda"),
        level1_masks=step["masks"] if depth == 1 else None,
        level2_masks=step["masks"] if depth == 2 else None,
        level2_sparse=None,
        level_cardinalities=(size, size * size, size**3),
    )
    return step


def torch_step(step):
    """Upstream's path: per group, trie top-k, joint selection, prefix advance and KV remap."""
    depth, trie = step["depth"], step["trie"]
    no_stop = torch.empty(0, dtype=torch.int64, device="cuda")
    out = []
    for g, width in enumerate(step["widths"]):
        pieces, normalizers = [step["leader"][g : g + 1]], None
        if depth:
            start = step["tail_starts"][g]
            pieces.append(step["tail"][start : start + width - 1])
            normalizers = [
                step["leader_norm"][g : g + 1],
                step["tail_norm"][start : start + width - 1],
            ]
        values, tokens = trie.topk_logprobs(
            pieces, step["prefix"][g], depth, width, normalizers
        )
        if depth == 2:
            out.append(select_final_topk(step["cum"][g], values, tokens, width))
            continue
        sel = joint_select(
            step["cum"][g], values, tokens, no_stop, width, num_output_candidates=width
        )
        out.append(
            (
                sel,
                trie.advance_prefixes(
                    step["prefix"][g], sel.parent_idx, sel.next_tokens, depth
                ),
            )
        )
        if depth == 1:
            out.append(
                remap_kv_mapping(
                    step["req_to_token"], step["kv_rows"][g], sel.parent_idx, 5, 6
                )
            )
    return out


def radix_step(step):
    """The opt-in path: one call for every group; the first step adds its row's log-sum-exp."""
    depth, size = step["depth"], step["size"]
    groups = len(step["widths"])
    if depth == 0:
        outs = []
        for g in range(groups):
            x = step["leader"][g : g + 1]
            lse = torch.logsumexp(x, dim=-1, keepdim=True)
            outs.append(
                radix_select_trie_beams(
                    leader_logits=x[:, TOKEN_START : TOKEN_START + size],
                    leader_normalizer=torch.nn.functional.pad(lse, (1, 0)),
                    leader_rows=[0],
                    tail_logits=None,
                    tail_normalizer=None,
                    tail_starts=None,
                    beam_widths=[step["widths"][g]],
                    cum_logprobs=step["cum"][g],
                    prefix_codes=step["prefix"][g],
                    child_masks=step["masks"],
                    codebook_size=size,
                    token_start=TOKEN_START,
                    final=False,
                )
            )
        return outs
    one = groups == 1
    return radix_select_trie_beams(
        leader_logits=step["leader"],
        leader_normalizer=step["leader_norm"],
        leader_rows=list(range(groups)),
        tail_logits=step["tail"],
        tail_normalizer=step["tail_norm"],
        tail_starts=step["tail_starts"],
        beam_widths=step["widths"],
        cum_logprobs=step["cum"][0] if one else torch.cat(step["cum"]),
        prefix_codes=step["prefix"][0] if one else torch.cat(step["prefix"]),
        child_masks=step["masks"],
        codebook_size=size,
        token_start=TOKEN_START + depth * size,
        final=depth == 2,
        req_to_token=step["req_to_token"],
        kv_rows=step["kv_rows"][0] if one else torch.cat(step["kv_rows"]),
        kv_starts=[5] * groups,
        kv_len=1 if depth == 1 else 0,
    )


def gpu_us(fn, step, replays):
    fn(step)
    fn(step)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        keep = fn(step)
    for _ in range(20):
        graph.replay()
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    start.record()
    for _ in range(replays):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    del keep
    return start.elapsed_time(end) * 1e3 / replays


def host_us(fn, step, calls=100):
    for _ in range(10):
        fn(step)
    torch.cuda.synchronize()
    begin = time.perf_counter()
    for _ in range(calls):
        fn(step)
    elapsed = time.perf_counter() - begin
    torch.cuda.synchronize()
    return elapsed * 1e6 / calls


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--codebook", type=int, default=256)
    parser.add_argument("--replays", type=int, default=200)
    parser.add_argument("--json")
    # A GPU just out of idle reads slow for its first seconds.
    parser.add_argument("--warmup-seconds", type=float, default=20.0)
    args = parser.parse_args()
    warm = make_step(1, [2000] * 8, args.codebook)
    until = time.perf_counter() + args.warmup_seconds
    while time.perf_counter() < until:
        torch_step(warm)
        torch.cuda.synchronize()
    print(
        f"{torch.cuda.get_device_name(0)}, torch {torch.__version__}, codebook {args.codebook}"
    )
    print(
        "| Step | Beam widths in the step | torch GPU (us) | radix GPU (us) | GPU ratio | torch host (us) | radix host (us) |"
    )
    print("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    rows = []
    for name, depth, widths in SHAPES:
        step = make_step(depth, widths, args.codebook)
        result = dict(
            step=name,
            widths=widths,
            torch_gpu=gpu_us(torch_step, step, args.replays),
            radix_gpu=gpu_us(radix_step, step, args.replays),
            torch_host=host_us(torch_step, step),
            radix_host=host_us(radix_step, step),
        )
        rows.append(result)
        label = (
            f"{len(widths)} x {widths[0]}"
            if len(set(widths)) == 1
            else " + ".join(map(str, widths))
        )
        print(
            f"| {name} | {label} | {result['torch_gpu']:.1f} | {result['radix_gpu']:.1f} | "
            f"{result['torch_gpu'] / result['radix_gpu']:.1f}x | {result['torch_host']:.0f} | "
            f"{result['radix_host']:.0f} |",
            flush=True,
        )
    if args.json:
        with open(args.json, "w") as out:
            json.dump(rows, out, indent=1)


if __name__ == "__main__":
    main()
