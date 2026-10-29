import pytest
import torch

from sglang.srt.beam_search.joint_select import joint_select, select_final_topk
from sglang.srt.beam_search.trie_config import TrieOutputHeadConfig
from sglang.srt.beam_search.trie_constraint import BeamTrieConstraint
from sglang.srt.beam_search.trie_select import (
    fused_select_supported,
    select_trie_beams,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=30, stage="base-b", runner_config="1-gpu-large")

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

TOKEN_START = 1000
NUM_CODEBOOKS = 3


def _constraint(codebook_size, density, generator):
    def masks(shape):
        mask = torch.rand(shape, generator=generator) < density
        # Every prefix keeps a child, as validated tries do.
        rows = mask.view(-1, codebook_size)
        empty = (~rows.any(dim=1)).nonzero().flatten()
        rows[
            empty, torch.randint(codebook_size, (len(empty),), generator=generator)
        ] = True
        return mask.cuda()

    config = TrieOutputHeadConfig(
        tensor_path="",
        token_start=TOKEN_START,
        codebook_size=codebook_size,
        num_codebooks=NUM_CODEBOOKS,
    )
    return BeamTrieConstraint(
        config=config,
        root_mask=masks((codebook_size,)),
        level1_masks=masks((codebook_size, codebook_size)),
        level2_masks=masks((codebook_size * codebook_size, codebook_size)),
        level2_sparse=None,
        level_cardinalities=(codebook_size, codebook_size, codebook_size),
    )


def _child_masks(trie, depth):
    if depth == 0:
        return trie.root_mask.unsqueeze(0)
    return trie.level1_masks if depth == 1 else trie.level2_masks


def _random_prefixes(trie, num_rows, depth, generator):
    # Walk valid children so every frontier row sits on the trie.
    size = trie.config.codebook_size
    codes = torch.empty((num_rows, depth), dtype=torch.int64)
    for row in range(num_rows):
        mask = trie.root_mask.cpu()
        for d in range(depth):
            choices = mask.nonzero().flatten()
            code = choices[torch.randint(len(choices), (1,), generator=generator)]
            codes[row, d] = code
            mask = (
                trie.level1_masks[code].cpu()
                if d == 0
                else trie.level2_masks[codes[row, 0] * size + code].cpu()
            ).flatten()
    return codes.cuda()


def _torch_path(trie, pieces, normalizers, cum, prefix, depth, beam_width, final):
    top_logprobs, top_tokens = trie.topk_logprobs(
        pieces, prefix, depth, beam_width, normalizers
    )
    if final:
        sel = select_final_topk(cum, top_logprobs, top_tokens, beam_width)
        return sel.tokens, sel.parent_idx, sel.cum_logprobs, None
    sel = joint_select(
        cum,
        top_logprobs,
        top_tokens,
        torch.tensor([7], dtype=torch.int64, device="cuda"),
        beam_width,
        num_output_candidates=beam_width,
    )
    next_prefix = trie.advance_prefixes(prefix, sel.parent_idx, sel.next_tokens, depth)
    return sel.next_tokens, sel.parent_idx, sel.new_cum_logprobs, next_prefix


def _assert_same_selection(fused, g, expected):
    # torch.topk orders equal scores arbitrarily for k <= 32 (an unstable
    # bitonic sort), so compare per run of equal scores; the kernel itself
    # orders ties by flat candidate index, which keeps parents ascending.
    tokens, parents, scores, next_prefix = expected
    assert torch.equal(fused.cum_logprobs[g], scores)
    runs = torch.unique_consecutive(scores.cpu(), return_counts=True)[1].tolist()
    start = 0
    for count in runs:
        span = slice(start, start + count)
        start += count

        def rows(tok, par, pre):
            cols = [par[span, None], tok[span, None]]
            if pre is not None:
                cols.append(pre[span])
            return sorted(map(tuple, torch.cat(cols, dim=1).tolist()))

        fused_prefix = None if fused.prefix_codes is None else fused.prefix_codes[g]
        assert rows(fused.tokens[g], fused.parent_idx[g], fused_prefix) == rows(
            tokens, parents, next_prefix
        )
        assert torch.all(fused.parent_idx[g][span].diff() >= 0)
    assert (fused.prefix_codes is None) == (next_prefix is None)


def _decode_case(*, codebook_size, beam_width, depth, num_groups, density, dtype, seed):
    generator = torch.Generator().manual_seed(seed)
    trie = _constraint(codebook_size, density, generator)
    num_tail = num_groups * (beam_width - 1)
    leader_logits = (
        (torch.randn((num_groups, codebook_size), generator=generator) * 4)
        .to(dtype)
        .cuda()
    )
    tail_logits = (
        (torch.randn((num_tail + 3, codebook_size), generator=generator) * 4)
        .to(dtype)
        .cuda()
    )
    leader_norm = torch.randn((num_groups, 2), generator=generator).abs().cuda() + 10
    tail_norm = torch.randn((num_tail + 3, 2), generator=generator).abs().cuda() + 10
    cum = -(torch.rand((num_groups, beam_width), generator=generator) * 20).cuda()
    prefix = torch.stack(
        [
            _random_prefixes(trie, beam_width, depth, generator)
            for _ in range(num_groups)
        ]
    )
    # Groups' member rows sit at shuffled offsets in the tail, as batch order sets them.
    starts = [3 + g * (beam_width - 1) for g in reversed(range(num_groups))]
    leaders = list(reversed(range(num_groups)))
    return (
        trie,
        leader_logits,
        tail_logits,
        leader_norm,
        tail_norm,
        cum,
        prefix,
        starts,
        leaders,
    )


@pytest.mark.parametrize(
    "codebook_size,beam_width,depth,num_groups,density,dtype",
    [
        (256, 256, 1, 1, 0.9, torch.float32),
        (256, 256, 2, 3, 0.06, torch.float32),
        (256, 256, 1, 2, 0.5, torch.bfloat16),
        (64, 32, 2, 2, 0.3, torch.bfloat16),
        (32, 8, 1, 2, 0.03, torch.float32),
    ],
)
@pytest.mark.parametrize("final", [False, True])
def test_decode_matches_torch_path(
    codebook_size, beam_width, depth, num_groups, density, dtype, final
):
    """Fused selection equals topk_logprobs + joint_select / select_final_topk
    + advance_prefixes bit for bit, -inf winners and their tokens included, up to
    the order of equal scores."""
    (
        trie,
        leader_logits,
        tail_logits,
        leader_norm,
        tail_norm,
        cum,
        prefix,
        starts,
        leaders,
    ) = _decode_case(
        codebook_size=codebook_size,
        beam_width=beam_width,
        depth=depth,
        num_groups=num_groups,
        density=density,
        dtype=dtype,
        seed=depth * 100 + codebook_size + num_groups,
    )
    fused = select_trie_beams(
        leader_logits=leader_logits,
        tail_logits=tail_logits,
        leader_normalizer=leader_norm,
        tail_normalizer=tail_norm,
        frontier_rows=torch.tensor(list(zip(leaders, starts)), device="cuda"),
        cum_logprobs=cum,
        prefix_codes=prefix,
        child_masks=_child_masks(trie, depth),
        codebook_size=codebook_size,
        code_offset=0,
        token_start=trie.config.token_start_for_depth(depth),
        beam_width=beam_width,
        final=final,
    )
    for g in range(num_groups):
        rows = slice(starts[g], starts[g] + beam_width - 1)
        tokens, parents, scores, next_prefix = _torch_path(
            trie,
            [leader_logits[leaders[g] : leaders[g] + 1], tail_logits[rows]],
            [leader_norm[leaders[g] : leaders[g] + 1], tail_norm[rows]],
            cum[g],
            prefix[g],
            depth,
            beam_width,
            final,
        )
        _assert_same_selection(fused, g, (tokens, parents, scores, next_prefix))
        if beam_width > 32:
            # Stable torch sorts: ties fall in the same order on both paths.
            assert torch.equal(fused.tokens[g], tokens)
            assert torch.equal(fused.parent_idx[g], parents)


@pytest.mark.parametrize("root_density", [1.0, 0.7])
def test_prefill_matches_torch_path(root_density):
    """Depth 0 scores the codebook slice of dense logits with a per-row
    log-sum-exp, as the torch path does for prefill."""
    generator = torch.Generator().manual_seed(7)
    codebook_size = beam_width = 256
    trie = _constraint(codebook_size, root_density, generator)
    vocab = TOKEN_START + NUM_CODEBOOKS * codebook_size + 500
    logits = (
        (torch.randn((2, vocab), generator=generator) * 4).to(torch.bfloat16).cuda()
    )
    piece = logits[1:2]
    lse = torch.logsumexp(piece.float(), dim=-1, keepdim=True)
    cum = torch.zeros((1, 1), device="cuda")
    prefix = torch.empty((1, 1, 0), dtype=torch.int64, device="cuda")
    fused = select_trie_beams(
        leader_logits=piece,
        tail_logits=piece,
        leader_normalizer=lse,
        tail_normalizer=lse,
        frontier_rows=torch.zeros((1, 2), dtype=torch.int64, device="cuda"),
        cum_logprobs=cum,
        prefix_codes=prefix,
        child_masks=_child_masks(trie, 0),
        codebook_size=codebook_size,
        code_offset=TOKEN_START,
        token_start=TOKEN_START,
        beam_width=beam_width,
        final=False,
    )
    expected = _torch_path(trie, [piece], None, cum[0], prefix[0], 0, beam_width, False)
    _assert_same_selection(fused, 0, expected)
    assert torch.equal(fused.tokens[0], expected[0])


def test_shapes_beyond_one_program_take_the_torch_path():
    """The merge sorts every band's keys in one program; wider beams fall back
    rather than compile a sort that spills."""
    assert fused_select_supported(beam_width=256, num_rows=256, codebook_size=256)
    assert fused_select_supported(beam_width=256, num_rows=1, codebook_size=256)
    assert not fused_select_supported(beam_width=512, num_rows=512, codebook_size=256)
    assert not fused_select_supported(beam_width=256, num_rows=1, codebook_size=128)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
