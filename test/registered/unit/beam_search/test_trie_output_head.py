import sys

import pytest
import torch

from sglang.srt.beam_search.mixed_requests import BeamDenseRows
from sglang.srt.beam_search.trie_config import TrieOutputHeadConfig
from sglang.srt.beam_search.trie_output_head import BeamTrieOutputHead
from sglang.srt.environ import envs
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


def _config():
    return TrieOutputHeadConfig(
        tensor_path="/unused/trie_output_head.safetensors",
        token_start=8,
        codebook_size=4,
        num_codebooks=2,
    )


def test_full_projection_preserves_scores():
    torch.manual_seed(0)
    hidden_states = torch.randn(3, 5)
    weight = torch.randn(16, 5)
    depths = torch.tensor([0, 1, 0])

    def project(states, selected_weight):
        return torch.matmul(states, selected_weight.T)

    def transform(logits):
        return logits

    chunked = BeamTrieOutputHead(_config(), vocab_size=16, chunk_size=3)
    full = BeamTrieOutputHead(_config(), vocab_size=16, chunk_size=16)

    chunked_output = chunked.project_mixed(
        hidden_states,
        weight,
        depths,
        project,
        transform,
    )
    full_output = full.project_mixed(
        hidden_states,
        weight,
        depths,
        project,
        transform,
    )

    assert torch.equal(full_output.logits, chunked_output.logits)
    assert torch.allclose(
        full_output.normalizer,
        chunked_output.normalizer,
        rtol=1e-6,
        atol=1e-6,
    )


def test_topk_logprob_skips_full_vocabulary_normalizer():
    torch.manual_seed(0)
    hidden_states = torch.randn(3, 5)
    weight = torch.randn(16, 5)
    depths = torch.tensor([0, 1, 0])
    projected_widths = []

    def project(states, selected_weight):
        projected_widths.append(selected_weight.shape[0])
        return torch.matmul(states, selected_weight.T)

    with envs.SGLANG_OPT_BEAM_TRIE_TOPK_LOGPROB.override(True):
        output = BeamTrieOutputHead(_config(), vocab_size=16).project_mixed(
            hidden_states,
            weight,
            depths,
            project,
            lambda logits: logits,
        )

    assert output.normalizer is None
    assert projected_widths == [8]


def test_kept_rows_are_the_transformed_full_vocabulary_logits():
    """Ordinary requests in a beam step sample from the rows the normalizer pass
    keeps: they must equal the dense LM head after the head's transform, across
    chunk edges, while compact logits and normalizers stay as without keeping. A
    captured buffer keeps only the rows its count names."""
    torch.manual_seed(0)
    hidden_states = torch.randn(4, 5)
    weight = torch.randn(18, 5)
    depths = torch.tensor([0, 1, 0, 1])

    def project(states, selected_weight):
        return torch.matmul(states, selected_weight.T)

    def transform(logits):
        return logits * 0.5

    head = BeamTrieOutputHead(_config(), vocab_size=16, chunk_size=3)
    plain = head.project_mixed(hidden_states, weight, depths, project, transform)
    kept = head.project_mixed(
        hidden_states,
        weight,
        depths,
        project,
        transform,
        dense_rows=BeamDenseRows(num_rows=3),
    )

    # Chunk by chunk, as the head projects: one GEMM over all 16 columns rounds
    # differently on CPU.
    dense = torch.cat(
        [
            transform(project(hidden_states, weight[start : min(start + 3, 16)]))
            for start in range(0, 16, 3)
        ],
        dim=1,
    )
    assert torch.equal(kept.dense_logits, dense[:3])
    assert torch.equal(kept.logits, plain.logits)
    assert torch.equal(kept.normalizer, plain.normalizer)

    captured = BeamDenseRows(
        num_rows=3,
        count=torch.tensor([1], dtype=torch.int32),
        logits=torch.full((3, 16), -7.0),
    )
    head.project_mixed(
        hidden_states, weight, depths, project, transform, dense_rows=captured
    )
    assert torch.equal(captured.logits[:1], dense[:1])
    assert bool((captured.logits[1:] == -7.0).all())


def test_kept_rows_need_the_full_vocabulary_pass():
    """Approximate mode computes no full-vocabulary logits to keep: asking for
    kept rows fails instead of silently returning none."""
    with envs.SGLANG_OPT_BEAM_TRIE_TOPK_LOGPROB.override(True):
        with pytest.raises(RuntimeError, match="normalizer pass"):
            BeamTrieOutputHead(_config(), vocab_size=16).project_mixed(
                torch.randn(2, 5),
                torch.randn(16, 5),
                torch.tensor([0, 1]),
                lambda states, selected_weight: torch.matmul(states, selected_weight.T),
                lambda logits: logits,
                dense_rows=BeamDenseRows(num_rows=1),
            )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__]))
