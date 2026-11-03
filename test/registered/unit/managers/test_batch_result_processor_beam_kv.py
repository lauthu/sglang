import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch

from sglang.srt.managers.scheduler_components.batch_result_processor import (
    SchedulerBatchResultProcessor,
)
from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode
from sglang.srt.runtime_context import get_context
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=2, suite="base-a-test-cpu")

_MODULE = "sglang.srt.managers.scheduler_components.batch_result_processor"


def _make_processor(parent: Mock) -> SchedulerBatchResultProcessor:
    return SchedulerBatchResultProcessor(
        is_generation=True,
        disaggregation_mode=None,
        enable_overlap=False,
        enable_overlap_mlx=False,
        model_config=SimpleNamespace(think_end_ids=None),
        token_to_kv_pool_allocator=parent.allocator,
        tree_cache=None,
        hisparse_coordinator=None,
        req_to_token_pool=None,
        decode_offload_manager=None,
        metrics_collector=None,
        metrics_reporter=Mock(num_generated_tokens=0, forward_ct_decode=0),
        draft_worker=None,
        model_worker=Mock(),
        logprob_result_processor=None,
        output_streamer=parent.output_streamer,
        beam_coordinator=parent.beam_coordinator,
        abort_request=lambda *args, **kwargs: None,
    )


def _result(**fields):
    return SimpleNamespace(
        copy_done=None,
        auxiliary_host_output=None,
        routed_experts_output=None,
        indexer_topk_output=None,
        can_run_cuda_graph=False,
        **fields,
    )


class TestFinishedBeamKvFreedAfterStreaming(CustomTestCase):
    """Freeing a finished beam group's member KV syncs the host, so each result
    pass frees it only after streaming its outputs, and within the pass, or the
    slots would leak."""

    def _assert_called_in_order(self, parent: Mock, *names: str):
        called = [name for name, _, _ in parent.mock_calls]
        positions = [called.index(name) for name in names]
        self.assertEqual(positions, sorted(positions))

    def test_decode_pass(self):
        parent = Mock()
        group = SimpleNamespace()
        parent.beam_coordinator.commit_decode.return_value = {id(group)}
        leader = SimpleNamespace(
            beam_group=group, finished=lambda: True, time_stats=Mock()
        )
        batch = SimpleNamespace(
            reqs=[leader],
            return_logprob=False,
            spec_algorithm=SimpleNamespace(is_none=lambda: True),
            batch_size=lambda: 1,
        )
        result = _result(
            logits_output=SimpleNamespace(),
            next_token_ids=None,
            num_correct_drafts=0,
        )
        with (
            patch.object(
                SchedulerBatchResultProcessor,
                "_normalize_decode_outputs",
                return_value=([[0]], None),
            ),
            patch.object(
                SchedulerBatchResultProcessor, "_handle_finish_state_updated_req"
            ),
            patch(
                f"{_MODULE}.get_observability",
                return_value=SimpleNamespace(enable_metrics=False),
            ),
        ):
            _make_processor(parent).process_batch_result_decode(batch, result)

        self._assert_called_in_order(
            parent,
            "beam_coordinator.commit_decode",
            "output_streamer.stream_output",
            "beam_coordinator.free_finished_member_kv",
            "allocator.free_group_end",
        )

    def test_prefill_pass(self):
        # The prefill pass reads the server's hidden-state ceiling.
        override = get_context().override_server_args(enable_return_hidden_states=False)
        override.install()
        self.addCleanup(override.restore)
        parent = Mock()
        req = SimpleNamespace(
            rid="r0",
            inflight_middle_chunks=0,
            return_hidden_states=False,
            hidden_states=[],
            is_retracted=False,
            output_ids=[],
            time_stats=Mock(),
            return_logprob=False,
            return_sampling_mask=False,
            grammar=None,
            require_reasoning=False,
            customized_info=None,
            beam_group=None,
            finished=lambda: False,
            update_finish_state=lambda: None,
        )
        batch = SimpleNamespace(
            reqs=[req],
            decoding_reqs=[],
            return_logprob=False,
            return_hidden_states=False,
            return_hidden_states_mode=CaptureHiddenMode.NULL,
            spec_info=None,
            prefill_stats=None,
            dp_cooperation_info=None,
        )
        result = _result(
            logits_output=SimpleNamespace(hidden_states=None, customized_info=None),
            next_token_ids=torch.tensor([0]),
            extend_input_len_per_req=[2],
            extend_logprob_start_len_per_req=None,
            grammar_advanced=False,
            skipped_output_comm=False,
        )
        with (
            patch(f"{_MODULE}.maybe_cache_unfinished_req"),
            patch(
                f"{_MODULE}.get_memory",
                return_value=SimpleNamespace(enable_hisparse=False),
            ),
        ):
            _make_processor(parent).process_batch_result_prefill(batch, result)

        self._assert_called_in_order(
            parent,
            "output_streamer.stream_output",
            "beam_coordinator.free_finished_member_kv",
        )


if __name__ == "__main__":
    unittest.main()
