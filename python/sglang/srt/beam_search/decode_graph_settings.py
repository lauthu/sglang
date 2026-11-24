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
"""Which decode batches the beam decode graphs replay, as the settings choose.

With SGLANG_ENABLE_BEAM_DECODE_GRAPH on and shared-context attention:

- SGLANG_BEAM_DECODE_GRAPH_FIXED_WIDTH set: graphs for requests of that one beam
  width.
- Unset: width-free graphs, for beam requests of any widths. Shared-context
  attention then runs its width-free layout for every beam decode batch, so a
  batch gets the same kernels whether or not a graph replays it.

SGLANG_ENABLE_BEAM_DECODE_GRAPH_MIXED_BATCH also keeps batches that mix beam
and regular requests on the beam path, and so on the beam graphs.
"""

from __future__ import annotations

import logging

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_WIDTH_FREE_KERNELS = ("default", "stream_k")
_WIDTH_FREE_GRAPHS_HOWTO = (
    "set SGLANG_ENABLE_BEAM_DECODE_GRAPH and leave "
    "SGLANG_BEAM_DECODE_GRAPH_FIXED_WIDTH unset"
)


def width_free_graphs_enabled() -> bool:
    return (
        envs.SGLANG_ENABLE_BEAM_DECODE_GRAPH.get()
        and envs.SGLANG_BEAM_DECODE_GRAPH_FIXED_WIDTH.get() is None
    )


def stream_k_selected() -> bool:
    """The width-free layout runs the stream-K kernel, not the tile-per-CTA one."""
    return envs.SGLANG_BEAM_SHARED_CONTEXT_KERNEL.get() == "stream_k"


def mixed_batch_enabled() -> bool:
    return envs.SGLANG_ENABLE_BEAM_DECODE_GRAPH_MIXED_BATCH.get()


def max_mixed_batch_requests() -> int:
    field = envs.SGLANG_BEAM_DECODE_GRAPH_MIXED_BATCH_MAX_REQUESTS
    value = field.get()
    if value <= 0:
        raise ValueError(f"{field.name} must be positive, got {value}")
    return value


def check_settings(*, deterministic_inference: bool) -> None:
    """Startup check: raise on settings that cannot run together, and name a
    setting that has no effect."""
    kernel = envs.SGLANG_BEAM_SHARED_CONTEXT_KERNEL.get()
    if kernel not in _WIDTH_FREE_KERNELS:
        raise ValueError(
            f"SGLANG_BEAM_SHARED_CONTEXT_KERNEL={kernel!r} is not one of "
            f"{', '.join(_WIDTH_FREE_KERNELS)}"
        )
    width_free = width_free_graphs_enabled()
    if kernel != "default" and not width_free:
        raise ValueError(
            f"SGLANG_BEAM_SHARED_CONTEXT_KERNEL={kernel!r} is a kernel of the "
            f"width-free beam decode graphs: {_WIDTH_FREE_GRAPHS_HOWTO}"
        )
    if mixed_batch_enabled():
        if not width_free:
            raise ValueError(
                "SGLANG_ENABLE_BEAM_DECODE_GRAPH_MIXED_BATCH needs the width-free "
                f"beam decode graphs: {_WIDTH_FREE_GRAPHS_HOWTO}"
            )
        if deterministic_inference:
            raise ValueError(
                "SGLANG_ENABLE_BEAM_DECODE_GRAPH_MIXED_BATCH computes a regular "
                "request's logits on the beam path whenever a beam request shares "
                "its decode batch, so they vary with the batch: not with "
                "--enable-deterministic-inference"
            )
        max_mixed_batch_requests()
    max_new_tokens = envs.SGLANG_BEAM_DECODE_GRAPH_FIXED_WIDTH_MAX_NEW_TOKENS
    if width_free and max_new_tokens.is_set():
        logger.warning(
            "%s has no effect without SGLANG_BEAM_DECODE_GRAPH_FIXED_WIDTH: the "
            "width-free beam decode graphs serve any decode length.",
            max_new_tokens.name,
        )
