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
"""SGLANG_BEAM_DECODE_GRAPH_LEVEL and the settings that go with a level.

The level selects the shared-context attention layout and whether ordinary
requests join beam decode steps; those are what let a beam decode graph replay
a step. A step that replays no graph runs the same kernels eagerly.
"""

from __future__ import annotations

import logging

from sglang.srt.environ import BeamDecodeGraphLevel, envs

logger = logging.getLogger(__name__)

_WIDTH_FREE_KERNELS = ("default", "stream_k")


def decode_graph_level() -> BeamDecodeGraphLevel:
    value = envs.SGLANG_BEAM_DECODE_GRAPH_LEVEL.get()
    if value == 1:
        raise ValueError(
            "SGLANG_BEAM_DECODE_GRAPH_LEVEL=1 (one width per step) has no mode of its "
            "own: use 2, which also replays steps of one width"
        )
    level = BeamDecodeGraphLevel(value)
    kernel = envs.SGLANG_BEAM_SHARED_CONTEXT_KERNEL.get()
    if kernel not in _WIDTH_FREE_KERNELS:
        raise ValueError(
            f"SGLANG_BEAM_SHARED_CONTEXT_KERNEL={kernel!r} is not one of "
            f"{', '.join(_WIDTH_FREE_KERNELS)}"
        )
    if kernel != "default" and level == BeamDecodeGraphLevel.FIXED_WIDTH:
        raise ValueError(
            f"SGLANG_BEAM_SHARED_CONTEXT_KERNEL={kernel!r} needs "
            "SGLANG_BEAM_DECODE_GRAPH_LEVEL 2 or 3: level 0 runs only the default kernel"
        )
    return level


def stream_k_selected() -> bool:
    """Levels 2 and 3 run the stream-K kernel instead of the tile-per-CTA one."""
    return envs.SGLANG_BEAM_SHARED_CONTEXT_KERNEL.get() == "stream_k"


def max_mixed_requests() -> int:
    field = envs.SGLANG_BEAM_DECODE_GRAPH_MAX_MIXED_REQUESTS
    value = field.get()
    if value <= 0:
        raise ValueError(f"{field.name} must be positive, got {value}")
    return value


def check_decode_graph_settings(*, deterministic_inference: bool) -> None:
    """Startup check: raise on a level, kernel or request cap the server cannot
    run, and name the settings a level leaves without effect."""
    level = decode_graph_level()
    if level == BeamDecodeGraphLevel.FIXED_WIDTH:
        return
    if level >= BeamDecodeGraphLevel.MIXED_REQUESTS:
        max_mixed_requests()
        if deterministic_inference:
            raise ValueError(
                "SGLANG_BEAM_DECODE_GRAPH_LEVEL=3 computes an ordinary request's "
                "logits on the beam path whenever a beam request shares its decode "
                "step, so they vary with the batch: not with "
                "--enable-deterministic-inference; use level 2"
            )
    if not envs.SGLANG_ENABLE_BEAM_DECODE_GRAPH.get():
        logger.warning(
            "SGLANG_BEAM_DECODE_GRAPH_LEVEL=%d is set without "
            "SGLANG_ENABLE_BEAM_DECODE_GRAPH: beam decode steps run its "
            "kernels eagerly and replay no graph.",
            level,
        )
    for unused in (
        envs.SGLANG_BEAM_DECODE_GRAPH_FIXED_WIDTH,
        envs.SGLANG_BEAM_DECODE_GRAPH_FIXED_WIDTH_MAX_NEW_TOKENS,
    ):
        if unused.is_set():
            logger.warning(
                "%s has no effect with SGLANG_BEAM_DECODE_GRAPH_LEVEL=%d: its beam "
                "decode graphs serve any beam width and decode length.",
                unused.name,
                level,
            )
