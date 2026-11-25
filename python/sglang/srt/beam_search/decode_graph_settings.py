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
"""How the beam decode graph settings combine, and which shared-context layout
the server runs as a result.

With SGLANG_ENABLE_BEAM_DECODE_GRAPH on and shared-context attention:

- SGLANG_BEAM_DECODE_GRAPH_FIXED_WIDTH set: graphs for requests of that one beam
  width.
- Unset: variable-width graphs, for beam requests of any widths. Every beam decode
  batch then gets the variable-width layout and its kernels, so a batch runs the
  same kernels whether or not a graph replays it.
"""

from __future__ import annotations

import logging

from sglang.srt.environ import envs

logger = logging.getLogger(__name__)

_VARIABLE_WIDTH_KERNELS = ("default", "stream_k")


def variable_width_graphs_enabled() -> bool:
    """SGLANG_ENABLE_BEAM_DECODE_GRAPH is on with no fixed width. With
    shared-context attention on, which the callers check, the beam decode graphs
    then take any beam width, and every beam decode batch gets the variable-width
    layout, replayed or not."""
    return (
        envs.SGLANG_ENABLE_BEAM_DECODE_GRAPH.get()
        and envs.SGLANG_BEAM_DECODE_GRAPH_FIXED_WIDTH.get() is None
    )


def stream_k_selected() -> bool:
    """Whether the variable-width layout runs its stream-K kernel, which spreads a
    batch's work evenly over the GPU, instead of its default kernel."""
    return envs.SGLANG_BEAM_SHARED_CONTEXT_KERNEL.get() == "stream_k"


def check_settings() -> None:
    """Startup check: raise on settings that cannot run together, and name a
    setting that has no effect."""
    kernel = envs.SGLANG_BEAM_SHARED_CONTEXT_KERNEL.get()
    variable_width_graphs = variable_width_graphs_enabled()
    # With a fixed width, or without the graphs, shared_context_attention checks
    # the name against its own three kernels.
    if variable_width_graphs and kernel not in _VARIABLE_WIDTH_KERNELS:
        raise ValueError(
            f"SGLANG_BEAM_SHARED_CONTEXT_KERNEL={kernel!r} is not available when "
            "the beam decode graphs take any beam width "
            "(SGLANG_BEAM_DECODE_GRAPH_FIXED_WIDTH unset): pick one of "
            f"{', '.join(_VARIABLE_WIDTH_KERNELS)}, or set the fixed width"
        )
    max_new_tokens_setting = envs.SGLANG_BEAM_DECODE_GRAPH_FIXED_WIDTH_MAX_NEW_TOKENS
    if variable_width_graphs and max_new_tokens_setting.is_set():
        logger.warning(
            "%s has no effect without SGLANG_BEAM_DECODE_GRAPH_FIXED_WIDTH: the "
            "beam decode graphs for any beam width have no token limit.",
            max_new_tokens_setting.name,
        )
