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

from sglang.srt.environ import envs

# The variable-width layout has no split-KV kernel.
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


def variable_width_kernel() -> str:
    """The kernel that SGLANG_BEAM_SHARED_CONTEXT_KERNEL selects for the
    variable-width layout: "default" or "stream_k"."""
    kernel = envs.SGLANG_BEAM_SHARED_CONTEXT_KERNEL.get()
    if kernel not in _VARIABLE_WIDTH_KERNELS:
        raise ValueError(
            f"SGLANG_BEAM_SHARED_CONTEXT_KERNEL is {kernel!r}, not one of "
            f"{_VARIABLE_WIDTH_KERNELS}: the beam decode graphs for any beam width "
            "(SGLANG_BEAM_DECODE_GRAPH_FIXED_WIDTH unset) have no other kernel."
        )
    return kernel
