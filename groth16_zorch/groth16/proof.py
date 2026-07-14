# Copyright 2026 The Groth16Zorch Authors.
#
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

"""Groth16 proof data structure and serialization.

Provides the proof container and snarkjs-compatible JSON export.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from jax import Array

# Mapping from ZK dtype names to snarkjs curve names.
_DTYPE_TO_CURVE = {
    "bn254_g1_affine": "bn128",
    "bn254_g2_affine": "bn128",
}


def _curve_name(dtype) -> str:
    """Derive snarkjs curve name from a ZK dtype."""
    name = _DTYPE_TO_CURVE.get(dtype.name)
    if name is None:
        raise ValueError(f"Unknown curve for dtype {dtype}")
    return name


@dataclass
class Groth16Proof:
    """Groth16 proof containing three elliptic curve points.

    Attributes:
        pi_a: G1 point (affine representation).
        pi_b: G2 point (affine representation).
        pi_c: G1 point (affine representation).
    """

    pi_a: Array
    pi_b: Array
    pi_c: Array

    def to_json(self) -> dict:
        """Serialize to snarkjs-compatible JSON format.

        Formats as:
        ``{"pi_a": [x, y, "1"], "pi_b": [[x0,x1],[y0,y1],["1","0"]],
           "pi_c": [x, y, "1"], "protocol": "groth16", "curve": "bn128"}``
        """
        pi_a = np.array(self.pi_a).item()
        pi_b = np.array(self.pi_b).item()
        pi_c = np.array(self.pi_c).item()

        ax, ay = (int(v) for v in pi_a.raw)
        bx, by = pi_b.raw
        cx, cy = (int(v) for v in pi_c.raw)

        return {
            "pi_a": [str(ax), str(ay), "1"],
            "pi_b": [
                [str(int(bx[0])), str(int(bx[1]))],
                [str(int(by[0])), str(int(by[1]))],
                ["1", "0"],
            ],
            "pi_c": [str(cx), str(cy), "1"],
            "protocol": "groth16",
            "curve": _curve_name(self.pi_a.dtype),
        }


def write_public_signals(witnesses: list[int], num_public: int) -> list[str]:
    """Extract public signals as string list (z[1:l+1]).

    Args:
        witnesses: Full witness vector (z[0] is the constant 1).
        num_public: Number of public inputs (l).

    Returns:
        List of public signal values as decimal strings.
    """
    return [str(witnesses[i]) for i in range(1, num_public + 1)]
