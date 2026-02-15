# Copyright 2026 The RabbitSNARK Authors.
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

# BN254 base field modulus (for XYZZ→affine modular inversion).
_BN254_BF_P = (
    21888242871839275222246405745257275088696311157297823662689037894645226208583
)


@dataclass
class Groth16Proof:
    """Groth16 proof containing three elliptic curve points.

    Attributes:
        pi_a: G1 point (XYZZ representation).
        pi_b: G2 point (XYZZ representation).
        pi_c: G1 point (XYZZ representation).
    """

    pi_a: Array
    pi_b: Array
    pi_c: Array

    def to_json(self) -> dict:
        """Serialize to snarkjs-compatible JSON format.

        Converts XYZZ points to affine coordinates and formats as:
        ``{"pi_a": [x, y, "1"], "pi_b": [[x0,x1],[y0,y1],["1","0"]],
           "pi_c": [x, y, "1"], "protocol": "groth16", "curve": "bn128"}``
        """
        pi_a_x, pi_a_y = _xyzz_g1_to_affine(self.pi_a)
        pi_b_xy = _xyzz_g2_to_affine(self.pi_b)
        pi_c_x, pi_c_y = _xyzz_g1_to_affine(self.pi_c)

        return {
            "pi_a": [str(pi_a_x), str(pi_a_y), "1"],
            "pi_b": [
                [str(pi_b_xy[0][0]), str(pi_b_xy[0][1])],
                [str(pi_b_xy[1][0]), str(pi_b_xy[1][1])],
                ["1", "0"],
            ],
            "pi_c": [str(pi_c_x), str(pi_c_y), "1"],
            "protocol": "groth16",
            "curve": "bn128",
        }


def _xyzz_g1_to_affine(point: Array) -> tuple[int, int]:
    """Convert a G1 XYZZ point to affine (x, y).

    XYZZ has 4 coordinates (X, Y, ZZ, ZZZ) where x = X / ZZ, y = Y / ZZZ.
    """
    raw = np.array(point).item().raw
    x, y, zz, zzz = (int(v) for v in raw)
    if zz == 0:
        return (0, 0)
    return (
        x * pow(zz, -1, _BN254_BF_P) % _BN254_BF_P,
        y * pow(zzz, -1, _BN254_BF_P) % _BN254_BF_P,
    )


def _xyzz_g2_to_affine(
    point: Array,
) -> tuple[tuple[int, int], tuple[int, int]]:
    """Convert a G2 XYZZ point to affine coordinates.

    G2 XYZZ has 4 coordinates in Fq² (each is a 2-element tuple).
    Returns ((x0, x1), (y0, y1)).
    """
    raw = np.array(point).item().raw
    x_fq2, y_fq2, zz_fq2, zzz_fq2 = raw
    # Each coordinate is an Fq² element (tuple of 2 ints)
    x0, x1 = int(x_fq2[0]), int(x_fq2[1])
    y0, y1 = int(y_fq2[0]), int(y_fq2[1])
    zz0, zz1 = int(zz_fq2[0]), int(zz_fq2[1])
    zzz0, zzz1 = int(zzz_fq2[0]), int(zzz_fq2[1])

    # Fq² inversion: (a + b*u)⁻¹ = (a - b*u) / (a² + b²)
    # For ZZ:
    p = _BN254_BF_P
    zz_norm = (zz0 * zz0 + zz1 * zz1) % p
    zz_inv_norm = pow(zz_norm, -1, p)
    zz_inv = (zz0 * zz_inv_norm % p, (-zz1 * zz_inv_norm) % p)

    zzz_norm = (zzz0 * zzz0 + zzz1 * zzz1) % p
    zzz_inv_norm = pow(zzz_norm, -1, p)
    zzz_inv = (zzz0 * zzz_inv_norm % p, (-zzz1 * zzz_inv_norm) % p)

    # Fq² multiplication: (a + b*u)(c + d*u) = (ac - bd) + (ad + bc)*u
    ax = (x0 * zz_inv[0] - x1 * zz_inv[1]) % p
    ax1 = (x0 * zz_inv[1] + x1 * zz_inv[0]) % p
    ay = (y0 * zzz_inv[0] - y1 * zzz_inv[1]) % p
    ay1 = (y0 * zzz_inv[1] + y1 * zzz_inv[0]) % p

    return ((ax, ax1), (ay, ay1))


def write_public_signals(witnesses: list[int], num_public: int) -> list[str]:
    """Extract public signals as string list (z[1:l+1]).

    Args:
        witnesses: Full witness vector (z[0] is the constant 1).
        num_public: Number of public inputs (l).

    Returns:
        List of public signal values as decimal strings.
    """
    return [str(witnesses[i]) for i in range(1, num_public + 1)]
