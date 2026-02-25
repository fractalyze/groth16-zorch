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

"""Tests for MSM (Multi-Scalar Multiplication) utilities."""

import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from zk_dtypes import bn254_g1_affine, bn254_g1_xyzz, bn254_sf

from rabbitsnark.msm import MSM

# BN254 base field modulus (for XYZZ→affine conversion).
BN254_BF_P = (
    21888242871839275222246405745257275088696311157297823662689037894645226208583
)


def _xyzz_to_affine(xyzz_point):
    """Convert XYZZ point to affine (x, y) via modular inversion."""
    raw = np.array(xyzz_point).item().raw
    x, y, zz, zzz = (int(v) for v in raw)
    if zz == 0:
        return (0, 0)  # Point at infinity
    x_aff = x * pow(zz, -1, BN254_BF_P) % BN254_BF_P
    y_aff = y * pow(zzz, -1, BN254_BF_P) % BN254_BF_P
    return (x_aff, y_aff)


class TestScalarMul(absltest.TestCase):
    """Correctness tests for EC scalar multiplication via ``*`` operator."""

    def test_scalar_mul_matches_repeated_add(self):
        """3 * G via ``*`` matches G + G + G."""
        g_xyzz = MSM.affine_to_xyzz(
            jnp.array([bn254_g1_affine((1, 2))], dtype=bn254_g1_affine),
            bn254_g1_xyzz,
        )
        s = jnp.array([bn254_sf(3)], dtype=bn254_sf)

        scalar_mul_result = s * g_xyzz
        add_result = g_xyzz + g_xyzz + g_xyzz

        self.assertEqual(
            _xyzz_to_affine(scalar_mul_result), _xyzz_to_affine(add_result)
        )

    def test_naive_msm_via_mul(self):
        """sum(s_i * P_i) via ``*`` and ``+`` for small input."""
        g = bn254_g1_affine((1, 2))
        points = MSM.affine_to_xyzz(
            jnp.array([g, g, g], dtype=bn254_g1_affine), bn254_g1_xyzz
        )
        scalars = jnp.array([2, 3, 4], dtype=bn254_sf)

        # sum(s_i * P_i) = (2 + 3 + 4) * G = 9 * G
        naive_msm = scalars[0] * points[0]
        for i in range(1, scalars.shape[0]):
            naive_msm = naive_msm + scalars[i] * points[i]

        nine_g = jnp.array([bn254_sf(9)], dtype=bn254_sf) * points[0:1]

        self.assertEqual(_xyzz_to_affine(naive_msm), _xyzz_to_affine(nine_g[0]))


if __name__ == "__main__":
    absltest.main()
