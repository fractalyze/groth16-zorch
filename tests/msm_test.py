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

"""Tests for MSM (Multi-Scalar Multiplication) implementations."""

import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from zk_dtypes import bn254_g1_affine, bn254_sf

from rabbitsnark.msm import MSMBn254

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


def _naive_msm(scalars, points, msm):
    """Compute MSM naively: sum of individual s_i * P_i."""
    result = msm.compute(scalars[0:1], points[0:1])
    for i in range(1, scalars.shape[0]):
        result = result + msm.compute(scalars[i : i + 1], points[i : i + 1])
    return result


class TestMSMCorrectness(absltest.TestCase):
    """Correctness tests: Pippenger MSM vs naive MSM."""

    def setUp(self):
        self.msm = MSMBn254()
        # BN254 G1 generator: (1, 2)
        self.generator = jnp.array([bn254_g1_affine((1, 2))], dtype=bn254_g1_affine)

    def test_multi_point_msm(self):
        """Pippenger MSM matches naive MSM for small inputs."""
        g = bn254_g1_affine((1, 2))
        points = jnp.array([g, g, g], dtype=bn254_g1_affine)
        scalars = jnp.array([2, 3, 4], dtype=bn254_sf)

        pippenger_result = self.msm.compute(scalars, points)
        naive_result = _naive_msm(scalars, points, self.msm)

        # XYZZ points use projective coordinates — compare in affine form.
        self.assertEqual(
            _xyzz_to_affine(pippenger_result), _xyzz_to_affine(naive_result)
        )


if __name__ == "__main__":
    absltest.main()
