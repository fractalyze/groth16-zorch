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

"""MSM implementation for BN254 curve.

BN254 (also known as alt_bn128, bn128) is a pairing-friendly elliptic curve
defined by:
    y² = x³ + 3 (mod p)

Where:
    - Base field (Fq)
    - Scalar field (Fr)
    - Curve parameter: a = 0, b = 3

This curve is used in Ethereum's precompiles and many ZK-SNARK systems.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from jax.tree_util import register_pytree_node_class
from zk_dtypes import bn254_g1_affine, bn254_g2_affine, bn254_sf

from .msm import MSM
from .pippenger import pippenger_msm

if TYPE_CHECKING:
    from jax import Array


# BN254 constants
BN254_SCALAR_BITS = 254


@register_pytree_node_class
class MSMBn254(MSM):
    """MSM implementation for BN254 G1 curve.

    Uses Pippenger's bucket method with EC point dtypes.  The ``+`` operator
    on EC point arrays automatically lowers to ``ec.add`` / ``ec.double``.

    Example:
        >>> msm = MSMBn254()
        >>> scalars = jnp.array([1, 2, 3], dtype=bn254_sf)
        >>> points = jnp.array([...], dtype=bn254_g1_affine)
        >>> result = msm.compute(scalars, points)
    """

    SCALAR_BITS = BN254_SCALAR_BITS
    SCALAR_DTYPE = bn254_sf
    POINT_DTYPE = bn254_g1_affine

    def compute(
        self,
        scalars: Array,
        points: Array,
        *,
        window_bits: int | None = None,
        num_chunks: int = 1,
    ) -> Array:
        """Compute MSM: sum_{i} scalars[i] * points[i].

        Args:
            scalars: Array of BN254 scalar field elements (shape: ``[n]``).
            points: Array of BN254 G1 points (shape: ``[n]``, EC point dtype).
            window_bits: Window size for Pippenger's algorithm.
                         If None, automatically estimated.
            num_chunks: Parallel chunks for ``vmap`` (default 1).

        Returns:
            Single EC point (XYZZ representation).
        """
        return pippenger_msm(
            scalars,
            points,
            scalar_bits=self.SCALAR_BITS,
            window_bits=window_bits,
            num_chunks=num_chunks,
        )


@register_pytree_node_class
class MSMBn254G2(MSM):
    """MSM implementation for BN254 G2 curve.

    Uses Pippenger's bucket method with EC point dtypes.  The ``+`` operator
    on EC point arrays automatically lowers to ``ec.add`` / ``ec.double``.

    Example:
        >>> msm = MSMBn254G2()
        >>> scalars = jnp.array([1, 2, 3], dtype=bn254_sf)
        >>> points = jnp.array([...], dtype=bn254_g2_affine)
        >>> result = msm.compute(scalars, points)
    """

    SCALAR_BITS = BN254_SCALAR_BITS
    SCALAR_DTYPE = bn254_sf
    POINT_DTYPE = bn254_g2_affine

    def compute(
        self,
        scalars: Array,
        points: Array,
        *,
        window_bits: int | None = None,
        num_chunks: int = 1,
    ) -> Array:
        """Compute MSM: sum_{i} scalars[i] * points[i].

        Args:
            scalars: Array of BN254 scalar field elements (shape: ``[n]``).
            points: Array of BN254 G2 points (shape: ``[n]``, EC point dtype).
            window_bits: Window size for Pippenger's algorithm.
                         If None, automatically estimated.
            num_chunks: Parallel chunks for ``vmap`` (default 1).

        Returns:
            Single EC point (XYZZ representation).
        """
        return pippenger_msm(
            scalars,
            points,
            scalar_bits=self.SCALAR_BITS,
            window_bits=window_bits,
            num_chunks=num_chunks,
        )
