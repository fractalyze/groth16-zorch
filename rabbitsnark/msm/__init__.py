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

"""MSM (Multi-Scalar Multiplication) implementations in JAX.

This module provides efficient MSM implementations using Pippenger's algorithm
(bucket method) for elliptic curves commonly used in zero-knowledge proofs.

EC point dtypes (e.g. ``bn254_g1_affine``, ``bn254_g1_xyzz``) are atomic
JAX dtypes.  The ``+`` operator on these types automatically lowers to
``ec.add`` / ``ec.double``, so the algorithm contains no manual point
arithmetic.

Example usage:
    >>> from rabbitsnark.msm import MSMBn254
    >>> import jax.numpy as jnp
    >>> from zk_dtypes import bn254_sf, bn254_g1_affine
    >>>
    >>> msm = MSMBn254()
    >>> scalars = jnp.array([...], dtype=bn254_sf)
    >>> points = jnp.array([...], dtype=bn254_g1_affine)
    >>> result = msm.compute(scalars, points)

References:
    - Pippenger's Algorithm: https://encrypt.a41.io/primitives/abstract-algebra/elliptic-curve/msm/pippengers-algorithm
    - Explicit Formulas Database: https://www.hyperelliptic.org/EFD/
"""

from .msm import MSM
from .msm_bn254 import MSMBn254, MSMBn254G2
from .pippenger import pippenger_msm

__all__ = [
    # Base class
    "MSM",
    # Curve-specific implementations
    "MSMBn254",
    "MSMBn254G2",
    # Algorithm
    "pippenger_msm",
]
