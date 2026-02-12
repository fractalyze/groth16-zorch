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

"""Base MSM (Multi-Scalar Multiplication) class.

MSM computes the sum: sum_{i=0}^{n-1} s_i * P_i

where s_i are scalars and P_i are elliptic curve points.

This is a fundamental operation in zero-knowledge proofs, used in:
- Groth16 proving
- PLONK polynomial commitments
- KZG commitments
- Bulletproofs inner product arguments
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

from jax.tree_util import register_pytree_node_class

from .pippenger import _estimate_optimal_window_bits

if TYPE_CHECKING:
    from jax import Array


@register_pytree_node_class
class MSM(ABC):
    """Abstract base class for MSM implementations.

    Subclasses must define:
        - SCALAR_BITS: Number of bits in scalar field
        - SCALAR_DTYPE: Dtype for scalar field elements
        - POINT_DTYPE: Dtype for curve points
    """

    # Subclasses must override these
    SCALAR_BITS: int
    SCALAR_DTYPE: type
    POINT_DTYPE: type

    @staticmethod
    def estimate_optimal_window_bits(scalar_bits: int, num_points: int) -> int:
        """Estimate optimal window size for Pippenger's algorithm.

        Args:
            scalar_bits: Number of bits in scalars.
            num_points: Number of scalar-point pairs.

        Returns:
            Optimal window size in bits.
        """
        return _estimate_optimal_window_bits(scalar_bits, num_points)

    @abstractmethod
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
            scalars: Array of scalar field elements (shape: ``[n]``).
            points: Array of EC points (shape: ``[n]``, EC point dtype).
            window_bits: Window size for Pippenger's algorithm.
                         If None, automatically estimated.
            num_chunks: Parallel chunks for ``vmap`` (default 1).

        Returns:
            Single EC point (XYZZ representation).
        """

    def tree_flatten(self):
        """Flatten for JAX pytree."""
        return (), ()

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Unflatten from JAX pytree."""
        return cls()
