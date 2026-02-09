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

"""NTT (Number Theoretic Transform) implementation in JAX.

This module provides a vectorized NTT implementation using:
- Cooley-Tukey butterfly for forward transform (DIT)
- Gentleman-Sande butterfly for inverse transform (DIF)

The algorithm operates in O(n log n) time with log₂(n) stages.
Each stage is fully vectorized using reshape-based butterfly operations,
producing KernelThunk(bit_reverse) + unrolled butterfly HLO structure.

Montgomery Form Notes:
    Field types from zk_dtypes (e.g., bn254_sf_mont) use Montgomery form
    for efficient modular multiplication. The NTT operates directly in Montgomery
    form - the transform is valid because Montgomery form is a ring isomorphism:
        Mont(a * b) = Mont(a) * Mont(b) (mod p)
        Mont(a + b) = Mont(a) + Mont(b) (mod p)
"""

from __future__ import annotations

import math
from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
from jax import jit, lax, vmap
from jax.tree_util import register_pytree_node_class
from zk_dtypes import pfinfo

if TYPE_CHECKING:
    from jax import Array


@partial(jit, static_argnums=(1,))
def _forward_ntt(coeffs: Array, log_n: int, *stage_twiddles: Array) -> Array:
    """JIT-compiled forward NTT kernel (Cooley-Tukey DIT).

    Fully vectorized: each butterfly stage processes all groups in parallel
    using reshape + slice + elementwise arithmetic + concatenate.

    Args:
        coeffs: Input coefficients array of size n = 2^log_n.
        log_n: log₂(n), static.
        *stage_twiddles: Per-stage twiddle arrays passed as varargs.
            stage_twiddles[s] has size 2^s.
    """
    n = coeffs.shape[0]

    # DIT: bit-reverse input using native HLO BitReverse op
    data = lax.bit_reverse(coeffs, dimensions=[0])

    # Unrolled vectorized butterfly stages (log_n is static)
    for s in range(log_n):
        half_m = 1 << s
        block_size = 2 * half_m
        num_groups = n // block_size
        omega = stage_twiddles[s]

        # Reshape to (num_groups, block_size), split into upper/lower halves
        blocks = data.reshape(num_groups, block_size)
        upper = lax.slice(blocks, [0, 0], [num_groups, half_m])
        lower = lax.slice(blocks, [0, half_m], [num_groups, block_size])

        # Cooley-Tukey butterfly: A' = A + B*ω, B' = A - B*ω
        # omega broadcasts from (half_m,) over the num_groups dimension
        b_omega = lower * omega
        new_upper = upper + b_omega
        new_lower = upper - b_omega

        data = jnp.concatenate([new_upper, new_lower], axis=1).reshape(n)

    return data


@partial(jit, static_argnums=(2,))
def _inverse_ntt(
    evaluations: Array, inv_n: Array, log_n: int, *stage_twiddles: Array
) -> Array:
    """JIT-compiled inverse NTT kernel (Gentleman-Sande DIF).

    Fully vectorized butterfly stages.

    Args:
        evaluations: NTT evaluations array of size n = 2^log_n.
        inv_n: Inverse of n for final scaling.
        log_n: log₂(n), static.
        *stage_twiddles: Per-stage inverse twiddle arrays passed as varargs.
            stage_twiddles[s] has size 2^(log_n - 1 - s).
    """
    n = evaluations.shape[0]
    data = evaluations

    # Unrolled vectorized butterfly stages (log_n is static)
    for s in range(log_n):
        actual_stage = log_n - 1 - s
        half_m = 1 << actual_stage
        block_size = 2 * half_m
        num_groups = n // block_size
        omega = stage_twiddles[s]

        blocks = data.reshape(num_groups, block_size)
        upper = lax.slice(blocks, [0, 0], [num_groups, half_m])
        lower = lax.slice(blocks, [0, half_m], [num_groups, block_size])

        # Gentleman-Sande butterfly: A' = A + B, B' = (A - B)*ω
        new_upper = upper + lower
        new_lower = (upper - lower) * omega

        data = jnp.concatenate([new_upper, new_lower], axis=1).reshape(n)

    data = data * inv_n

    # DIF: bit-reverse output using native HLO BitReverse op
    data = lax.bit_reverse(data, dimensions=[0])
    return data


# Module-level twiddle cache keyed by dtype
_twiddle_cache: dict[type, tuple[list, list, list]] = {}


@register_pytree_node_class
class NTT:
    """NTT (Number Theoretic Transform) for prime fields.

    Args:
        dtype: The field element type (e.g., bn254_sf_mont).
        root_of_unity: A primitive 2^k-th root of unity for the field.
    """

    def __init__(self, dtype: type, root_of_unity: int):
        self.DTYPE = dtype
        self.ROOT_OF_UNITY = root_of_unity
        pf = pfinfo(dtype)
        self.MODULUS = pf.modulus
        self.MAX_LOG_N = pf.two_adicity

    def _compute_twiddles(self) -> tuple[list, list, list]:
        """Compute twiddle factors for all supported NTT sizes.

        For each size n = 2^k (k = 1, ..., practical_max), compute:
        - Forward twiddles: [omega⁰, omega¹, ..., omega^(n - 1)]
          where omega is an n-th root of unity
        - Inverse twiddles: [omega⁻⁰, omega⁻¹, ..., omega^(-(n - 1))]
        - Inverse degree: 1 / n for scaling the inverse transform

        Uses vectorized doubling: at each step, concatenate the existing
        array with itself multiplied by a stepping power of omega. This
        replaces O(n) Python big-integer iterations with O(log n) JAX
        field-element operations.

        Returns:
            Tuple of (twiddles, inv_twiddles, inv_degrees).
        """
        # Limit precomputation to 2²⁰ for memory efficiency
        practical_max_log_n = min(self.MAX_LOG_N, 20)
        one = self.DTYPE(1)

        with jax.ensure_compile_time_eval():
            all_twiddles = []
            all_inv_twiddles = []
            all_inv_degrees = []

            for log_n in range(1, practical_max_log_n + 1):
                n = 1 << log_n

                # Compute n-th root of unity via repeated squaring:
                # omega_n = root^(2^(MAX_LOG_N - log_n))
                omega = self.DTYPE(self.ROOT_OF_UNITY)
                for _ in range(self.MAX_LOG_N - log_n):
                    omega = omega * omega
                omega_inv = one / omega

                # Build twiddle arrays via doubling:
                #   Step 0: [1]
                #   Step 1: [1, omega^(n/2)]
                #   Step 2: [1, omega^(n/4), omega^(n/2), omega^(3n/4)]
                #   ...
                #   Step k: [omega⁰, omega¹, ..., omega^(n - 1)]
                twiddles = jnp.array([one])
                inv_twiddles = jnp.array([one])
                step_mul = omega
                step_mul_inv = omega_inv
                for _ in range(log_n):
                    twiddles = jnp.concatenate([twiddles, twiddles * step_mul])
                    inv_twiddles = jnp.concatenate(
                        [inv_twiddles, inv_twiddles * step_mul_inv]
                    )
                    step_mul = step_mul * step_mul
                    step_mul_inv = step_mul_inv * step_mul_inv

                all_twiddles.append(twiddles)
                all_inv_twiddles.append(inv_twiddles)
                all_inv_degrees.append(one / self.DTYPE(n))

        return all_twiddles, all_inv_twiddles, all_inv_degrees

    def _get_twiddles(self) -> tuple[list, list, list]:
        """Get or compute cached twiddle factors."""
        if self.DTYPE not in _twiddle_cache:
            _twiddle_cache[self.DTYPE] = self._compute_twiddles()
        return _twiddle_cache[self.DTYPE]

    def forward(self, coeffs: Array) -> Array:
        """Compute forward NTT (Cooley-Tukey decimation-in-time).

        Fully vectorized: each stage processes all butterfly groups in parallel
        using reshape-based slicing and elementwise arithmetic.

        Target HLO: KernelThunk(bit_reverse) -> unrolled butterfly stages

        Butterfly operation (Cooley-Tukey):
            A' = A + B * omega
            B' = A - B * omega

        Args:
            coeffs: Input coefficients in standard order.

        Returns:
            NTT evaluations.
        """
        n = coeffs.shape[0]
        log_n = int(math.log2(n))
        all_twiddles, _, _ = self._get_twiddles()
        roots = all_twiddles[log_n - 1]

        # Extract per-stage twiddles: stage s needs roots[::stride] with
        # stride = n // 2^(s+1), giving 2^s elements.
        # Use numpy stride slicing then convert through Python ints to avoid
        # both gather (broken for ZK types) and batched_device_put (fails for
        # 256-bit types like bn254_sf_mont).
        roots_np = np.array(roots)
        stage_twiddles = []
        for s in range(log_n):
            half_m = 1 << s
            stride = n // (2 * half_m)
            tw_ints = [int(x) for x in roots_np[::stride][:half_m]]
            tw = jnp.array(tw_ints, dtype=coeffs.dtype)
            stage_twiddles.append(tw)
        return _forward_ntt(coeffs, log_n, *stage_twiddles)

    def inverse(self, evaluations: Array) -> Array:
        """Compute inverse NTT (Gentleman-Sande decimation-in-frequency).

        Fully vectorized butterfly stages.

        Butterfly operation (Gentleman-Sande):
            A' = A + B
            B' = (A - B) * omega

        Args:
            evaluations: NTT evaluations.

        Returns:
            Polynomial coefficients in standard order.
        """
        n = evaluations.shape[0]
        log_n = int(math.log2(n))
        _, all_inv_twiddles, all_inv_degrees = self._get_twiddles()
        inv_roots = all_inv_twiddles[log_n - 1]
        inv_n = all_inv_degrees[log_n - 1]

        # Extract per-stage inverse twiddles.
        # Use numpy stride slicing then convert through Python ints (same
        # workaround as forward — see comment there).
        inv_roots_np = np.array(inv_roots)
        stage_twiddles = []
        for s in range(log_n):
            actual_stage = log_n - 1 - s
            half_m = 1 << actual_stage
            stride = n // (2 * half_m)
            tw_ints = [int(x) for x in inv_roots_np[::stride][:half_m]]
            tw = jnp.array(tw_ints, dtype=evaluations.dtype)
            stage_twiddles.append(tw)
        return _inverse_ntt(evaluations, inv_n, log_n, *stage_twiddles)

    @staticmethod
    def bit_reverse(data: Array) -> Array:
        """Apply bit-reversal permutation along the first dimension.

        For an array of size n (must be a power of 2), the element at index i
        is moved to index bit_reverse(i, log₂(n)).

        Args:
            data: Input array whose first dimension size is a power of 2.

        Returns:
            Bit-reversed array.
        """
        return lax.bit_reverse(data, dimensions=[0])

    def ntt(self, coeffs: Array, inverse: bool = False) -> Array:
        """Unified NTT interface.

        Args:
            coeffs: Input array (coefficients for forward, evaluations for inverse).
            inverse: If True, compute inverse NTT.

        Returns:
            Transformed array.
        """
        if inverse:
            return self.inverse(coeffs)
        return self.forward(coeffs)

    def tree_flatten(self):
        """Flatten for JAX pytree."""
        return (), (self.DTYPE, self.ROOT_OF_UNITY)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        """Unflatten from JAX pytree."""
        dtype, root_of_unity = aux_data
        return cls(dtype, root_of_unity)


@jit
def batch_ntt(ntt_instance: NTT, batch: Array, inverse: bool = False) -> Array:
    """Apply NTT to a batch of polynomials.

    Applies NTT (or inverse NTT) to each row of the input array.

    Args:
        ntt_instance: The NTT implementation to use.
        batch: 2D array of shape (batch_size, poly_degree).
        inverse: If True, compute inverse NTT.

    Returns:
        Transformed batch with same shape.
    """
    ntt_fn = ntt_instance.inverse if inverse else ntt_instance.forward
    return vmap(ntt_fn)(batch)
