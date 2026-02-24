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
Each stage is traced as a separate HLO fusion via an unrolled Python
``for`` loop, allowing the compiler to fuse and parallelize across stages.
Each stage uses reshape/slice/concatenate for the butterfly pattern,
avoiding scatter operations (which crash with ZK field types).

Bit-reversal is performed via ``lax.bit_reverse`` (native HLO BitReverse op).

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
from jax import jit, lax
from jax.tree_util import register_pytree_node_class
from zk_dtypes import pfinfo

if TYPE_CHECKING:
    from jax import Array


@partial(jit, static_argnums=(1,))
def _forward_ntt(coeffs: Array, log_n: int, *stage_twiddles: Array) -> Array:
    """JIT-compiled forward NTT kernel (Cooley-Tukey DIT).

    Each butterfly stage is traced separately via an unrolled Python ``for``
    loop (no WhileOp), allowing inter-stage fusion and parallelization.
    Each stage uses reshape/slice/concatenate for the butterfly pattern.

    Args:
        coeffs: Input coefficients array of size n = 2^log_n.
        log_n: log₂(n), static.
        *stage_twiddles: Per-stage twiddle arrays. stage_twiddles[s] has
            shape (2ˢ,) containing the roots for stage s.
    """
    n = coeffs.shape[0]

    # DIT: bit-reverse input (keep native HLO BitReverse)
    data = lax.bit_reverse(coeffs, dimensions=[0])
    data = data[:, None]  # (n,) → (n, 1) for reshape butterfly

    # Unrolled Cooley-Tukey DIT stages (Python for loop → no WhileOp)
    for s in range(log_n):
        block_size = 1 << (s + 1)
        half_block = block_size // 2
        num_blocks = n // block_size
        tw = stage_twiddles[s][None, :, None]  # (1, half_block, 1)
        blocks = data.reshape(num_blocks, block_size, 1)
        top = blocks[:, :half_block, :]
        bot = blocks[:, half_block:, :]
        bot_tw = bot * tw
        data = jnp.concatenate([top + bot_tw, top - bot_tw], axis=1).reshape(n, 1)

    return data[:, 0]  # (n, 1) → (n,)


@partial(jit, static_argnums=(2,))
def _inverse_ntt(
    evaluations: Array, inv_n: Array, log_n: int, *stage_twiddles: Array
) -> Array:
    """JIT-compiled inverse NTT kernel (Gentleman-Sande DIF).

    Each butterfly stage is traced separately via an unrolled Python ``for``
    loop (no WhileOp). Stages run from large blocks to small (DIF order).
    Each stage uses reshape/slice/concatenate for the butterfly pattern.

    Args:
        evaluations: NTT evaluations array of size n = 2^log_n.
        inv_n: Inverse of n for final scaling.
        log_n: log₂(n), static.
        *stage_twiddles: Per-stage inverse twiddle arrays. stage_twiddles[i]
            corresponds to DIF stage i (large blocks first).
    """
    n = evaluations.shape[0]
    data = evaluations[:, None]  # (n,) → (n, 1)

    # Unrolled Gentleman-Sande DIF stages (large blocks → small)
    for stage_idx in range(log_n):
        s = log_n - 1 - stage_idx
        block_size = 1 << (s + 1)
        half_block = block_size // 2
        num_blocks = n // block_size
        tw = stage_twiddles[stage_idx][None, :, None]
        blocks = data.reshape(num_blocks, block_size, 1)
        top = blocks[:, :half_block, :]
        bot = blocks[:, half_block:, :]
        data = jnp.concatenate([top + bot, (top - bot) * tw], axis=1).reshape(n, 1)

    # DIF: bit-reverse output (keep native HLO BitReverse)
    data = lax.bit_reverse(data[:, 0], dimensions=[0])
    return data * inv_n


# Module-level twiddle cache keyed by dtype
_twiddle_cache: dict[type, tuple[list, list, list, list, list]] = {}


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

    def _compute_twiddles(self) -> tuple[list, list, list, list, list]:
        """Compute twiddle factors for all supported NTT sizes.

        For each size n = 2^k (k = 1, ..., practical_max), compute:
        - Forward twiddles: [ω⁰, ω¹, ..., ω^(n - 1)]
          where ω is an n-th root of unity
        - Inverse twiddles: [ω⁻⁰, ω⁻¹, ..., ω^(-(n - 1))]
        - Inverse degree: 1 / n for scaling the inverse transform
        - Forward per-stage twiddles: tuple of per-stage arrays for DIT
        - Inverse per-stage twiddles: tuple of per-stage arrays for DIF

        Per-stage twiddle arrays are extracted via static strided slicing
        (Slice HLO, not gather) for the unrolled NTT kernels.

        Uses Python big-integer arithmetic for twiddle generation to avoid
        a ZKX concatenation bug (segfault at >= 2¹⁴ elements from repeated
        concat). Only the final arrays are converted to JAX field elements.

        Returns:
            Tuple of (twiddles, inv_twiddles, inv_degrees,
            fwd_stage_twiddles, inv_stage_twiddles).
        """
        # Limit precomputation to 2²⁰ for memory efficiency
        practical_max_log_n = min(self.MAX_LOG_N, 20)
        p = self.MODULUS

        # Compute the 2^practical_max_log_n-th root from the 2^MAX_LOG_N-th root,
        # then derive smaller roots by squaring.  omega_ints[k] = primitive 2^k-th root.
        max_omega = pow(
            self.ROOT_OF_UNITY, 1 << (self.MAX_LOG_N - practical_max_log_n), p
        )
        omega_ints: list[int] = [0] * (practical_max_log_n + 1)
        omega_ints[practical_max_log_n] = max_omega
        for k in range(practical_max_log_n - 1, 0, -1):
            omega_ints[k] = pow(omega_ints[k + 1], 2, p)

        with jax.ensure_compile_time_eval():
            all_twiddles = []
            all_inv_twiddles = []
            all_inv_degrees = []
            all_fwd_stages = []
            all_inv_stages = []

            for log_n in range(1, practical_max_log_n + 1):
                n = 1 << log_n
                omega = omega_ints[log_n]
                omega_inv = pow(omega, p - 2, p)

                # Build twiddle arrays via Python big-int arithmetic:
                #   tw[i] = omega^i mod p
                tw = [0] * n
                tw_inv = [0] * n
                tw[0] = 1
                tw_inv[0] = 1
                for i in range(1, n):
                    tw[i] = tw[i - 1] * omega % p
                    tw_inv[i] = tw_inv[i - 1] * omega_inv % p

                twiddles = jnp.array(tw, dtype=self.DTYPE)
                inv_twiddles = jnp.array(tw_inv, dtype=self.DTYPE)
                inv_deg = jnp.array([pow(n, p - 2, p)], dtype=self.DTYPE)[0]

                all_twiddles.append(twiddles)
                all_inv_twiddles.append(inv_twiddles)
                all_inv_degrees.append(inv_deg)

                # Forward DIT: stage s needs roots[::stride][:half_m]
                fwd_stages = []
                for s in range(log_n):
                    half_m = 1 << s
                    stride = n // (2 * half_m)
                    fwd_stages.append(twiddles[::stride][:half_m])
                all_fwd_stages.append(tuple(fwd_stages))

                # Inverse DIF: stages run in reverse order
                inv_stages = []
                for stage_idx in range(log_n):
                    actual_stage = log_n - 1 - stage_idx
                    half_m = 1 << actual_stage
                    stride = n // (2 * half_m)
                    inv_stages.append(inv_twiddles[::stride][:half_m])
                all_inv_stages.append(tuple(inv_stages))

        return (
            all_twiddles,
            all_inv_twiddles,
            all_inv_degrees,
            all_fwd_stages,
            all_inv_stages,
        )

    def _get_twiddles(self) -> tuple[list, list, list, list, list]:
        """Get or compute cached twiddle factors."""
        if self.DTYPE not in _twiddle_cache:
            _twiddle_cache[self.DTYPE] = self._compute_twiddles()
        return _twiddle_cache[self.DTYPE]

    def get_twiddle_arrays(self, n: int) -> tuple[Array, Array, Array]:
        """Return cached (fwd_roots, inv_roots, inv_n) for size n.

        These are the full root-of-unity arrays used by external callers
        (e.g., groth16 prover) that extract per-stage twiddles themselves.

        Args:
            n: NTT size (must be a power of 2).

        Returns:
            Tuple of (forward_roots, inverse_roots, inv_n).
        """
        log_n = int(math.log2(n))
        all_tw, all_inv_tw, all_inv_deg, _, _ = self._get_twiddles()
        return all_tw[log_n - 1], all_inv_tw[log_n - 1], all_inv_deg[log_n - 1]

    def _get_fwd_stage_twiddles(self, log_n: int) -> tuple[Array, ...]:
        """Return cached per-stage forward twiddles for size 2^log_n."""
        _, _, _, all_fwd_stages, _ = self._get_twiddles()
        return all_fwd_stages[log_n - 1]

    def _get_inv_stage_twiddles(self, log_n: int) -> tuple[Array, ...]:
        """Return cached per-stage inverse twiddles for size 2^log_n."""
        _, _, _, _, all_inv_stages = self._get_twiddles()
        return all_inv_stages[log_n - 1]

    def forward(self, coeffs: Array) -> Array:
        """Compute forward NTT (Cooley-Tukey decimation-in-time).

        Each butterfly stage is traced as a separate HLO fusion via an
        unrolled Python ``for`` loop.

        Butterfly operation (Cooley-Tukey):
            A' = A + B * ω
            B' = A - B * ω

        Args:
            coeffs: Input coefficients in standard order.

        Returns:
            NTT evaluations.
        """
        log_n = int(math.log2(coeffs.shape[0]))
        fwd_stages = self._get_fwd_stage_twiddles(log_n)
        return _forward_ntt(coeffs, log_n, *fwd_stages)

    def inverse(self, evaluations: Array) -> Array:
        """Compute inverse NTT (Gentleman-Sande decimation-in-frequency).

        Each butterfly stage is traced as a separate HLO fusion via an
        unrolled Python ``for`` loop.

        Butterfly operation (Gentleman-Sande):
            A' = A + B
            B' = (A - B) * ω

        Args:
            evaluations: NTT evaluations.

        Returns:
            Polynomial coefficients in standard order.
        """
        log_n = int(math.log2(evaluations.shape[0]))
        _, _, all_inv_deg, _, _ = self._get_twiddles()
        inv_n = all_inv_deg[log_n - 1]
        inv_stages = self._get_inv_stage_twiddles(log_n)
        return _inverse_ntt(evaluations, inv_n, log_n, *inv_stages)

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


def batch_ntt(ntt_instance: NTT, batch: Array, inverse: bool = False) -> Array:
    """Apply NTT to a batch of polynomials.

    Applies NTT (or inverse NTT) to each row of the input array.
    Each row is processed through the JIT-compiled NTT kernel.

    Args:
        ntt_instance: The NTT implementation to use.
        batch: 2D array of shape (batch_size, poly_degree).
        inverse: If True, compute inverse NTT.

    Returns:
        Transformed batch with same shape.
    """
    ntt_fn = ntt_instance.inverse if inverse else ntt_instance.forward
    return jnp.stack([ntt_fn(batch[i]) for i in range(batch.shape[0])])
