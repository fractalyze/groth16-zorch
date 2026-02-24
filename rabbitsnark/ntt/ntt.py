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

from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
from jax import jit, lax
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
_twiddle_cache: dict[type, tuple[list, list, list]] = {}


def _build_twiddle_array(one: Array, omega: Array, log_size: int) -> Array:
    """Build [1, ω, ω², ..., ω^(2^log_size - 1)] via O(log_size) doublings.

    Uses Montgomery multiplication through the ``*`` operator on field elements.
    Only O(log_size) concatenations (safe for ZKX concat).
    """
    arr = jnp.array([one], dtype=one.dtype)
    step = omega
    for _ in range(log_size):
        arr = jnp.concatenate([arr, arr * step])
        step = step * step
    return arr


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
        """Compute per-stage twiddle factors for all supported NTT sizes.

        For each size n = 2^k (k = 1, ..., practical_max), compute:
        - Forward per-stage twiddles: tuple of per-stage arrays for DIT
        - Inverse per-stage twiddles: tuple of per-stage arrays for DIF
        - Inverse degree: 1 / n for scaling the inverse transform

        Uses zk_dtypes Montgomery multiplication (``*`` operator on field
        elements) and O(log n) doubling for twiddle array construction.

        Returns:
            Tuple of (inv_degrees, fwd_stage_twiddles, inv_stage_twiddles).
        """
        # Limit precomputation to 2²⁰ for memory efficiency
        practical_max_log_n = min(self.MAX_LOG_N, 20)
        p = self.MODULUS
        dtype = self.DTYPE

        # Compute omega_ints[k] = primitive 2^k-th root (Python int).
        # Start from 2^MAX_LOG_N-th root, then derive smaller roots by squaring.
        max_omega = pow(
            self.ROOT_OF_UNITY, 1 << (self.MAX_LOG_N - practical_max_log_n), p
        )
        omega_ints: list[int] = [0] * (practical_max_log_n + 1)
        omega_ints[practical_max_log_n] = max_omega
        for k in range(practical_max_log_n - 1, 0, -1):
            omega_ints[k] = pow(omega_ints[k + 1], 2, p)

        with jax.ensure_compile_time_eval():
            one = dtype(1)
            all_inv_degrees = []
            all_fwd_stages = []
            all_inv_stages = []

            for log_n in range(1, practical_max_log_n + 1):
                all_inv_degrees.append(one / dtype(1 << log_n))

                # Forward DIT: stage s uses 2^(s + 1)-th root of unity
                fwd_stages = []
                for s in range(log_n):
                    omega_s = dtype(omega_ints[s + 1])
                    fwd_stages.append(_build_twiddle_array(one, omega_s, s))
                all_fwd_stages.append(tuple(fwd_stages))

                # Inverse DIF: stages in reverse order, using inverse roots
                inv_stages = []
                for stage_idx in range(log_n):
                    actual_stage = log_n - 1 - stage_idx
                    omega_s_inv = one / dtype(omega_ints[actual_stage + 1])
                    inv_stages.append(
                        _build_twiddle_array(one, omega_s_inv, actual_stage)
                    )
                all_inv_stages.append(tuple(inv_stages))

        return all_inv_degrees, all_fwd_stages, all_inv_stages

    def _get_twiddles(self) -> tuple[list, list, list]:
        """Get or compute cached twiddle factors."""
        if self.DTYPE not in _twiddle_cache:
            _twiddle_cache[self.DTYPE] = self._compute_twiddles()
        return _twiddle_cache[self.DTYPE]

    def get_stage_twiddles(
        self, log_n: int
    ) -> tuple[tuple[Array, ...], tuple[Array, ...], Array]:
        """Return cached (fwd_stages, inv_stages, inv_n) for size 2^log_n.

        Args:
            log_n: log₂ of the NTT size.

        Returns:
            Tuple of (fwd_stage_twiddles, inv_stage_twiddles, inv_n).
        """
        all_inv_deg, all_fwd, all_inv = self._get_twiddles()
        return all_fwd[log_n - 1], all_inv[log_n - 1], all_inv_deg[log_n - 1]
