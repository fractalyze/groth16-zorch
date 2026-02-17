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

"""Coset NTT utilities for zero-knowledge proofs.

Coset NTT evaluates a polynomial on a coset g*H where H is a multiplicative
subgroup and g is a generator element (shift). This is commonly used in:
- PLONK-style proofs for evaluating at shifted domains
- FRI protocol for polynomial commitment schemes
- Low-degree testing algorithms

The coset NTT is computed as:
    coset_ntt(f, g) = ntt(f * [g^0, g^1, g^2, ..., g^(n-1)])
    coset_intt(v, g) = intt(v) * [g^(-0), g^(-1), ..., g^(-(n-1))]
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import jax.numpy as jnp

if TYPE_CHECKING:
    from jax import Array

    from .ntt import NTT


def coset_ntt(ntt_instance: NTT, coeffs: Array, shift: Array) -> Array:
    """Compute NTT on a coset (shifted domain).

    Evaluates the polynomial at points {shift * omega^i} instead of {omega^i}.

    Args:
        ntt_instance: The NTT implementation to use.
        coeffs: Polynomial coefficients.
        shift: The coset generator (domain shift).

    Returns:
        Evaluations on the coset.
    """
    n = coeffs.shape[0]
    dtype = coeffs.dtype

    # Compute shift powers via O(log n) doubling:
    #   Step 0: [1]
    #   Step 1: [1, shift]
    #   Step 2: [1, shift, shift², shift³]
    #   ...
    one = dtype.type(1)
    shift_powers = jnp.array([one], dtype=dtype)
    step_mul = shift
    for _ in range(int(math.log2(n))):
        shift_powers = jnp.concatenate([shift_powers, shift_powers * step_mul])
        step_mul = step_mul * step_mul

    # Multiply coefficients by shift powers
    shifted_coeffs = coeffs * shift_powers

    # Apply standard NTT
    return ntt_instance.forward(shifted_coeffs)


def coset_intt(ntt_instance: NTT, evaluations: Array, shift: Array) -> Array:
    """Compute inverse NTT from coset evaluations.

    Inverse of coset_ntt - recovers polynomial coefficients from coset evaluations.

    Args:
        ntt_instance: The NTT implementation to use.
        evaluations: Evaluations on the coset.
        shift: The coset generator (domain shift).

    Returns:
        Polynomial coefficients.
    """
    n = evaluations.shape[0]
    dtype = evaluations.dtype

    # Apply standard inverse NTT
    coeffs = ntt_instance.inverse(evaluations)

    # Compute inverse shift powers via O(log n) doubling.
    # shift⁻¹ = shift^(p - 2) by Fermat's little theorem.
    one = dtype.type(1)
    shift_inv = one / shift

    inv_shift_powers = jnp.array([one], dtype=dtype)
    step_mul = shift_inv
    for _ in range(int(math.log2(n))):
        inv_shift_powers = jnp.concatenate(
            [inv_shift_powers, inv_shift_powers * step_mul]
        )
        step_mul = step_mul * step_mul

    # Multiply by inverse shift powers
    return coeffs * inv_shift_powers
