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

"""Pippenger's algorithm for MSM (Multi-Scalar Multiplication).

Window-outer Pippenger using EC point dtypes.  The ``+`` operator on EC point
arrays (e.g. ``bn254_g1_xyzz``) automatically lowers to
``prime_ir::elliptic_curve::AddOp``, and ``P + P`` is canonicalized to
``ec.double``.  This module therefore contains **no manual point arithmetic**;
all group operations are delegated to the compiler via ``lax.add``.

The algorithm has four phases:
1. Scalar Decomposition: Split scalars into window slices
2. Bucket Accumulation: Add points to buckets based on scalar windows
3. Bucket Reduction: Reduce each window's buckets to a single point
4. Window Reduction: Combine window results into final MSM result

Reference: https://encrypt.a41.io/primitives/abstract-algebra/elliptic-curve/msm/pippengers-algorithm
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import zk_dtypes
from jax import dtypes, lax

if TYPE_CHECKING:
    from jax import Array


def _to_xyzz_dtype(point_dtype):
    """Map any EC point dtype to the corresponding XYZZ dtype."""
    info = dtypes.ecinfo(point_dtype)
    if info.point_repr == "xyzz":
        return point_dtype

    suffix = "_mont" if info.is_montgomery else ""
    name = f"bn254_{info.curve_group}_xyzz{suffix}"
    return getattr(zk_dtypes, name)


def pippenger_msm(
    scalars: Array,
    points: Array,
    *,
    scalar_bits: int = 254,
    window_bits: int | None = None,
    num_chunks: int = 1,
) -> Array:
    """Compute MSM using Pippenger's bucket method with EC point dtypes.

    Args:
        scalars: Array of scalar field elements (shape: ``[n]``).
        points: Array of EC points (shape: ``[n]``, e.g. ``bn254_g1_affine``).
        scalar_bits: Number of bits in scalar field (default 254 for BN254).
        window_bits: Number of bits per window.  ``None`` ⇒ auto-estimated.
        num_chunks: Parallel chunks for ``vmap`` (default 1).

    Returns:
        Single EC point (XYZZ representation).
    """
    n = scalars.shape[0]

    if window_bits is None:
        window_bits = _estimate_optimal_window_bits(scalar_bits, n)

    xyzz_dtype = _to_xyzz_dtype(points.dtype)
    zero = jnp.zeros((), dtype=xyzz_dtype)

    if num_chunks <= 1:
        return _pippenger_msm(scalars, points, zero, scalar_bits, window_bits)

    chunk_size = math.ceil(n / num_chunks)
    pad_total = chunk_size * num_chunks

    # Pad scalars with zeros and points with identity
    scalars_padded = jnp.zeros(pad_total, dtype=scalars.dtype)
    scalars_padded = scalars_padded.at[:n].set(scalars)
    points_padded = jnp.zeros(pad_total, dtype=points.dtype)
    points_padded = points_padded.at[:n].set(points)

    scalars_chunks = scalars_padded.reshape(num_chunks, chunk_size)
    points_chunks = points_padded.reshape(num_chunks, chunk_size)

    chunk_results = jax.vmap(
        lambda s, p: _pippenger_msm(s, p, zero, scalar_bits, window_bits)
    )(scalars_chunks, points_chunks)

    # Reduce chunk results: [num_chunks] → single point
    def _reduce(i, acc):
        return acc + chunk_results[i]

    return lax.fori_loop(0, num_chunks, _reduce, zero)


def _estimate_optimal_window_bits(scalar_bits: int, num_points: int) -> int:
    """Estimate optimal window size for Pippenger's algorithm.

    Cost model: ``ceil(scalar_bits / s) * (num_points + 2^s - 1)``
    """
    best_cost = float("inf")
    best_s = 1

    for s in range(1, scalar_bits + 1):
        num_windows = math.ceil(scalar_bits / s)
        buckets_per_window = (1 << s) - 1
        cost = num_windows * (num_points + buckets_per_window)

        if cost < best_cost:
            best_cost = cost
            best_s = s
        elif cost > best_cost:
            break

    return best_s


def _pippenger_msm(
    scalars: Array,
    points: Array,
    zero: Array,
    scalar_bits: int,
    window_bits: int,
) -> Array:
    """Window-outer Pippenger MSM using EC point dtypes.

    Args:
        scalars: ``[n]`` scalar field array.
        points: ``[n]`` EC point array (any representation).
        zero: Scalar identity point (XYZZ, zero-initialized).
        scalar_bits: Number of bits per scalar.
        window_bits: Window size in bits.

    Returns:
        Single XYZZ EC point.
    """
    n = scalars.shape[0]
    num_windows = (scalar_bits + window_bits - 1) // window_bits
    num_buckets = (1 << window_bits) - 1
    mask = (1 << window_bits) - 1

    xyzz_dtype = zero.dtype

    # Window sums: [num_windows] XYZZ points
    window_sums = jnp.zeros(num_windows, dtype=xyzz_dtype)

    def _process_window(w, window_sums):
        # 1-D bucket array per window
        buckets = jnp.zeros(num_buckets, dtype=xyzz_dtype)

        # Accumulate all n points for this window
        def _acc_point(i, buckets):
            window_slice = (scalars[i] >> (w * window_bits)) & mask

            def _add():
                idx = window_slice - 1
                return buckets.at[idx].set(buckets[idx] + points[i])

            return lax.cond(window_slice > 0, _add, lambda: buckets)

        buckets = lax.fori_loop(0, n, _acc_point, buckets)

        # Bucket reduction: running + window_sum
        def _reduce(j, state):
            running, wsum = state
            running = running + buckets[num_buckets - 1 - j]
            wsum = wsum + running
            return running, wsum

        _, window_sum = lax.fori_loop(0, num_buckets, _reduce, (zero, zero))
        return window_sums.at[w].set(window_sum)

    window_sums = lax.fori_loop(0, num_windows, _process_window, window_sums)

    # Window combination via Horner's rule (MSB → LSB)
    def _combine(w_rev, result):
        w = num_windows - 2 - w_rev

        # Double `window_bits` times: P + P → ec.double via canonicalization
        def _double(_, p):
            return p + p

        result = lax.fori_loop(0, window_bits, _double, result)
        return result + window_sums[w]

    result = window_sums[num_windows - 1]
    return lax.fori_loop(0, num_windows - 1, _combine, result)
