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
from functools import partial
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp
import numpy as np
import zk_dtypes
from jax import dtypes, lax

if TYPE_CHECKING:
    from jax import Array


def _decompose_scalars(scalars: Array, scalar_bits: int, window_bits: int) -> Array:
    """Decompose field scalars into integer window indices.

    Returns an int32 array of shape ``[num_windows, n]`` where
    ``result[w][i] = (int(scalars[i]) >> (w * window_bits)) & mask``.
    """
    mask = (1 << window_bits) - 1
    num_windows = (scalar_bits + window_bits - 1) // window_bits
    n = scalars.shape[0]

    scalar_ints = [int(s) for s in scalars]
    indices = np.empty((num_windows, n), dtype=np.int32)
    for w in range(num_windows):
        shift = w * window_bits
        for i in range(n):
            indices[w, i] = (scalar_ints[i] >> shift) & mask

    return jnp.array(indices)


def _decompose_scalars_jit(
    scalars_std: Array, scalar_bits: int, window_bits: int
) -> Array:
    """JIT-compatible scalar decomposition via bitcast to raw bytes.

    bn254_sf[n] → uint8[n, 32] (little-endian), then extract window
    indices using JAX integer ops. All operations are JIT-traceable.

    Args:
        scalars_std: Scalar field elements in standard form (bn254_sf).
        scalar_bits: Number of bits per scalar (e.g. 254).
        window_bits: Window size in bits.

    Returns:
        int32 array of shape [num_windows, n].
    """
    num_windows = (scalar_bits + window_bits - 1) // window_bits
    mask = (1 << window_bits) - 1

    raw_bytes = lax.bitcast_convert_type(scalars_std, jnp.uint8)  # [n, 32]

    windows = []
    for w in range(num_windows):
        start_bit = w * window_bits
        byte_idx = start_bit // 8
        bit_offset = start_bit % 8

        val = raw_bytes[:, byte_idx].astype(jnp.int32)
        if byte_idx + 1 < 32:
            val = val | (raw_bytes[:, byte_idx + 1].astype(jnp.int32) << 8)
        if byte_idx + 2 < 32:
            val = val | (raw_bytes[:, byte_idx + 2].astype(jnp.int32) << 16)

        windows.append((val >> bit_offset) & mask)

    return jnp.stack(windows)  # [num_windows, n]


def _affine_to_xyzz(points: Array, xyzz_dtype) -> Array:
    """Convert affine EC points to XYZZ representation.

    Affine (x, y) → XYZZ (x, y, zz=1, zzz=1).
    """
    ctor = xyzz_dtype.type if hasattr(xyzz_dtype, "type") else xyzz_dtype
    np_points = np.array(points)
    xyzz_list = [ctor((*p.item().raw, 1, 1)) for p in np_points]
    return jnp.array(xyzz_list, dtype=xyzz_dtype)


def _ec_zeros(shape: int | tuple[int, ...], dtype) -> jnp.ndarray:
    """Create a JAX array of EC identity points.

    ``jnp.zeros`` cannot create EC-typed arrays because numpy has no cast
    path from ``int64`` → EC dtypes.  This helper builds the array via the
    dtype constructor instead.

    For the scalar (0-D) case, we create a 1-element array and reshape to
    avoid the ``convert_element_type`` path that has a buffer size mismatch
    for XYZZ types.
    """
    # dtype may be a numpy.dtype wrapper; .type gives the raw constructor.
    ctor = dtype.type if hasattr(dtype, "type") else dtype
    identity = ctor(0)
    if isinstance(shape, int):
        shape = (shape,)
    n = max(math.prod(shape), 1) if shape else 1
    arr = jnp.array([identity] * n, dtype=dtype)
    return arr.reshape(shape) if shape else arr.reshape(())


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
        if n <= 4:
            window_bits = min(16, scalar_bits)
        else:
            window_bits = _estimate_optimal_window_bits(scalar_bits, n)

    xyzz_dtype = _to_xyzz_dtype(points.dtype)
    zero = _ec_zeros((), xyzz_dtype)

    # Convert points to XYZZ so all inner ops use same dtype.
    info = dtypes.ecinfo(points.dtype)
    if info.point_repr != "xyzz":
        points = _affine_to_xyzz(points, xyzz_dtype)

    # Pre-decompose scalars into window indices (int32) outside JIT.
    # Bitwise ops (>>, &) are not supported on prime field types.
    window_indices = _decompose_scalars(scalars, scalar_bits, window_bits)

    # Pre-allocate zero arrays outside JIT so _ec_zeros (Python-level
    # construction) does not become baked-in constants during tracing.
    num_windows = (scalar_bits + window_bits - 1) // window_bits
    num_buckets = (1 << window_bits) - 1
    window_sums_init = _ec_zeros(num_windows, xyzz_dtype)
    buckets_init = _ec_zeros(num_buckets, xyzz_dtype)

    if num_chunks <= 1:
        return _pippenger_msm(
            window_indices,
            points,
            zero,
            window_sums_init,
            buckets_init,
            scalar_bits,
            window_bits,
        )

    chunk_size = math.ceil(n / num_chunks)
    pad_total = chunk_size * num_chunks

    # Pad window_indices and points
    wi_padded = jnp.zeros((num_windows, pad_total), dtype=jnp.int32)
    wi_padded = wi_padded.at[:, :n].set(window_indices)
    points_padded = _ec_zeros(pad_total, xyzz_dtype)
    points_padded = points_padded.at[:n].set(points)

    # Reshape: [num_windows, num_chunks, chunk_size] and [num_chunks, chunk_size]
    wi_chunks = wi_padded.reshape(num_windows, num_chunks, chunk_size)
    points_chunks = points_padded.reshape(num_chunks, chunk_size)

    chunk_results = jax.vmap(
        lambda wi, p: _pippenger_msm(
            wi,
            p,
            zero,
            window_sums_init,
            buckets_init,
            scalar_bits,
            window_bits,
        )
    )(
        wi_chunks.transpose(1, 0, 2),  # [num_chunks, num_windows, chunk_size]
        points_chunks,
    )

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


@partial(jax.jit, static_argnums=(5, 6))
def _pippenger_msm(
    window_indices: Array,
    points: Array,
    zero: Array,
    window_sums_init: Array,
    buckets_init: Array,
    scalar_bits: int,
    window_bits: int,
) -> Array:
    """Window-outer Pippenger MSM using EC point dtypes.

    Args:
        window_indices: ``[num_windows, n]`` int32 array of window slices.
        points: ``[n]`` EC point array (XYZZ representation).
        zero: Scalar identity point (XYZZ, zero-initialized).
        window_sums_init: Pre-allocated ``[num_windows]`` zero XYZZ array.
        buckets_init: Pre-allocated ``[num_buckets]`` zero XYZZ array.
        scalar_bits: Number of bits per scalar.
        window_bits: Window size in bits.

    Returns:
        Single XYZZ EC point.
    """
    n = points.shape[0]
    num_windows = (scalar_bits + window_bits - 1) // window_bits
    num_buckets = (1 << window_bits) - 1

    window_sums = window_sums_init

    def _process_window(w, window_sums):
        buckets = buckets_init

        # Accumulate all n points for this window
        def _acc_point(i, buckets):
            window_slice = window_indices[w, i]

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
