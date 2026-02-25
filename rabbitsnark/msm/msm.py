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

"""MSM (Multi-Scalar Multiplication) utilities.

MSM computes the sum: sum_{i=0}^{n-1} s_i * P_i

where s_i are scalars and P_i are elliptic curve points.

This module provides static methods used inside the JIT-compiled Groth16
prove kernel:

- ``decompose_scalars``: JIT-compatible scalar windowing
- ``pippenger``: P-partition Pippenger bucket method
- ``estimate_optimal_window_bits``: Window size estimation
- ``affine_to_xyzz`` / ``ec_zeros`` / ``to_xyzz_dtype``: EC array utilities
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
import zk_dtypes
from jax import dtypes, lax

if TYPE_CHECKING:
    from jax import Array


class MSM:
    """Static utility methods for Pippenger MSM inside JIT context."""

    # ------------------------------------------------------------------
    # Cost model
    # ------------------------------------------------------------------

    @staticmethod
    def estimate_optimal_window_bits(scalar_bits: int, num_points: int) -> int:
        """Estimate optimal window size for Pippenger's algorithm.

        Cost model: ``ceil(scalar_bits / s) * (num_points + 2ˢ - 1)``

        Args:
            scalar_bits: Number of bits in scalars.
            num_points: Number of scalar-point pairs.

        Returns:
            Optimal window size in bits.
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

    # ------------------------------------------------------------------
    # JIT-internal methods (called inside JIT context)
    # ------------------------------------------------------------------

    @staticmethod
    def decompose_scalars(
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

    @staticmethod
    def pippenger(
        wi: Array,
        points: Array,
        zero: Array,
        buckets_2d: Array,
        ws_2d: Array,
        scalar_bits: int,
        window_bits: int,
        num_parts: int,
        chunk_size: int,
    ) -> Array:
        """P-partition Pippenger bucket method.

        Called inside JIT context. Splits input into P partitions, each
        running a full Pippenger. Partition results are summed to produce
        the final MSM result.

        Args:
            wi: ``[W, padded_n]`` int32 window indices.
            points: ``[padded_n]`` XYZZ EC points.
            zero: Scalar EC identity (0-D, reduce init).
            buckets_2d: ``[P, 1 << window_bits]`` EC zero array.
            ws_2d: ``[P, W]`` EC zero array.
            scalar_bits: Number of bits per scalar.
            window_bits: Window size in bits.
            num_parts: Number of data partitions (P).
            chunk_size: Points per partition (C).

        Returns:
            Single XYZZ EC point.
        """
        num_windows = (scalar_bits + window_bits - 1) // window_bits
        num_bk = (1 << window_bits) - 1

        partition_results = []
        for p in range(num_parts):
            start = p * chunk_size
            chunk_pts = lax.dynamic_slice_in_dim(points, start, chunk_size, 0)
            chunk_wi = lax.dynamic_slice_in_dim(wi, start, chunk_size, 1)

            window_sum_list = []
            for w in range(num_windows):
                bk = buckets_2d[p].at[chunk_wi[w]].add(chunk_pts)
                actual_bk = bk[1:]

                def _reduce(j, state, *, ab=actual_bk):
                    running, wsum = state
                    running = running + ab[num_bk - 1 - j]
                    wsum = wsum + running
                    return running, wsum

                _, wsum = lax.fori_loop(0, num_bk, _reduce, (zero, ws_2d[p][w]))
                window_sum_list.append(wsum)

            result_p = window_sum_list[num_windows - 1]
            for w in range(num_windows - 2, -1, -1):
                result_p = lax.fori_loop(
                    0, window_bits, lambda _, pt: pt + pt, result_p
                )
                result_p = result_p + window_sum_list[w]
            partition_results.append(result_p)

        final = partition_results[0]
        for p in range(1, num_parts):
            final = final + partition_results[p]
        return final

    # ------------------------------------------------------------------
    # Compile-time utilities
    # ------------------------------------------------------------------

    @staticmethod
    def affine_to_xyzz(points: Array, xyzz_dtype) -> Array:
        """Convert affine EC points to XYZZ representation.

        Affine (x, y) → XYZZ (x, y, zz=1, zzz=1).
        """
        ctor = xyzz_dtype.type if hasattr(xyzz_dtype, "type") else xyzz_dtype
        np_points = np.array(points)
        xyzz_list = [ctor((*p.item().raw, 1, 1)) for p in np_points]
        return jnp.array(xyzz_list, dtype=xyzz_dtype)

    @staticmethod
    def ec_zeros(shape: int | tuple[int, ...], dtype) -> jnp.ndarray:
        """Create a JAX array of EC identity points.

        ``jnp.zeros`` cannot create EC-typed arrays because numpy has no cast
        path from ``int64`` → EC dtypes.  This helper builds the array via the
        dtype constructor instead.

        For the scalar (0-D) case, we create a 1-element array and reshape to
        avoid the ``convert_element_type`` path that has a buffer size mismatch
        for XYZZ types.
        """
        ctor = dtype.type if hasattr(dtype, "type") else dtype
        identity = ctor(0)
        if isinstance(shape, int):
            shape = (shape,)
        n = max(math.prod(shape), 1) if shape else 1
        arr = jnp.array([identity] * n, dtype=dtype)
        return arr.reshape(shape) if shape else arr.reshape(())

    @staticmethod
    def to_xyzz_dtype(point_dtype):
        """Map any EC point dtype to the corresponding XYZZ dtype."""
        info = dtypes.ecinfo(point_dtype)
        if info.point_repr == "xyzz":
            return point_dtype

        suffix = "_mont" if info.is_montgomery else ""
        name = f"bn254_{info.curve_group}_xyzz{suffix}"
        return getattr(zk_dtypes, name)
