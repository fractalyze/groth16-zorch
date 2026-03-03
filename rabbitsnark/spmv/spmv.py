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

"""Sparse matrix-vector multiplication (SpMV) for BN254 scalar field.

Implements y = A @ x where A is a sparse matrix and x is a dense vector,
both in Montgomery form.

Uses SELL (Sliced ELL) format: rows are sorted by NNZ count and partitioned
into groups with similar density. Each partition is padded to its own
max_nnz, reducing memory waste from 80-95% (plain ELL) to near-optimal.

Per-partition SpMV:
    1. Gather:  gathered[k] = x[col_indices_p[k]]
    2. Multiply: products[k] = values_p[k] * gathered[k]
    3. Reshape + sum: y_p[i] = sum(products[i, :])

Results are concatenated and reordered via inverse_perm gather.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import jit

if TYPE_CHECKING:
    from jax import Array

    from .sell import SELLConfig, SELLMatrix


@partial(jit, static_argnums=(0,))
def _spmv_sell_kernel(
    config: SELLConfig,
    x: Array,
    inverse_perm: Array,
    *partition_arrays: Array,
) -> Array:
    """JIT-compiled SELL SpMV kernel with unrolled partition loop.

    Each partition is processed independently: gather -> multiply -> reshape ->
    sum. Results are concatenated and reordered via inverse_perm gather.

    The partition loop is unrolled at trace time (Python-level for loop),
    producing P separate HLO fusions -- one per partition.

    Args:
        config: Static SELL configuration (used via static_argnums).
        x: Dense input vector (Montgomery form), shape (n_cols,).
        inverse_perm: Gather indices to restore original row order, (n_rows,).
        *partition_arrays: Alternating (col_indices_p, values_p) for P
            partitions, each flat with size partition_sizes[p] *
            partition_max_nnz[p].

    Returns:
        Result vector y = A @ x, shape (n_rows,).
    """
    results = []
    for p in range(config.num_partitions):
        col_p = partition_arrays[2 * p]
        val_p = partition_arrays[2 * p + 1]
        gathered = x[col_p]
        products = val_p * gathered
        y_p = jnp.sum(
            products.reshape(config.partition_sizes[p], config.partition_max_nnz[p]),
            axis=1,
        )
        results.append(y_p)
    y_sorted = jnp.concatenate(results)
    return y_sorted[inverse_perm]


def spmv_sell(matrix: SELLMatrix, x: Array) -> Array:
    """Compute y = A @ x using SELL-format JIT kernel.

    Both the matrix values and x must be in Montgomery form.

    Args:
        matrix: SELL sparse matrix.
        x: Dense input vector, shape (n_cols,).

    Returns:
        Dense result vector, shape (n_rows,).
    """
    # Flatten partition arrays into alternating (col, val, col, val, ...)
    partition_arrays = []
    for p in range(matrix.config.num_partitions):
        partition_arrays.append(matrix.partition_col_indices[p])
        partition_arrays.append(matrix.partition_values[p])

    return _spmv_sell_kernel(
        matrix.config,
        x,
        matrix.inverse_perm,
        *partition_arrays,
    )
