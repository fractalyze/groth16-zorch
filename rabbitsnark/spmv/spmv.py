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

Uses ELL (ELLPACK) format: the sparse matrix is stored as padded 2D arrays
(n_rows, max_nnz_per_row) so that SpMV reduces to:
    1. Gather:  gathered[k] = x[ell_col_indices[k]]
    2. Multiply: products[k] = ell_values[k] * gathered[k]
    3. Reshape + sum: y[i] = sum(products[i, :])

This avoids scatter operations (y.at[i].set()) which the ZKX backend does
not support under JIT for ZK field types. Phase 2 will transparently
replace this with the ZKX backend's native CSR SpMV.
"""

from __future__ import annotations

from functools import partial
from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import jit

if TYPE_CHECKING:
    from jax import Array

    from .csr_matrix import CSRMatrix


@partial(jit, static_argnums=(3, 4))
def _spmv_kernel(
    ell_col_indices: Array,
    ell_values: Array,
    x: Array,
    n_rows: int,
    max_nnz_per_row: int,
) -> Array:
    """JIT-compiled SpMV kernel using vectorized ELL-format operations.

    Args:
        ell_col_indices: ELL column indices, flat (n_rows * max_nnz_per_row,).
        ell_values: ELL values (Montgomery form), flat (n_rows * max_nnz_per_row,).
            Padding slots have value 0 (field zero).
        x: Dense input vector (Montgomery form), shape (n_cols,).
        n_rows: Number of rows in the sparse matrix (static).
        max_nnz_per_row: Maximum nonzeros per row (static).

    Returns:
        Result vector y = A @ x, shape (n_rows,).
    """
    # Step 1: Gather x values at column indices
    gathered = x[ell_col_indices]

    # Step 2: Element-wise multiply (padding slots contribute 0 * x[0] = 0)
    products = ell_values * gathered

    # Step 3: Reshape to (n_rows, max_nnz_per_row) and sum along columns
    return jnp.sum(products.reshape(n_rows, max_nnz_per_row), axis=1)


def spmv(matrix: CSRMatrix, x: Array) -> Array:
    """Compute y = A @ x with automatic backend selection.

    Tries the ZKX native CSR SpMV backend first (via MLIR construction
    and EmitMatrixVectorMultiplicationOp). Falls back to the ELL-format
    JIT kernel if the backend path fails.

    Both the matrix values and x must be in Montgomery form.

    Args:
        matrix: CSR sparse matrix (with ELL-format view).
        x: Dense input vector, shape (n_cols,).

    Returns:
        Dense result vector, shape (n_rows,).
    """
    try:
        from .backend import spmv_backend

        return spmv_backend(matrix, x)
    except Exception:
        return _spmv_kernel(
            matrix.ell_col_indices,
            matrix.ell_values,
            x,
            matrix.n_rows,
            matrix.max_nnz_per_row,
        )
