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

"""CSR (Compressed Sparse Row) matrix for BN254 scalar field.

Stores sparse matrices in CSR format with an ELL (ELLPACK) view for
JIT-compatible SpMV. The ZKX backend does not support scatter operations
on ZK field types under JIT, so the SpMV kernel uses a vectorized
gather + reshape + sum approach instead of fori_loop with scatter.

CSR layout:
    row_ptrs[i]   = start index of row i in col_indices/values
    row_ptrs[i+1] = end index (exclusive)
    col_indices[k] = column of the k-th nonzero
    values[k]      = value of the k-th nonzero

ELL layout (derived from CSR, padded to max nonzeros per row):
    ell_col_indices: flat (n_rows * max_nnz_per_row,), padded with 0
    ell_values: flat (n_rows * max_nnz_per_row,), padded with field zero
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from rabbitsnark.circom.zkey.coefficient import Coefficient


@dataclass
class CSRMatrix:
    """CSR sparse matrix with Montgomery-form field element values.

    Attributes:
        row_ptrs: Row pointer array (numpy), shape (n_rows + 1,).
        col_indices: Column index array (JAX int32), shape (nnz,).
        values: Nonzero values in Montgomery form, shape (nnz,).
        n_rows: Number of rows.
        n_cols: Number of columns.
        ell_col_indices: ELL-format column indices, flat (n_rows * max_nnz_per_row,).
        ell_values: ELL-format values, flat (n_rows * max_nnz_per_row,).
        max_nnz_per_row: Maximum nonzeros in any single row.
    """

    row_ptrs: np.ndarray
    col_indices: jnp.ndarray
    values: jnp.ndarray
    n_rows: int
    n_cols: int
    ell_col_indices: jnp.ndarray
    ell_values: jnp.ndarray
    max_nnz_per_row: int

    @property
    def nnz(self) -> int:
        """Number of nonzero elements."""
        return self.values.shape[0]

    @staticmethod
    def _build_ell(
        row_ptrs: np.ndarray,
        col_indices_list: list[int],
        value_ints: list[int],
        n_rows: int,
        dtype: type,
    ) -> tuple[jnp.ndarray, jnp.ndarray, int]:
        """Build ELL-format arrays from CSR data.

        Returns:
            (ell_col_indices, ell_values, max_nnz_per_row)
        """
        row_lengths = np.diff(row_ptrs)
        max_nnz = int(row_lengths.max()) if n_rows > 0 and len(row_lengths) > 0 else 0
        if max_nnz == 0:
            max_nnz = 1  # Avoid zero-size arrays

        ell_cols = [0] * (n_rows * max_nnz)
        ell_vals = [0] * (n_rows * max_nnz)

        for i in range(n_rows):
            start = int(row_ptrs[i])
            end = int(row_ptrs[i + 1])
            for k, idx in enumerate(range(start, end)):
                ell_cols[i * max_nnz + k] = col_indices_list[idx]
                ell_vals[i * max_nnz + k] = value_ints[idx]

        ell_col_indices = jnp.array(ell_cols, dtype=jnp.int32)
        ell_values = jnp.array(ell_vals, dtype=dtype)
        return ell_col_indices, ell_values, max_nnz

    @classmethod
    def from_coefficients(
        cls,
        coefficients: list[Coefficient],
        is_matrix_a: bool,
        n_rows: int,
        n_cols: int,
        dtype: type,
        modulus: int,
    ) -> CSRMatrix:
        """Build CSR from zkey Coefficient list.

        Coefficients are parsed in standard form. This method converts them
        to Montgomery form for computation (Montgomery form is a ring
        isomorphism: Mont(a * b) = Mont(a) * Mont(b)).

        Args:
            coefficients: List of Coefficient objects from zkey parser.
            is_matrix_a: If True, filter matrix A coefficients; else matrix B.
            n_rows: Number of constraints (rows).
            n_cols: Number of signals (columns).
            dtype: ZK field dtype (e.g., bn254_sf_mont).
            modulus: Scalar field modulus for Montgomery conversion.
        """
        # Filter by matrix type
        if is_matrix_a:
            filtered = [c for c in coefficients if c.is_matrix_a()]
        else:
            filtered = [c for c in coefficients if c.is_matrix_b()]

        # Sort by (constraint, signal) for CSR ordering
        filtered.sort(key=lambda c: (c.constraint, c.signal))

        # Build CSR arrays
        row_ptrs = np.zeros(n_rows + 1, dtype=np.int32)
        col_indices_list = []
        value_ints = []

        for coeff in filtered:
            row_ptrs[coeff.constraint + 1] += 1
            col_indices_list.append(coeff.signal)
            # Values are in standard form from the parser. The bn254_sf_mont
            # dtype auto-converts to Montgomery form on array creation, so we
            # pass raw standard-form values without explicit conversion.
            value_ints.append(coeff.value)

        # Cumulative sum for row pointers
        np.cumsum(row_ptrs, out=row_ptrs)

        # Convert to JAX arrays via Python lists (avoids device_put issues)
        values = jnp.array(value_ints, dtype=dtype)
        col_indices_jax = jnp.array(col_indices_list, dtype=jnp.int32)

        # Build ELL-format arrays for JIT-compatible SpMV
        ell_col_indices, ell_values, max_nnz = cls._build_ell(
            row_ptrs,
            col_indices_list,
            value_ints,
            n_rows,
            dtype,
        )

        return cls(
            row_ptrs=row_ptrs,
            col_indices=col_indices_jax,
            values=values,
            n_rows=n_rows,
            n_cols=n_cols,
            ell_col_indices=ell_col_indices,
            ell_values=ell_values,
            max_nnz_per_row=max_nnz,
        )

    @classmethod
    def from_arrays(
        cls,
        row_ptrs: np.ndarray,
        col_indices: np.ndarray,
        values: jnp.ndarray,
        n_rows: int,
        n_cols: int,
    ) -> CSRMatrix:
        """Build CSR from raw arrays (for testing).

        Values must already be in the target dtype/form.
        """
        col_indices_jax = jnp.array(col_indices.tolist(), dtype=jnp.int32)

        # Build ELL-format arrays
        # Convert values to Python int list for _build_ell
        value_ints = [int(v) for v in values]
        ell_col_indices, ell_values, max_nnz = cls._build_ell(
            row_ptrs,
            col_indices.tolist(),
            value_ints,
            n_rows,
            values.dtype,
        )

        return cls(
            row_ptrs=row_ptrs,
            col_indices=col_indices_jax,
            values=values,
            n_rows=n_rows,
            n_cols=n_cols,
            ell_col_indices=ell_col_indices,
            ell_values=ell_values,
            max_nnz_per_row=max_nnz,
        )
