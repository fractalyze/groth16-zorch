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

Stores sparse matrices in CSR format as an intermediate representation
for SELL conversion. The SELL (Sliced ELL) format is used for
JIT-compatible SpMV.

CSR layout:
    row_ptrs[i]   = start index of row i in col_indices/values
    row_ptrs[i+1] = end index (exclusive)
    col_indices[k] = column of the k-th nonzero
    values[k]      = value of the k-th nonzero
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
    """

    row_ptrs: np.ndarray
    col_indices: jnp.ndarray
    values: jnp.ndarray
    n_rows: int
    n_cols: int

    @property
    def nnz(self) -> int:
        """Number of nonzero elements."""
        return self.values.shape[0]

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

        return cls(
            row_ptrs=row_ptrs,
            col_indices=col_indices_jax,
            values=values,
            n_rows=n_rows,
            n_cols=n_cols,
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

        return cls(
            row_ptrs=row_ptrs,
            col_indices=col_indices_jax,
            values=values,
            n_rows=n_rows,
            n_cols=n_cols,
        )
