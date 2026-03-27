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

"""Convert zkey coefficients to CSR format for r1cs-solver compute_abc.

The zkey file stores R1CS coefficients as a list of (matrix, constraint,
signal, value) tuples.  This module converts them to CSR format compatible
with the native ``compute_abc`` function from r1cs-solver.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from rabbitsnark.circom.zkey.zkey import ZKeyV1
    from rabbitsnark.r1cs_solver import CSRMatrices

FIELD_ELEM_SIZE = 32  # 256-bit BN254 scalar = 32 bytes


def zkey_to_csr(zkey: "ZKeyV1") -> "CSRMatrices":
    """Convert zkey coefficients to CSR format.

    Builds CSR matrices for A and B from zkey coefficients.
    C matrix is empty (Groth16 computes Cz = Az ⊙ Bz via Hadamard).

    Args:
        zkey: Parsed proving key (ZKeyV1).

    Returns:
        CSRMatrices with A, B populated and C empty.
    """
    from rabbitsnark.r1cs_solver import CSRMatrices

    n = zkey.domain_size
    modulus = zkey.header_groth.r.to_int()

    # Collect COO entries for A and B
    a_coo: dict[tuple[int, int], int] = {}
    b_coo: dict[tuple[int, int], int] = {}

    for coeff in zkey.coefficients:
        row = coeff.constraint
        col = coeff.signal
        val = coeff.value
        if coeff.matrix == 0:
            a_coo[(row, col)] = (a_coo.get((row, col), 0) + val) % modulus
        else:
            b_coo[(row, col)] = (b_coo.get((row, col), 0) + val) % modulus

    def _build_csr(
        coo: dict[tuple[int, int], int], num_rows: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert COO dict to CSR arrays."""
        # Sort by (row, col)
        entries = sorted(
            ((r, c, v) for (r, c), v in coo.items() if v != 0),
            key=lambda x: (x[0], x[1]),
        )
        nnz = len(entries)

        indices = np.zeros(nnz, dtype=np.int32) if nnz else np.array([], dtype=np.int32)
        values = (
            np.zeros((nnz, FIELD_ELEM_SIZE), dtype=np.uint8)
            if nnz
            else np.zeros((0, FIELD_ELEM_SIZE), dtype=np.uint8)
        )

        for i, (row, col, val) in enumerate(entries):
            indices[i] = col
            values[i] = np.frombuffer(
                val.to_bytes(FIELD_ELEM_SIZE, "little"), dtype=np.uint8
            )

        # Build CSR indptr: cumulative count of nonzeros per row
        indptr = np.zeros(num_rows + 1, dtype=np.int64)
        for row, _col, _val in entries:
            indptr[row + 1] += 1
        np.cumsum(indptr, out=indptr)

        return indptr, indices, values

    a_indptr, a_indices, a_values = _build_csr(a_coo, n)
    b_indptr, b_indices, b_values = _build_csr(b_coo, n)

    # Empty C matrix (Groth16: Cz = Az ⊙ Bz)
    c_indptr = np.zeros(n + 1, dtype=np.int64)
    c_indices = np.array([], dtype=np.int32)
    c_values = np.zeros((0, FIELD_ELEM_SIZE), dtype=np.uint8)

    return CSRMatrices(
        a_indptr=a_indptr,
        a_indices=a_indices,
        a_values=a_values,
        b_indptr=b_indptr,
        b_indices=b_indices,
        b_values=b_values,
        c_indptr=c_indptr,
        c_indices=c_indices,
        c_values=c_values,
    )
