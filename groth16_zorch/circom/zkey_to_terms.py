# Copyright 2026 The Groth16Zorch Authors.
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

"""Convert zkey coefficients to term-based format.

The zkey file stores R1CS coefficients as a list of (matrix, constraint,
signal, value) tuples.  This module converts them to the term-based
``TermMatrices`` consumed by ``groth16_zorch.r1cs.compute_abc``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from groth16_zorch.circom.zkey.zkey import ZKeyV1
    from groth16_zorch.r1cs import TermMatrices

FIELD_ELEM_SIZE = 32  # 256-bit BN254 scalar = 32 bytes


def zkey_to_terms(zkey: "ZKeyV1") -> tuple["TermMatrices", np.ndarray]:
    """Convert zkey coefficients to term-based format.

    Builds term matrices for A and B from zkey coefficients.
    C is omitted — Groth16 recovers Cz = Az ⊙ Bz via a Hadamard product.

    Args:
        zkey: Parsed proving key (ZKeyV1).

    Returns:
        Tuple of (TermMatrices, coefficients) where coefficients is
        (num_coefficients, 32) uint8 array.
    """
    from groth16_zorch.r1cs import TermMatrices

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

    # Build deduplicated coefficient table
    all_values: set[bytes] = set()
    for coo in [a_coo, b_coo]:
        for val in coo.values():
            if val != 0:
                all_values.add(val.to_bytes(FIELD_ELEM_SIZE, "little"))
    coeff_list = sorted(all_values)
    coeff_to_id = {v: i for i, v in enumerate(coeff_list)}

    def _build_terms(
        coo: dict[tuple[int, int], int], num_rows: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Convert COO dict to term arrays (offsets + interleaved terms)."""
        entries = sorted(
            ((r, c, v) for (r, c), v in coo.items() if v != 0),
            key=lambda x: (x[0], x[1]),
        )
        nnz = len(entries)

        offsets = np.zeros(num_rows + 1, dtype=np.int64)
        terms = np.zeros(nnz * 2, dtype=np.int32)

        for i, (row, col, val) in enumerate(entries):
            val_bytes = val.to_bytes(FIELD_ELEM_SIZE, "little")
            terms[i * 2] = coeff_to_id[val_bytes]
            terms[i * 2 + 1] = col
            offsets[row + 1] += 1
        np.cumsum(offsets, out=offsets)

        return offsets, terms

    a_offsets, a_terms = _build_terms(a_coo, n)
    b_offsets, b_terms = _build_terms(b_coo, n)

    # Build coefficient table as (num_coefficients, 32) uint8
    coeff_array = (
        np.array([list(v) for v in coeff_list], dtype=np.uint8).reshape(
            -1, FIELD_ELEM_SIZE
        )
        if coeff_list
        else np.zeros((0, FIELD_ELEM_SIZE), dtype=np.uint8)
    )

    return (
        TermMatrices(
            a_offsets=a_offsets,
            a_terms=a_terms,
            b_offsets=b_offsets,
            b_terms=b_terms,
        ),
        coeff_array,
    )
