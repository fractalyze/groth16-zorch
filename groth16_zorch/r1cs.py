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

"""Term-based R1CS matrices and Az/Bz evaluation in pure FRX.

The A/B sparse matrix-vector products (Az = A·z, Bz = B·z over BN254 Fr) are a
segmented sum of coefficient·wire products, which ``frx.ops.segment_sum``
computes directly over the ``bn254_sf_mont`` field dtype. Running on the
default device means Az/Bz land on the GPU alongside the rest of the prover,
with no native shared library and no CPU→GPU copy.

Addition is invariant under the Montgomery map, so the sum is taken directly in
Montgomery form — the result is byte-identical to a standard-form evaluation.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import frx
import frx.numpy as jnp
import numpy as np
from zk_dtypes import bn254_sf_mont

_MONT_DT = np.dtype(bn254_sf_mont)


@dataclass
class TermMatrices:
    """Term-based representation of the R1CS A and B matrices.

    Each term is a ``(coeff_id: u32, wire_id: i32)`` pair referencing a shared
    coefficient table. Storing term ids instead of 32-byte values keeps the
    matrices compact (~8 bytes/term vs 32). The C matrix is not stored —
    Groth16 recovers Cz = Az ⊙ Bz via a Hadamard product in the prover.

    Offsets are pre-padded to the NTT domain (``domain_size + 1`` entries).
    """

    a_offsets: np.ndarray  # (domain_size + 1,) int64
    a_terms: np.ndarray  # (nnz_a * 2,) int32 — interleaved (coeff_id, wire_id)

    b_offsets: np.ndarray
    b_terms: np.ndarray


@partial(frx.jit, static_argnames="num_segments")
def _segment_matvec(
    coeff: frx.Array,
    witness: frx.Array,
    coeff_ids: frx.Array,
    wire_ids: frx.Array,
    row_ids: frx.Array,
    num_segments: int,
) -> frx.Array:
    """Evaluate one matrix row-sum: out[r] = Σ coeff[coeff_id]·witness[wire_id]."""
    prod = coeff[coeff_ids] * witness[wire_ids]
    return frx.ops.segment_sum(prod, row_ids, num_segments=num_segments)


def _row_ids(offsets: np.ndarray) -> np.ndarray:
    """Map each flattened term to its constraint (row) index from CSR offsets."""
    counts = np.diff(offsets)
    return np.repeat(np.arange(offsets.size - 1, dtype=np.int32), counts)


def _matvec(
    coeff: frx.Array,
    witness: frx.Array,
    terms: np.ndarray,
    offsets: np.ndarray,
    domain_size: int,
) -> frx.Array:
    """Az / Bz for one matrix. Offsets are already padded to ``domain_size``."""
    if terms.size == 0:
        return jnp.zeros(domain_size, dtype=bn254_sf_mont)
    coeff_ids = jnp.asarray(terms[0::2])
    wire_ids = jnp.asarray(terms[1::2])
    row_ids = jnp.asarray(_row_ids(offsets))
    return _segment_matvec(coeff, witness, coeff_ids, wire_ids, row_ids, domain_size)


def compute_abc(
    witness: np.ndarray,
    terms: TermMatrices,
    coefficients: np.ndarray,
    domain_size: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute Az, Bz via term-based sparse mat-vec over BN254 Fr.

    C is intentionally not evaluated — Groth16 recovers Cz = Az ⊙ Bz via a
    Hadamard product in the prover.

    Args:
        witness: (num_wires,) bn254_sf_mont array (Montgomery form).
        terms: A/B term matrices (offsets already padded to the NTT domain).
        coefficients: (num_coefficients, 32) uint8 — Montgomery-form coeff table.
        domain_size: NTT domain size (power of 2); output length and row count.

    Returns:
        (az, bz) FRX arrays of shape (domain_size,) in Montgomery form.
    """
    coeff = jnp.asarray(
        coefficients.reshape(-1).view(_MONT_DT)
        if coefficients.size > 0
        else np.zeros(1, dtype=_MONT_DT)
    )
    w = jnp.asarray(np.ascontiguousarray(witness).view(_MONT_DT).reshape(-1))

    az = _matvec(coeff, w, terms.a_terms, terms.a_offsets, domain_size)
    bz = _matvec(coeff, w, terms.b_terms, terms.b_offsets, domain_size)
    return az, bz
