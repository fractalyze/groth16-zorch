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

"""R1CS sparse matrix construction from zkey files.

Provides utilities to:
1. Build CSR matrices A and B from parsed zkey coefficients
2. Convert witness data from standard form to Montgomery form
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp

if TYPE_CHECKING:
    import numpy as np
    from jax import Array

    from rabbitsnark.circom.zkey import ZKeyV1

from .csr_matrix import CSRMatrix


def build_r1cs_matrices(
    zkey: ZKeyV1,
    dtype: type,
) -> tuple[CSRMatrix, CSRMatrix]:
    """Build sparse A and B matrices from parsed zkey.

    The R1CS system is: A * z . B * z = C * z (Hadamard product).
    This builds the A and B matrices in CSR format with values in
    Montgomery form.

    Args:
        zkey: Parsed ZKeyV1 containing coefficients and header info.
        dtype: ZK field dtype for values (e.g., bn254_sf_mont).

    Returns:
        Tuple of (A, B) as CSRMatrix objects.
    """
    n_rows = zkey.domain_size
    n_cols = zkey.header_groth.num_vars

    matrix_a = CSRMatrix.from_coefficients(
        zkey.coefficients,
        is_matrix_a=True,
        n_rows=n_rows,
        n_cols=n_cols,
        dtype=dtype,
    )

    matrix_b = CSRMatrix.from_coefficients(
        zkey.coefficients,
        is_matrix_a=False,
        n_rows=n_rows,
        n_cols=n_cols,
        dtype=dtype,
    )

    return matrix_a, matrix_b


def witness_to_montgomery(
    witnesses: np.ndarray | list[int],
    dtype: type,
) -> Array:
    """Convert witness data from standard form to Montgomery form.

    Witness values from wtns files are in standard (non-Montgomery) form.
    This converts them to Montgomery form for use with SpMV.

    Args:
        witnesses: Witness values in standard form (integers).
        dtype: Target ZK field dtype (e.g., bn254_sf_mont).

    Returns:
        JAX array of witness values in Montgomery form.
    """
    # The bn254_sf_mont dtype auto-converts to Montgomery form on array
    # creation, so we pass standard-form values without explicit conversion.
    std_values = [int(w) for w in witnesses]
    return jnp.array(std_values, dtype=dtype)
