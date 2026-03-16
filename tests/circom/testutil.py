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

"""Shared test utilities for circom Groth16 tests."""

from __future__ import annotations

from typing import TYPE_CHECKING

import jax.numpy as jnp
from zk_dtypes import bn254_sf_mont

if TYPE_CHECKING:
    from jax import Array

    from rabbitsnark.circom.wtns.wtns import WtnsV2
    from rabbitsnark.circom.zkey.zkey import ZKeyV1


def compute_az_bz(
    zkey: ZKeyV1,
    wtns: WtnsV2,
) -> tuple[Array, Array]:
    """Compute A*z and B*z in Montgomery form from zkey coefficients.

    Simple dense implementation for testing (not suitable for large circuits).
    """
    n = zkey.domain_size
    m = zkey.header_groth.num_vars
    modulus = zkey.header_groth.r.to_int()

    # Build dense A, B as Python ints
    a_dense = [[0] * m for _ in range(n)]
    b_dense = [[0] * m for _ in range(n)]
    for coeff in zkey.coefficients:
        row, col, val = coeff.constraint, coeff.signal, coeff.value
        if coeff.matrix == 0:
            a_dense[row][col] = (a_dense[row][col] + val) % modulus
        else:
            b_dense[row][col] = (b_dense[row][col] + val) % modulus

    # Dense matrix-vector multiply mod p
    z = [int(w) for w in wtns.witnesses]
    az_vals = [sum(a_dense[i][j] * z[j] for j in range(m)) % modulus for i in range(n)]
    bz_vals = [sum(b_dense[i][j] * z[j] for j in range(m)) % modulus for i in range(n)]

    return (
        jnp.array(az_vals, dtype=bn254_sf_mont),
        jnp.array(bz_vals, dtype=bn254_sf_mont),
    )
