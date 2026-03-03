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

Provides CSR sparse matrix construction from R1CS constraints and
SELL-format SpMV for Groth16 proving (Az = A * z, Bz = B * z).

Example usage:
    >>> from rabbitsnark.spmv import build_r1cs_matrices, SELLMatrix, spmv_sell
    >>> A, B = build_r1cs_matrices(zkey, bn254_sf_mont)
    >>> sell_A = SELLMatrix.from_csr(A)
    >>> Az = spmv_sell(sell_A, z_mont)
"""

from .csr_matrix import CSRMatrix
from .r1cs import build_r1cs_matrices, witness_to_montgomery
from .sell import SELLConfig, SELLMatrix
from .spmv import spmv_sell

__all__ = [
    "CSRMatrix",
    "SELLConfig",
    "SELLMatrix",
    "build_r1cs_matrices",
    "spmv_sell",
    "witness_to_montgomery",
]
