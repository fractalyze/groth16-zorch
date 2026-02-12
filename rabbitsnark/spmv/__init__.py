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
vectorized SpMV for Groth16 proving (Az = A * z, Bz = B * z).

Example usage:
    >>> from rabbitsnark.spmv import build_r1cs_matrices, spmv
    >>> A, B = build_r1cs_matrices(zkey, bn254_sf_mont)
    >>> Az = spmv(A, z_mont)
    >>> Bz = spmv(B, z_mont)
"""

from .backend import spmv_backend
from .csr_matrix import CSRMatrix
from .r1cs import build_r1cs_matrices, witness_to_montgomery
from .spmv import spmv

__all__ = [
    "CSRMatrix",
    "build_r1cs_matrices",
    "spmv",
    "spmv_backend",
    "witness_to_montgomery",
]
