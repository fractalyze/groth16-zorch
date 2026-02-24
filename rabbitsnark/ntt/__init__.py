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

"""NTT (Number Theoretic Transform) implementations in JAX.

This module provides an efficient NTT implementation for prime fields
used in zero-knowledge proofs.

Example usage:
    >>> from rabbitsnark.ntt import NTT, BN254_FR_ROOT_OF_UNITY
    >>> import jax.numpy as jnp
    >>> from zk_dtypes import bn254_sf_mont
    >>>
    >>> ntt = NTT(bn254_sf_mont, BN254_FR_ROOT_OF_UNITY)
    >>> coeffs = jnp.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=bn254_sf_mont)
    >>> evals = ntt.forward(coeffs)  # Forward NTT
    >>> recovered = ntt.inverse(evals)  # Inverse NTT
"""

from .ntt import NTT, _forward_ntt, _inverse_ntt, batch_ntt

# Primitive 2²⁸-th root of unity in BN254 Fr (standard form)
# Computed as: 7^((p - 1) / 2²⁸) mod p
BN254_FR_ROOT_OF_UNITY = (
    1748695177688661943023146337482803886740723238769601073607632802312037301404
)

__all__ = [
    "NTT",
    "BN254_FR_ROOT_OF_UNITY",
    "batch_ntt",
    "_forward_ntt",
    "_inverse_ntt",
]
