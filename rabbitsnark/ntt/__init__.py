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
    >>> ntt = NTT(bn254_sf_mont, BN254_FR_ROOT_OF_UNITY)
    >>> fwd_tw, inv_tw, inv_n = ntt.get_stage_twiddles(log_n)
    >>> evals = _forward_ntt(coeffs, log_n, *fwd_tw)
    >>> coeffs = _inverse_ntt(evals, inv_n, log_n, *inv_tw)
"""

from .ntt import NTT, _forward_ntt, _inverse_ntt

# Primitive 2²⁸-th root of unity in BN254 Fr (standard form)
# Computed as: 7^((p - 1) / 2²⁸) mod p
BN254_FR_ROOT_OF_UNITY = (
    1748695177688661943023146337482803886740723238769601073607632802312037301404
)

__all__ = [
    "NTT",
    "BN254_FR_ROOT_OF_UNITY",
    "_forward_ntt",
    "_inverse_ntt",
]
