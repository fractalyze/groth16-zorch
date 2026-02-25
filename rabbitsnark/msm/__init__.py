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

"""MSM (Multi-Scalar Multiplication) utilities for Groth16 proving.

Provides the ``MSM`` class with static methods for Pippenger's bucket method
used inside the JIT-compiled prove kernel.

EC point dtypes (e.g. ``bn254_g1_xyzz``) are atomic JAX dtypes.  The ``+``
and ``*`` operators on these types automatically lower to ``ec.add`` /
``ec.double`` / ``ec.scalar_mul``, so the algorithm contains no manual point
arithmetic.
"""

from .msm import MSM

__all__ = [
    "MSM",
]
