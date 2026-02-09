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

"""Coefficient representation for R1CS constraints.

Note: Coefficients in zkey files are stored in double Montgomery form,
which requires special handling that zk_dtypes doesn't provide directly.
We use manual Montgomery conversion for coefficient values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..base.buffer import ReadOnlyBuffer


@dataclass
class Coefficient:
    """Represents a coefficient in R1CS constraints.

    R1CS is represented as A * z . B * z = C * z, where . is the Hadamard product.
    Each constraint is composed of:
    - [a_i,0, ..., a_i,m-1] * [z_0, ..., z_m-1]
    - [b_i,0, ..., b_i,m-1] * [z_0, ..., z_m-1]
    - [c_i,0, ..., c_i,m-1] * [z_0, ..., z_m-1]

    where i is the index of the constraints (0 <= i < n),
    m is the number of QAP variables, and n is the number of constraints.
    """

    matrix: int  # 0 for matrix A, non-zero for matrix B
    constraint: int  # The index of the constraint (0 <= i < n)
    signal: int  # The index of the QAP variables (0 <= j < m)
    value: int  # The coefficient value (in standard form)

    @classmethod
    def read(
        cls, buffer: ReadOnlyBuffer, field_size: int, modulus: int
    ) -> Coefficient:
        """Read a coefficient from the buffer.

        Args:
            buffer: The buffer to read from.
            field_size: Size of the field element in bytes.
            modulus: The scalar field modulus for Montgomery conversion.
        """
        matrix = buffer.read_uint32()
        constraint = buffer.read_uint32()
        signal = buffer.read_uint32()
        # Coefficient values are stored in double Montgomery form in zkey files
        value_double_mont = buffer.read_field_element(field_size)
        value = from_montgomery(from_montgomery(value_double_mont, modulus), modulus)
        return cls(matrix, constraint, signal, value)

    @classmethod
    def from_ints(
        cls, matrix: int, constraint: int, signal: int, value: int
    ) -> Coefficient:
        """Create a Coefficient from integer values (for testing)."""
        return cls(matrix, constraint, signal, value)

    def is_matrix_a(self) -> bool:
        """Return True if this coefficient is for matrix A."""
        return self.matrix == 0

    def is_matrix_b(self) -> bool:
        """Return True if this coefficient is for matrix B."""
        return self.matrix != 0
