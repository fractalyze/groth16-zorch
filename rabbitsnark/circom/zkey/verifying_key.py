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

"""Verifying key structures for Groth16 using zk_dtypes."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from zk_dtypes import bn254_g1_affine, bn254_g2_affine

if TYPE_CHECKING:
    from ..base.buffer import ReadOnlyBuffer


def _parse_g1_repr(s: str) -> tuple[int, int]:
    """Parse the string representation of a G1 point to extract x and y."""
    match = re.match(r"\((\d+), (\d+)\)", s)
    if match:
        return int(match.group(1)), int(match.group(2))
    raise ValueError(f"Invalid G1 point representation: {s}")


def _parse_g2_repr(s: str) -> tuple[tuple[int, int], tuple[int, int]]:
    """Parse the string representation of a G2 point to extract coordinates.

    Format is: ([x0,x1], [y0,y1])
    """
    match = re.match(r"\(\[(\d+),(\d+)\], \[(\d+),(\d+)\]\)", s)
    if match:
        x0, x1, y0, y1 = (int(match.group(i)) for i in range(1, 5))
        return (x0, x1), (y0, y1)
    raise ValueError(f"Invalid G2 point representation: {s}")


@dataclass
class G1Point:
    """A point on the G1 curve (affine coordinates) using zk_dtypes.

    The point is stored internally in Montgomery form using bn254_g1_affine dtype.
    """

    _data: np.ndarray  # dtype=bn254_g1_affine, shape=()

    @property
    def x(self) -> int:
        """Return the x coordinate as an integer."""
        x, _ = _parse_g1_repr(str(self._data))
        return x

    @property
    def y(self) -> int:
        """Return the y coordinate as an integer."""
        _, y = _parse_g1_repr(str(self._data))
        return y

    @classmethod
    def read(cls, buffer: ReadOnlyBuffer, field_size: int, modulus: int) -> G1Point:
        """Read a G1 point from the buffer.

        Args:
            buffer: The buffer to read from.
            field_size: Size of the field element in bytes (should be 32 for BN254).
            modulus: The base field modulus (unused, kept for API compatibility).
        """
        raw_bytes = buffer.read_bytes(field_size * 2)
        data = np.frombuffer(raw_bytes, dtype=bn254_g1_affine)[0]
        return cls(data)

    @classmethod
    def from_ints(cls, x: int, y: int) -> G1Point:
        """Create a G1Point from integer coordinates (for testing).

        The x and y values are stored directly without conversion, using the
        standard form dtype so that the string representation matches the input.
        """
        x_bytes = x.to_bytes(32, "little")
        y_bytes = y.to_bytes(32, "little")
        raw_bytes = x_bytes + y_bytes
        data = np.frombuffer(raw_bytes, dtype=bn254_g1_affine)[0]
        return cls(data)

    def is_zero(self) -> bool:
        """Return True if this is the point at infinity."""
        return self.x == 0 and self.y == 0

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, G1Point):
            return NotImplemented
        # Compare Montgomery form values (via string repr) since raw bytes differ
        return self.x == other.x and self.y == other.y

    def __hash__(self) -> int:
        return hash((self.x, self.y))

    def __repr__(self) -> str:
        return f"G1Point(x={self.x}, y={self.y})"


@dataclass
class G2Point:
    """
    A point on the G2 curve (affine coordinates with extension field) using zk_dtypes.

    The point is stored internally in Montgomery form using bn254_g2_affine dtype.
    """

    _data: np.ndarray  # dtype=bn254_g2_affine, shape=()

    @property
    def x(self) -> tuple[int, int]:
        """Return the x coordinate as a tuple (x0, x1) for Fq2."""
        (x0, x1), _ = _parse_g2_repr(str(self._data))
        return (x0, x1)

    @property
    def y(self) -> tuple[int, int]:
        """Return the y coordinate as a tuple (y0, y1) for Fq2."""
        _, (y0, y1) = _parse_g2_repr(str(self._data))
        return (y0, y1)

    @classmethod
    def read(cls, buffer: ReadOnlyBuffer, field_size: int, modulus: int) -> G2Point:
        """Read a G2 point from the buffer.

        Args:
            buffer: The buffer to read from.
            field_size: Size of the field element in bytes (should be 32 for BN254).
            modulus: The base field modulus (unused, kept for API compatibility).
        """
        raw_bytes = buffer.read_bytes(field_size * 4)
        data = np.frombuffer(raw_bytes, dtype=bn254_g2_affine)[0]
        return cls(data)

    @classmethod
    def from_ints(cls, x: tuple[int, int], y: tuple[int, int]) -> G2Point:
        """Create a G2Point from integer coordinates (for testing).

        The x and y values are stored directly without conversion, using the
        standard form dtype so that the string representation matches the input.
        """
        x0_bytes = x[0].to_bytes(32, "little")
        x1_bytes = x[1].to_bytes(32, "little")
        y0_bytes = y[0].to_bytes(32, "little")
        y1_bytes = y[1].to_bytes(32, "little")
        raw_bytes = x0_bytes + x1_bytes + y0_bytes + y1_bytes
        data = np.frombuffer(raw_bytes, dtype=bn254_g2_affine)[0]
        return cls(data)

    def is_zero(self) -> bool:
        """Return True if this is the point at infinity."""
        return self.x == (0, 0) and self.y == (0, 0)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, G2Point):
            return NotImplemented
        # Compare Montgomery form values (via string repr) since raw bytes differ
        return self.x == other.x and self.y == other.y

    def __hash__(self) -> int:
        return hash((self.x, self.y))

    def __repr__(self) -> str:
        return f"G2Point(x={self.x}, y={self.y})"


@dataclass
class VerifyingKey:
    """Groth16 verifying key."""

    alpha_g1: G1Point
    beta_g1: G1Point
    beta_g2: G2Point
    gamma_g2: G2Point
    delta_g1: G1Point
    delta_g2: G2Point

    @classmethod
    def read(
        cls, buffer: ReadOnlyBuffer, field_size: int, modulus: int
    ) -> VerifyingKey:
        """Read a verifying key from the buffer."""
        alpha_g1 = G1Point.read(buffer, field_size, modulus)
        beta_g1 = G1Point.read(buffer, field_size, modulus)
        beta_g2 = G2Point.read(buffer, field_size, modulus)
        gamma_g2 = G2Point.read(buffer, field_size, modulus)
        delta_g1 = G1Point.read(buffer, field_size, modulus)
        delta_g2 = G2Point.read(buffer, field_size, modulus)
        return cls(alpha_g1, beta_g1, beta_g2, gamma_g2, delta_g1, delta_g2)
