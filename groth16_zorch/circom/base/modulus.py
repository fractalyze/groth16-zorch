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

"""Modulus representation for finite field parameters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .buffer import ReadOnlyBuffer


@dataclass
class Modulus:
    """Represents a modulus value for a finite field."""

    bytes_data: bytes

    @classmethod
    def read(cls, buffer: ReadOnlyBuffer) -> Modulus:
        """Read a modulus from the buffer.

        Format: num_bytes (uint32) + bytes (little-endian)
        """
        num_bytes = buffer.read_uint32()
        if num_bytes % 8 != 0:
            raise ValueError("Field size is not a multiple of 8")
        bytes_data = buffer.read_bytes(num_bytes)
        return cls(bytes_data)

    def to_int(self) -> int:
        """Convert the modulus to an integer."""
        return int.from_bytes(self.bytes_data, byteorder="little")

    @classmethod
    def from_int(cls, value: int, num_bytes: int = 32) -> Modulus:
        """Create a modulus from an integer."""
        bytes_data = value.to_bytes(num_bytes, byteorder="little")
        return cls(bytes_data)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Modulus):
            return NotImplemented
        return self.bytes_data == other.bytes_data

    def __repr__(self) -> str:
        return f"Modulus({self.to_int()})"
