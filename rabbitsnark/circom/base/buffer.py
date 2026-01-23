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

"""Read-only buffer for binary file parsing."""

from __future__ import annotations

import struct
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class ReadOnlyBuffer:
    """A read-only buffer for parsing binary data with little-endian byte order."""

    data: bytes
    offset: int = field(default=0)

    @classmethod
    def from_file(cls, path: str | Path) -> ReadOnlyBuffer:
        """Create a buffer from a file path."""
        with open(path, "rb") as f:
            return cls(f.read())

    def read_bytes(self, size: int) -> bytes:
        """Read raw bytes from the buffer."""
        if self.offset + size > len(self.data):
            raise ValueError(
                f"Buffer overflow: trying to read {size} bytes at offset "
                f"{self.offset}, but buffer size is {len(self.data)}"
            )
        result = self.data[self.offset : self.offset + size]
        self.offset += size
        return result

    def read_uint32(self) -> int:
        """Read a 32-bit unsigned integer (little-endian)."""
        return struct.unpack("<I", self.read_bytes(4))[0]

    def read_uint64(self) -> int:
        """Read a 64-bit unsigned integer (little-endian)."""
        return struct.unpack("<Q", self.read_bytes(8))[0]

    def read_field_element(self, size: int) -> int:
        """Read a field element as a little-endian integer."""
        data = self.read_bytes(size)
        return int.from_bytes(data, byteorder="little")

    def peek_bytes(self, size: int) -> bytes:
        """Peek at bytes without advancing the offset."""
        if self.offset + size > len(self.data):
            raise ValueError(
                f"Buffer overflow: trying to peek {size} bytes at offset "
                f"{self.offset}, but buffer size is {len(self.data)}"
            )
        return self.data[self.offset : self.offset + size]

    def set_offset(self, offset: int) -> None:
        """Set the buffer offset to a specific position."""
        if offset > len(self.data):
            raise ValueError(
                f"Invalid offset {offset}, buffer size is {len(self.data)}"
            )
        self.offset = offset

    def remaining(self) -> int:
        """Return the number of bytes remaining in the buffer."""
        return len(self.data) - self.offset
