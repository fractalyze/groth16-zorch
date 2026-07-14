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

"""Read-only buffer for binary file parsing using memory-mapped files.

This module provides efficient file reading using mmap, similar to the C++
RabbitSNARK implementation which uses tsl::ReadOnlyMemoryRegion with
madvise(MADV_SEQUENTIAL) for optimized sequential reads.
"""

from __future__ import annotations

import mmap
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import BinaryIO


@dataclass
class ReadOnlyBuffer:
    """A read-only buffer for parsing binary data with little-endian byte order.

    Uses memory-mapped files for efficient I/O, especially for large files
    like zkey proving keys.
    """

    data: mmap.mmap | bytes
    offset: int = field(default=0)
    _file: BinaryIO | None = field(default=None, repr=False)

    @classmethod
    def from_file(cls, path: str | Path) -> ReadOnlyBuffer:
        """Create a buffer from a file path using memory mapping.

        The file is memory-mapped for efficient access, similar to the C++
        implementation using tsl::ReadOnlyMemoryRegion.

        Args:
            path: Path to the file to read.

        Returns:
            A ReadOnlyBuffer backed by a memory-mapped file.
        """
        path = Path(path)
        file_size = path.stat().st_size

        if file_size == 0:
            # mmap doesn't support zero-length files
            return cls(b"")

        f = open(path, "rb")
        try:
            # Create read-only memory map
            # ACCESS_READ is equivalent to PROT_READ in C
            mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)

            # On Unix, provide sequential access hint like madvise(MADV_SEQUENTIAL)
            if hasattr(mmap, "MADV_SEQUENTIAL"):
                mm.madvise(mmap.MADV_SEQUENTIAL)

            return cls(data=mm, _file=f)
        except Exception:
            f.close()
            raise

    @classmethod
    def from_bytes(cls, data: bytes) -> ReadOnlyBuffer:
        """Create a buffer from raw bytes (for testing)."""
        return cls(data=data)

    def close(self) -> None:
        """Close the memory map and underlying file."""
        if isinstance(self.data, mmap.mmap):
            self.data.close()
        if self._file is not None:
            self._file.close()
            self._file = None

    def __del__(self) -> None:
        """Ensure resources are cleaned up."""
        self.close()

    def read_bytes(self, size: int) -> bytes:
        """Read raw bytes from the buffer."""
        if self.offset + size > len(self.data):
            raise ValueError(
                f"Buffer overflow: trying to read {size} bytes at offset "
                f"{self.offset}, but buffer size is {len(self.data)}"
            )
        result = self.data[self.offset : self.offset + size]
        self.offset += size
        return bytes(result)

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
        return bytes(self.data[self.offset : self.offset + size])

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

    def get_slice(self, start: int, end: int) -> memoryview:
        """Get a memory view of a slice without copying.

        This is useful for passing data directly to numpy without copying,
        similar to how the C++ implementation uses ReadPtr().

        Args:
            start: Start offset (absolute).
            end: End offset (absolute).

        Returns:
            A memoryview of the requested slice.
        """
        if end > len(self.data):
            raise ValueError(
                f"Buffer overflow: slice end {end} exceeds buffer size {len(self.data)}"
            )
        if isinstance(self.data, mmap.mmap):
            return memoryview(self.data)[start:end]
        return memoryview(self.data)[start:end]
