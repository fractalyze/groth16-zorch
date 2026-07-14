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

"""Witness (wtns) file parser using zk_dtypes."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from zk_dtypes import bn254_sf

from ..base.buffer import ReadOnlyBuffer
from ..base.modulus import Modulus
from ..base.sections import Sections

if TYPE_CHECKING:
    pass

WTNS_MAGIC = b"wtns"


class WtnsSectionType(IntEnum):
    """Section types in a wtns file."""

    HEADER = 0x1
    DATA = 0x2


def wtns_section_type_to_string(section_type: int) -> str:
    """Convert a section type to its string representation."""
    mapping = {
        WtnsSectionType.HEADER: "Header",
        WtnsSectionType.DATA: "Data",
    }
    return mapping.get(WtnsSectionType(section_type), f"Unknown({section_type})")


@dataclass
class WtnsHeaderSection:
    """Header section of a wtns file."""

    modulus: Modulus
    num_witness: int

    @classmethod
    def read(cls, buffer: ReadOnlyBuffer) -> WtnsHeaderSection:
        """Read the header section from the buffer."""
        modulus = Modulus.read(buffer)
        num_witness = buffer.read_uint32()
        return cls(modulus, num_witness)


@dataclass
class WtnsDataSection:
    """Data section of a wtns file containing witness values.

    Witness values are stored as a numpy array with bn254_sf dtype.
    The values are in standard (non-Montgomery) form.
    """

    _witnesses: np.ndarray  # dtype=bn254_sf

    @property
    def witnesses(self) -> list[int]:
        """Return witness values as a list of integers."""
        return [int(w) for w in self._witnesses]

    @classmethod
    def read(cls, buffer: ReadOnlyBuffer, header: WtnsHeaderSection) -> WtnsDataSection:
        """Read the data section from the buffer.

        Args:
            buffer: The buffer to read from.
            header: The header section containing field size information.
        """
        field_size = len(header.modulus.bytes_data)
        # Read all witness values at once using numpy
        raw_bytes = buffer.read_bytes(field_size * header.num_witness)
        witnesses = np.frombuffer(raw_bytes, dtype=bn254_sf)
        return cls(witnesses)


class Wtns(ABC):
    """Abstract base class for witness files."""

    @property
    @abstractmethod
    def version(self) -> int:
        """Return the version of the wtns file."""
        pass

    @property
    @abstractmethod
    def num_witness(self) -> int:
        """Return the number of witnesses."""
        pass

    @property
    @abstractmethod
    def witnesses(self) -> list[int]:
        """Return the witness values."""
        pass


@dataclass
class WtnsV2(Wtns):
    """Version 2 witness file."""

    header: WtnsHeaderSection
    data: WtnsDataSection

    @property
    def version(self) -> int:
        return 2

    @property
    def num_witness(self) -> int:
        return self.header.num_witness

    @property
    def witnesses(self) -> list[int]:
        return self.data.witnesses

    @classmethod
    def read(cls, buffer: ReadOnlyBuffer) -> WtnsV2:
        """Read a v2 wtns file from the buffer."""
        sections = Sections(buffer, wtns_section_type_to_string)
        sections.read()

        sections.move_to(WtnsSectionType.HEADER)
        header = WtnsHeaderSection.read(buffer)

        sections.move_to(WtnsSectionType.DATA)
        data = WtnsDataSection.read(buffer, header)

        return cls(header, data)


def parse_wtns(path: str | Path) -> Wtns:
    """Parse a wtns file from the given path.

    Args:
        path: Path to the wtns file.

    Returns:
        A Wtns object containing the parsed data.

    Raises:
        ValueError: If the file has invalid magic or unsupported version.
    """
    buffer = ReadOnlyBuffer.from_file(path)

    magic = buffer.read_bytes(4)
    if magic != WTNS_MAGIC:
        raise ValueError(f"Invalid magic: expected {WTNS_MAGIC!r}, got {magic!r}")

    version = buffer.read_uint32()
    if version == 2:
        return WtnsV2.read(buffer)
    else:
        raise ValueError(f"Unsupported version: expected 2, got {version}")
