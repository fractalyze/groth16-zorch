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

"""ZKey file parser for Groth16 proving keys."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import TYPE_CHECKING

from ..base.buffer import ReadOnlyBuffer
from ..base.modulus import Modulus
from ..base.sections import Sections
from .coefficient import Coefficient
from .verifying_key import G1Point, G2Point, VerifyingKey

if TYPE_CHECKING:
    pass

ZKEY_MAGIC = b"zkey"


class ZKeySectionType(IntEnum):
    """Section types in a zkey file."""

    HEADER = 0x1
    HEADER_GROTH = 0x2
    IC = 0x3
    COEFFICIENTS = 0x4
    POINTS_A1 = 0x5
    POINTS_B1 = 0x6
    POINTS_B2 = 0x7
    POINTS_C1 = 0x8
    POINTS_H1 = 0x9
    CONTRIBUTION = 0xA


def zkey_section_type_to_string(section_type: int) -> str:
    """Convert a section type to its string representation."""
    mapping = {
        ZKeySectionType.HEADER: "Header",
        ZKeySectionType.HEADER_GROTH: "HeaderGroth",
        ZKeySectionType.IC: "IC",
        ZKeySectionType.COEFFICIENTS: "Coefficients",
        ZKeySectionType.POINTS_A1: "PointsA1",
        ZKeySectionType.POINTS_B1: "PointsB1",
        ZKeySectionType.POINTS_B2: "PointsB2",
        ZKeySectionType.POINTS_C1: "PointsC1",
        ZKeySectionType.POINTS_H1: "PointsH1",
        ZKeySectionType.CONTRIBUTION: "Contribution",
    }
    return mapping.get(ZKeySectionType(section_type), f"Unknown({section_type})")


@dataclass
class ZKeyHeaderSection:
    """Header section of a zkey file."""

    prover_type: int

    @classmethod
    def read(cls, buffer: ReadOnlyBuffer) -> ZKeyHeaderSection:
        """Read the header section from the buffer."""
        prover_type = buffer.read_uint32()
        if prover_type != 1:
            raise ValueError(f"Unsupported prover type: {prover_type}")
        return cls(prover_type)


@dataclass
class ZKeyHeaderGrothSection:
    """Groth16-specific header section."""

    q: Modulus  # Base field modulus
    r: Modulus  # Scalar field modulus
    num_vars: int
    num_public_inputs: int
    domain_size: int
    vkey: VerifyingKey

    @classmethod
    def read(cls, buffer: ReadOnlyBuffer) -> ZKeyHeaderGrothSection:
        """Read the Groth16 header section from the buffer."""
        q = Modulus.read(buffer)
        r = Modulus.read(buffer)
        num_vars = buffer.read_uint32()
        num_public_inputs = buffer.read_uint32()
        domain_size = buffer.read_uint32()

        # Field size for G1/G2 points is the base field size
        field_size = len(q.bytes_data)
        base_field_modulus = q.to_int()
        vkey = VerifyingKey.read(buffer, field_size, base_field_modulus)

        return cls(q, r, num_vars, num_public_inputs, domain_size, vkey)


class ZKey(ABC):
    """Abstract base class for zkey files."""

    @property
    @abstractmethod
    def version(self) -> int:
        """Return the version of the zkey file."""
        pass

    @property
    @abstractmethod
    def domain_size(self) -> int:
        """Return the domain size."""
        pass

    @property
    @abstractmethod
    def num_instance_variables(self) -> int:
        """Return the number of instance (public) variables."""
        pass

    @property
    @abstractmethod
    def num_witness_variables(self) -> int:
        """Return the number of witness (private) variables."""
        pass

    @property
    @abstractmethod
    def coefficients(self) -> list[Coefficient]:
        """Return the R1CS coefficients."""
        pass


@dataclass
class ZKeyV1(ZKey):
    """Version 1 zkey file (Groth16)."""

    header: ZKeyHeaderSection
    header_groth: ZKeyHeaderGrothSection
    ic: list[G1Point]
    _coefficients: list[Coefficient]
    points_a1: list[G1Point]
    points_b1: list[G1Point]
    points_b2: list[G2Point]
    points_c1: list[G1Point]
    points_h1: list[G1Point]

    @property
    def version(self) -> int:
        return 1

    @property
    def domain_size(self) -> int:
        return self.header_groth.domain_size

    @property
    def num_instance_variables(self) -> int:
        return self.header_groth.num_public_inputs + 1

    @property
    def num_witness_variables(self) -> int:
        return self.header_groth.num_vars - self.header_groth.num_public_inputs - 1

    @property
    def coefficients(self) -> list[Coefficient]:
        return self._coefficients

    @property
    def verifying_key(self) -> VerifyingKey:
        """Return the verifying key."""
        return self.header_groth.vkey

    @classmethod
    def read(cls, buffer: ReadOnlyBuffer) -> ZKeyV1:
        """Read a v1 zkey file from the buffer.

        Args:
            buffer: The buffer to read from.
        """
        sections = Sections(buffer, zkey_section_type_to_string)
        sections.read()

        # Read header
        sections.move_to(ZKeySectionType.HEADER)
        header = ZKeyHeaderSection.read(buffer)

        # Read Groth16 header
        sections.move_to(ZKeySectionType.HEADER_GROTH)
        header_groth = ZKeyHeaderGrothSection.read(buffer)

        num_vars = header_groth.num_vars
        num_public_inputs = header_groth.num_public_inputs
        domain_size = header_groth.domain_size

        # Field sizes and moduli
        base_field_size = len(header_groth.q.bytes_data)
        scalar_field_size = len(header_groth.r.bytes_data)
        base_field_modulus = header_groth.q.to_int()

        # Read IC section
        sections.move_to(ZKeySectionType.IC)
        ic = cls._read_g1_points(
            buffer, num_public_inputs + 1, base_field_size, base_field_modulus
        )

        # Read coefficients section
        sections.move_to(ZKeySectionType.COEFFICIENTS)
        num_coefficients = buffer.read_uint32()
        coefficients = [
            Coefficient.read(buffer, scalar_field_size) for _ in range(num_coefficients)
        ]

        # Read points sections
        sections.move_to(ZKeySectionType.POINTS_A1)
        points_a1 = cls._read_g1_points(
            buffer, num_vars, base_field_size, base_field_modulus
        )

        sections.move_to(ZKeySectionType.POINTS_B1)
        points_b1 = cls._read_g1_points(
            buffer, num_vars, base_field_size, base_field_modulus
        )

        sections.move_to(ZKeySectionType.POINTS_B2)
        points_b2 = cls._read_g2_points(
            buffer, num_vars, base_field_size, base_field_modulus
        )

        sections.move_to(ZKeySectionType.POINTS_C1)
        points_c1 = cls._read_g1_points(
            buffer,
            num_vars - num_public_inputs - 1,
            base_field_size,
            base_field_modulus,
        )

        sections.move_to(ZKeySectionType.POINTS_H1)
        points_h1 = cls._read_g1_points(
            buffer, domain_size, base_field_size, base_field_modulus
        )

        return cls(
            header=header,
            header_groth=header_groth,
            ic=ic,
            _coefficients=coefficients,
            points_a1=points_a1,
            points_b1=points_b1,
            points_b2=points_b2,
            points_c1=points_c1,
            points_h1=points_h1,
        )

    @staticmethod
    def _read_g1_points(
        buffer: ReadOnlyBuffer, count: int, field_size: int, modulus: int
    ) -> list[G1Point]:
        """Read a list of G1 points from the buffer."""
        return [G1Point.read(buffer, field_size, modulus) for _ in range(count)]

    @staticmethod
    def _read_g2_points(
        buffer: ReadOnlyBuffer, count: int, field_size: int, modulus: int
    ) -> list[G2Point]:
        """Read a list of G2 points from the buffer."""
        return [G2Point.read(buffer, field_size, modulus) for _ in range(count)]


def parse_zkey(path: str | Path) -> ZKey:
    """Parse a zkey file from the given path.

    Args:
        path: Path to the zkey file.

    Returns:
        A ZKey object containing the parsed data.

    Raises:
        ValueError: If the file has invalid magic or unsupported version.
    """
    buffer = ReadOnlyBuffer.from_file(path)

    magic = buffer.read_bytes(4)
    if magic != ZKEY_MAGIC:
        raise ValueError(f"Invalid magic: expected {ZKEY_MAGIC!r}, got {magic!r}")

    version = buffer.read_uint32()
    if version == 1:
        return ZKeyV1.read(buffer)
    else:
        raise ValueError(f"Unsupported version: expected 1, got {version}")
