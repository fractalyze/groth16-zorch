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

"""Section parsing for circom binary files."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING, Callable

if TYPE_CHECKING:
    from .buffer import ReadOnlyBuffer


@dataclass
class Section:
    """Represents a section in a circom binary file."""

    section_type: int
    offset: int
    size: int


class Sections:
    """Parser for sections in circom binary files."""

    def __init__(
        self,
        buffer: ReadOnlyBuffer,
        type_to_string: Callable[[int], str] | None = None,
    ):
        self.buffer = buffer
        self.type_to_string = type_to_string or (lambda x: str(x))
        self.sections: list[Section] = []

    def read(self) -> None:
        """Read all section headers from the buffer."""
        num_sections = self.buffer.read_uint32()
        self.sections = []
        for _ in range(num_sections):
            self._add_section()

    def _add_section(self) -> None:
        """Add a single section header."""
        section_type = self.buffer.read_uint32()
        size = self.buffer.read_uint64()
        offset = self.buffer.offset
        self.sections.append(Section(section_type, offset, size))
        self.buffer.set_offset(offset + size)

    def move_to(self, section_type: int | IntEnum) -> None:
        """Move the buffer to the start of a section by type.

        Args:
            section_type: The type of section to move to.

        Raises:
            ValueError: If the section type is not found.
        """
        type_value = (
            section_type.value if isinstance(section_type, IntEnum) else section_type
        )
        for section in self.sections:
            if section.section_type == type_value:
                self.buffer.set_offset(section.offset)
                return
        raise ValueError(f"{self.type_to_string(type_value)} section is empty")

    def get_section(self, section_type: int | IntEnum) -> Section | None:
        """Get a section by type."""
        type_value = (
            section_type.value if isinstance(section_type, IntEnum) else section_type
        )
        for section in self.sections:
            if section.section_type == type_value:
                return section
        return None
