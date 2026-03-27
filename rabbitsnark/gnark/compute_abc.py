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

"""Gnark binary format loaders for R1CS solver data.

Reads CSR matrices, level/unknown info, hints, and coefficients from the
gnark binary export directory format.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from rabbitsnark.r1cs_solver import CSRMatrices, SolverData

from .loader import FIELD_ELEM_SIZE

_UNKNOWN_ENTRY_SIZE = 5  # 1B side + 4B wire_id


# ---------------------------------------------------------------------------
# File loading
# ---------------------------------------------------------------------------


def load_csr_matrices(export_dir: str | Path) -> CSRMatrices:
    """Load CSR matrices from binary files in the export directory.

    Expected files per matrix (a, b, c):
        r1cs_{a,b,c}_indptr.bin  — int64
        r1cs_{a,b,c}_indices.bin — int32
        r1cs_{a,b,c}_values.bin  — raw 32-byte field elements
    """
    d = Path(export_dir)

    def _load_csr(prefix: str):
        indptr = np.fromfile(str(d / f"r1cs_{prefix}_indptr.bin"), dtype=np.int64)
        indices = np.fromfile(str(d / f"r1cs_{prefix}_indices.bin"), dtype=np.int32)
        raw = np.fromfile(str(d / f"r1cs_{prefix}_values.bin"), dtype=np.uint8)
        nnz = raw.size // FIELD_ELEM_SIZE
        values = (
            raw.reshape(nnz, FIELD_ELEM_SIZE)
            if nnz > 0
            else raw.reshape(0, FIELD_ELEM_SIZE)
        )
        return indptr, indices, values

    a_indptr, a_indices, a_values = _load_csr("a")
    b_indptr, b_indices, b_values = _load_csr("b")
    c_indptr, c_indices, c_values = _load_csr("c")

    return CSRMatrices(
        a_indptr=a_indptr,
        a_indices=a_indices,
        a_values=a_values,
        b_indptr=b_indptr,
        b_indices=b_indices,
        b_values=b_values,
        c_indptr=c_indptr,
        c_indices=c_indices,
        c_values=c_values,
    )


def _parse_unknowns(
    path: Path,
    num_constraints: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Parse per-constraint unknown info: (sides uint8, wire_ids int32)."""
    raw = np.fromfile(str(path), dtype=np.uint8)
    assert raw.size == num_constraints * _UNKNOWN_ENTRY_SIZE
    data = raw.reshape(num_constraints, _UNKNOWN_ENTRY_SIZE)
    sides = data[:, 0].copy()
    wire_ids = np.ascontiguousarray(data[:, 1:5]).view(np.int32).flatten()
    return sides, wire_ids


def _parse_coefficients(path: Path) -> np.ndarray:
    """Parse coefficient table: uint32 count header + BN254 field elements."""
    raw = np.fromfile(str(path), dtype=np.uint8)
    count = int(np.frombuffer(raw[:4], dtype=np.uint32)[0])
    if count == 0:
        return np.zeros((0, FIELD_ELEM_SIZE), dtype=np.uint8)
    return raw[4:].reshape(count, FIELD_ELEM_SIZE)


def _parse_hints(path: Path) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    int,
]:
    """Parse and flatten hint binary data.

    Returns:
        (hint_ids, output_starts, num_outputs, num_inputs,
         le_offsets, term_coeff_ids, term_wire_ids,
         input_le_offsets, max_inputs)
    """
    raw = np.fromfile(str(path), dtype=np.uint8)
    if raw.size == 0:
        # No hints — return minimal arrays
        return (
            np.array([0], dtype=np.int32),  # hint_ids
            np.array([0], dtype=np.int32),  # output_starts
            np.array([0], dtype=np.int32),  # num_outputs
            np.array([0], dtype=np.int32),  # num_inputs
            np.array([0], dtype=np.int64),  # le_offsets (sentinel only at call site)
            np.array([0], dtype=np.int32),  # term_coeff_ids
            np.array([0], dtype=np.int32),  # term_wire_ids
            np.array(
                [0], dtype=np.int64
            ),  # input_le_offsets (sentinel only at call site)
            1,  # max_inputs (minimum 1 for scratch buffer)
        )

    # Parse variable-length hint records
    offset = 0
    hint_ids = []
    output_starts = []
    num_outputs_list = []
    num_inputs_list = []
    le_offsets = []
    term_coeff_ids = []
    term_wire_ids = []
    input_le_offsets = [0]
    le_offset = 0
    term_offset = 0
    max_inputs = 0

    buf = raw.tobytes()
    while offset < len(buf):
        hint_id = int.from_bytes(buf[offset : offset + 4], "little")
        _level_idx = int.from_bytes(buf[offset + 4 : offset + 8], "little")
        n_inputs = int.from_bytes(buf[offset + 8 : offset + 12], "little")
        n_outputs = int.from_bytes(buf[offset + 12 : offset + 16], "little")
        out_start = int.from_bytes(buf[offset + 16 : offset + 20], "little")
        offset += 20

        hint_ids.append(hint_id)
        output_starts.append(out_start)
        num_outputs_list.append(n_outputs)
        num_inputs_list.append(n_inputs)
        le_offsets.append(le_offset)

        if n_inputs > max_inputs:
            max_inputs = n_inputs

        for _ in range(n_inputs):
            num_terms = int.from_bytes(buf[offset : offset + 4], "little")
            offset += 4
            for _ in range(num_terms):
                coeff_id = int.from_bytes(buf[offset : offset + 4], "little")
                wire_id = int.from_bytes(buf[offset + 4 : offset + 8], "little")
                term_coeff_ids.append(coeff_id)
                term_wire_ids.append(wire_id)
                offset += 8
                term_offset += 1
            input_le_offsets.append(term_offset)
            le_offset += 1

    # Sentinel for last hint
    le_offsets.append(le_offset)

    # Ensure non-empty arrays
    if not term_coeff_ids:
        term_coeff_ids.append(0)
        term_wire_ids.append(0)
    if max_inputs == 0:
        max_inputs = 1

    return (
        np.array(hint_ids, dtype=np.uint32),
        np.array(output_starts, dtype=np.int32),
        np.array(num_outputs_list, dtype=np.int32),
        np.array(num_inputs_list, dtype=np.int32),
        np.array(le_offsets, dtype=np.int64),
        np.array(term_coeff_ids, dtype=np.int32),
        np.array(term_wire_ids, dtype=np.int32),
        np.array(input_le_offsets, dtype=np.int64),
        max_inputs,
    )


def load_solver_data(export_dir: str | Path) -> SolverData:
    """Load all data needed for native R1CS solving from the export directory.

    This includes CSR matrices, level/unknown info, hints, and coefficients.
    """
    d = Path(export_dir)

    with open(d / "metadata.json") as f:
        meta = json.load(f)

    num_constraints = meta["num_constraints"]
    num_levels = meta["num_levels"]

    csr = load_csr_matrices(d)

    unk_side, target_idx = _parse_unknowns(
        d / "r1cs_level_unknowns.bin",
        num_constraints,
    )
    unk_inv_coeff_raw = np.fromfile(
        str(d / "r1cs_unknown_inv_coeffs.bin"),
        dtype=np.uint8,
    )
    unk_inv_coeff = unk_inv_coeff_raw.reshape(num_constraints, FIELD_ELEM_SIZE)

    level_sizes_u32 = np.fromfile(str(d / "r1cs_level_sizes.bin"), dtype=np.uint32)
    level_order_u32 = np.fromfile(str(d / "r1cs_level_order.bin"), dtype=np.uint32)
    level_sizes = level_sizes_u32.astype(np.int32)
    level_cids = level_order_u32.astype(np.int32)

    # Build level_offsets (exclusive prefix sum of level_sizes)
    level_offsets = np.zeros(num_levels, dtype=np.int32)
    if num_levels > 1:
        level_offsets[1:] = np.cumsum(level_sizes[:-1])

    (
        hint_ids,
        hint_output_starts,
        hint_num_outputs,
        hint_num_inputs,
        hint_le_offsets,
        hint_term_coeff_ids,
        hint_term_wire_ids,
        hint_input_le_offsets,
        max_hint_inputs,
    ) = _parse_hints(d / "r1cs_hints.bin")

    hint_level_offsets_u32 = np.fromfile(
        str(d / "r1cs_hint_level_offsets.bin"),
        dtype=np.uint32,
    )
    hint_level_offsets = hint_level_offsets_u32.astype(np.int32)

    coefficients = _parse_coefficients(d / "r1cs_coefficients.bin")

    return SolverData(
        csr=csr,
        unk_side=unk_side,
        unk_inv_coeff=unk_inv_coeff,
        target_idx=target_idx,
        level_cids=level_cids,
        level_offsets=level_offsets,
        level_sizes=level_sizes,
        hint_ids=hint_ids,
        hint_output_starts=hint_output_starts,
        hint_num_outputs=hint_num_outputs,
        hint_num_inputs=hint_num_inputs,
        hint_le_offsets=hint_le_offsets,
        hint_term_coeff_ids=hint_term_coeff_ids,
        hint_term_wire_ids=hint_term_wire_ids,
        hint_input_le_offsets=hint_input_le_offsets,
        hint_level_offsets=hint_level_offsets,
        coefficients=coefficients,
        max_hint_inputs=max_hint_inputs,
        num_levels=num_levels,
        num_wires=meta["num_wires"],
        num_public=meta["num_public"],
        num_secret=meta["num_secret"],
        num_constraints=num_constraints,
        domain_size=meta["domain_size"],
    )
