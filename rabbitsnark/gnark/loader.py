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

"""Load gnark Groth16 exported binary data into Python objects."""

from __future__ import annotations

import json
import struct
import time
from pathlib import Path

import numpy as np
from zk_dtypes import bn254_sf_mont

from .types import GnarkProvingData, HintData, HintInstruction

FIELD_ELEM_SIZE = 32  # bytes per BN254 scalar field element
G1_POINT_SIZE = 64  # 2 × 32B (X, Y)
G2_POINT_SIZE = 128  # 4 × 32B (X.A0, X.A1, Y.A0, Y.A1)
COO_ENTRY_SIZE = 40  # 4B row + 4B col + 32B val
UNKNOWN_ENTRY_SIZE = 5  # 1B side + 4B wire_id


def load_gnark_export(
    export_dir: str | Path,
    *,
    solver_only: bool = False,
) -> GnarkProvingData:
    """Load all exported gnark data from a directory.

    Args:
        export_dir: Path to the export directory produced by cmd/export.
        solver_only: If True, skip PK/VK points and solutions (solver only).

    Returns:
        GnarkProvingData with all circuit data loaded.
    """
    d = Path(export_dir)

    # Metadata
    with open(d / "metadata.json") as f:
        meta = json.load(f)

    num_wires = meta["num_wires"]
    num_public = meta["num_public"]
    num_secret = meta["num_secret"]
    num_internal = meta["num_internal"]
    num_constraints = meta["num_constraints"]
    domain_size = meta["domain_size"]

    # Witness — load directly as bn254_sf_mont (Montgomery form on disk)
    t = time.perf_counter()
    witness_full = _read_field_elements_native(d / "witness_full.bin", num_wires)
    print(f"  witness_full: {time.perf_counter() - t:.1f}s")

    if solver_only:
        solution_a = solution_b = solution_c = np.array([], dtype=object)
        pk_a_g1 = pk_b_g1 = pk_k_g1 = pk_z_g1 = []
        pk_b_g2 = []
        pk_delta_g1 = (0, 0)
        pk_delta_g2 = ((0, 0), (0, 0))
        infinity_a = infinity_b = np.array([], dtype=bool)
        vk_alpha_g1 = vk_beta_g1 = (0, 0)
        vk_beta_g2 = vk_gamma_g2 = ((0, 0), (0, 0))
        vk_ic = []
    else:
        t = time.perf_counter()
        solution_a = _read_field_elements(d / "solution_a.bin", num_constraints)
        solution_b = _read_field_elements(d / "solution_b.bin", num_constraints)
        solution_c = _read_field_elements(d / "solution_c.bin", num_constraints)
        print(f"  solutions: {time.perf_counter() - t:.1f}s")

        t = time.perf_counter()
        pk_a_g1 = _read_g1_points(d / "pk_a_g1.bin")
        pk_b_g1 = _read_g1_points(d / "pk_b_g1.bin")
        pk_b_g2 = _read_g2_points(d / "pk_b_g2.bin")
        pk_k_g1 = _read_g1_points(d / "pk_k_g1.bin")
        pk_z_g1 = _read_g1_points(d / "pk_z_g1.bin")
        pk_delta_g1 = _read_g1_points(d / "pk_delta_g1.bin")[0]
        pk_delta_g2 = _read_g2_points(d / "pk_delta_g2.bin")[0]
        print(f"  PK points: {time.perf_counter() - t:.1f}s")

        infinity_a = np.fromfile(str(d / "infinity_a.bin"), dtype=np.uint8).astype(bool)
        infinity_b = np.fromfile(str(d / "infinity_b.bin"), dtype=np.uint8).astype(bool)

        vk_alpha_g1 = _read_g1_points(d / "vk_alpha_g1.bin")[0]
        vk_beta_g1 = _read_g1_points(d / "vk_beta_g1.bin")[0]
        vk_beta_g2 = _read_g2_points(d / "vk_beta_g2.bin")[0]
        vk_gamma_g2 = _read_g2_points(d / "vk_gamma_g2.bin")[0]
        vk_ic = _read_g1_points(d / "vk_ic.bin")

    # R1CS COO
    t = time.perf_counter()
    r1cs_a = _read_coo(d / "r1cs_a.bin")
    print(f"  r1cs_a: {time.perf_counter() - t:.1f}s ({len(r1cs_a[0]):,} entries)")
    t = time.perf_counter()
    r1cs_b = _read_coo(d / "r1cs_b.bin")
    print(f"  r1cs_b: {time.perf_counter() - t:.1f}s ({len(r1cs_b[0]):,} entries)")
    t = time.perf_counter()
    r1cs_c = _read_coo(d / "r1cs_c.bin")
    print(f"  r1cs_c: {time.perf_counter() - t:.1f}s ({len(r1cs_c[0]):,} entries)")

    # Levels
    level_sizes = np.fromfile(str(d / "r1cs_level_sizes.bin"), dtype=np.uint32)
    level_order = np.fromfile(str(d / "r1cs_level_order.bin"), dtype=np.uint32)
    level_unknowns = _read_unknowns(d / "r1cs_level_unknowns.bin", num_constraints)

    # Hints (optional)
    hint_data = None
    if (d / "r1cs_hints.bin").exists():
        t = time.perf_counter()
        hint_data = _load_hints(d)
        print(
            f"  hints: {time.perf_counter() - t:.1f}s "
            f"({len(hint_data.instructions):,} hints, "
            f"{len(hint_data.coefficients):,} coefficients)"
        )

    return GnarkProvingData(
        num_wires=num_wires,
        num_public=num_public,
        num_secret=num_secret,
        num_internal=num_internal,
        num_constraints=num_constraints,
        domain_size=domain_size,
        witness_full=witness_full,
        solution_a=solution_a,
        solution_b=solution_b,
        solution_c=solution_c,
        pk_a_g1=pk_a_g1,
        pk_b_g1=pk_b_g1,
        pk_b_g2=pk_b_g2,
        pk_k_g1=pk_k_g1,
        pk_z_g1=pk_z_g1,
        pk_delta_g1=pk_delta_g1,
        pk_delta_g2=pk_delta_g2,
        infinity_a=infinity_a,
        infinity_b=infinity_b,
        vk_alpha_g1=vk_alpha_g1,
        vk_beta_g1=vk_beta_g1,
        vk_beta_g2=vk_beta_g2,
        vk_gamma_g2=vk_gamma_g2,
        vk_ic=vk_ic,
        r1cs_a=r1cs_a,
        r1cs_b=r1cs_b,
        r1cs_c=r1cs_c,
        level_sizes=level_sizes,
        level_order=level_order,
        level_unknowns=level_unknowns,
        hint_data=hint_data,
    )


def _read_field_elements(path: Path, count: int) -> np.ndarray:
    """Read 32-byte LE field elements as Python int objects."""
    raw = np.fromfile(str(path), dtype=np.uint8)
    assert (
        raw.size == count * FIELD_ELEM_SIZE
    ), f"Expected {count * FIELD_ELEM_SIZE} bytes, got {raw.size} in {path}"
    return _bytes_to_field_ints(raw, count, FIELD_ELEM_SIZE)


def _read_field_elements_native(path: Path, count: int) -> np.ndarray:
    """Read 32-byte LE Montgomery-form field elements as bn254_sf_mont."""
    raw = np.fromfile(str(path), dtype=np.uint8)
    assert (
        raw.size == count * FIELD_ELEM_SIZE
    ), f"Expected {count * FIELD_ELEM_SIZE} bytes, got {raw.size} in {path}"
    return raw.view(np.dtype(bn254_sf_mont))


def _read_g1_points(path: Path) -> list[tuple[int, int]]:
    """Read G1 affine points (X, Y) as 32-byte LE integers."""
    raw = np.fromfile(str(path), dtype=np.uint8)
    count = raw.size // G1_POINT_SIZE
    ints = _bytes_to_field_ints(raw, count * 2, FIELD_ELEM_SIZE)
    return [(int(ints[i * 2]), int(ints[i * 2 + 1])) for i in range(count)]


def _read_g2_points(path: Path) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Read G2 affine points as nested ((x0, x1), (y0, y1)) tuples."""
    raw = np.fromfile(str(path), dtype=np.uint8)
    count = raw.size // G2_POINT_SIZE
    ints = _bytes_to_field_ints(raw, count * 4, FIELD_ELEM_SIZE)
    return [
        (
            (int(ints[i * 4]), int(ints[i * 4 + 1])),
            (int(ints[i * 4 + 2]), int(ints[i * 4 + 3])),
        )
        for i in range(count)
    ]


def _read_coo(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Read COO sparse matrix: (rows, cols, vals).

    Returns:
        (rows uint32, cols uint32, vals bn254_sf_mont numpy array).
    """
    raw = np.fromfile(str(path), dtype=np.uint8)
    count = raw.size // COO_ENTRY_SIZE
    data = raw.reshape(count, COO_ENTRY_SIZE)

    # Rows and cols: read as uint32 directly via view
    rows = np.ascontiguousarray(data[:, :4]).view(np.uint32).flatten()
    cols = np.ascontiguousarray(data[:, 4:8]).view(np.uint32).flatten()

    # Values: view 32-byte LE directly as bn254_sf_mont dtype (zero-copy)
    val_bytes = np.ascontiguousarray(data[:, 8:40])
    vals = val_bytes.view(np.dtype(bn254_sf_mont)).flatten()

    return rows, cols, vals


def _read_unknowns(path: Path, count: int) -> tuple[np.ndarray, np.ndarray]:
    """Read per-constraint unknown info as (sides, wire_ids) numpy arrays."""
    raw = np.fromfile(str(path), dtype=np.uint8)
    assert raw.size == count * UNKNOWN_ENTRY_SIZE
    data = raw.reshape(count, UNKNOWN_ENTRY_SIZE)

    sides = data[:, 0].copy()  # (count,) uint8
    wire_ids = (
        np.ascontiguousarray(data[:, 1:5]).view(np.uint32).flatten()
    )  # (count,) uint32
    return sides, wire_ids


def _load_hints(d: Path) -> HintData:
    """Load hint instructions, coefficient table, and per-level offsets."""
    # Load coefficient table
    coeff_raw = np.fromfile(str(d / "r1cs_coefficients.bin"), dtype=np.uint8)
    num_coeffs = int(np.frombuffer(coeff_raw[:4], dtype=np.uint32)[0])
    coeff_bytes = coeff_raw[4 : 4 + num_coeffs * FIELD_ELEM_SIZE]
    coefficients = coeff_bytes.view(np.dtype(bn254_sf_mont))

    # Load per-level hint offsets
    level_offsets = np.fromfile(
        str(d / "r1cs_hint_level_offsets.bin"),
        dtype=np.uint32,
    )

    # Load hint instructions (variable-length records)
    hint_raw = (d / "r1cs_hints.bin").read_bytes()
    instructions = []
    pos = 0
    while pos < len(hint_raw):
        hint_id, level_idx, num_inputs, num_outputs, output_start = struct.unpack_from(
            "<5I",
            hint_raw,
            pos,
        )
        pos += 20

        inputs: list[list[tuple[int, int]]] = []
        for _ in range(num_inputs):
            num_terms = struct.unpack_from("<I", hint_raw, pos)[0]
            pos += 4
            terms: list[tuple[int, int]] = []
            for _ in range(num_terms):
                coeff_id, wire_id = struct.unpack_from("<II", hint_raw, pos)
                pos += 8
                terms.append((coeff_id, wire_id))
            inputs.append(terms)

        instructions.append(
            HintInstruction(
                hint_id=hint_id,
                level_idx=level_idx,
                inputs=inputs,
                output_start=output_start,
                num_outputs=num_outputs,
            )
        )

    return HintData(
        instructions=instructions,
        coefficients=coefficients,
        level_offsets=level_offsets,
    )


def _bytes_to_field_ints(raw: np.ndarray, count: int, elem_size: int) -> np.ndarray:
    """Convert raw bytes to array of Python int objects via uint64 limbs.

    Uses numpy vectorized object array ops for ~10x speedup over per-element
    int.from_bytes.
    """
    data = raw.reshape(count, elem_size)
    n_limbs = elem_size // 8
    limbs = np.ascontiguousarray(data).view(np.uint64).reshape(count, n_limbs)

    # Vectorized combine: numpy object arrays support arbitrary-precision int
    result = limbs[:, 0].astype(object)
    for j in range(1, n_limbs):
        result = result | (limbs[:, j].astype(object) << (64 * j))
    return result
