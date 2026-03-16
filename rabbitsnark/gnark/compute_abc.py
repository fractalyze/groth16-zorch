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

"""Native R1CS solver and Az/Bz computation via r1cs-solver shared library.

Wraps ``_mlir_ciface_solve_r1cs`` and ``_mlir_ciface_compute_abc`` from the
prime-ir compiled r1cs-solver library, replacing the Go solver entirely.
"""

from __future__ import annotations

import ctypes
import json
import os
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from zk_dtypes import bn254_sf_mont

from .loader import FIELD_ELEM_SIZE

_MONT_DT = np.dtype(bn254_sf_mont)
_UNKNOWN_ENTRY_SIZE = 5  # 1B side + 4B wire_id

# Default path (relative to this file) for local development.
_DEFAULT_LIB_PATH = (
    Path(__file__).resolve().parents[2]
    / "r1cs-solver"
    / "bazel-bin"
    / "solver"
    / "libr1cs_solver.so"
)

# Bazel runfiles path for CI / bazel test.
_RUNFILES_LIB_PATH = "r1cs_solver/solver/libr1cs_solver.so"


# ---------------------------------------------------------------------------
# ctypes helpers
# ---------------------------------------------------------------------------


class _StridedMemRef1D(ctypes.Structure):
    """Mirrors MLIR's StridedMemRefType<T, 1>."""

    _fields_ = [
        ("basePtr", ctypes.c_void_p),
        ("data", ctypes.c_void_p),
        ("offset", ctypes.c_int64),
        ("sizes", ctypes.c_int64),
        ("strides", ctypes.c_int64),
    ]


def _make_memref(arr: np.ndarray) -> _StridedMemRef1D:
    """Create a StridedMemRef1D pointing to a numpy array's data."""
    ref = _StridedMemRef1D()
    ptr = arr.ctypes.data
    ref.basePtr = ptr
    ref.data = ptr
    ref.offset = 0
    ref.sizes = arr.shape[0]
    ref.strides = 1
    return ref


_lib_cache: ctypes.CDLL | None = None
_REF_PTR = ctypes.POINTER(_StridedMemRef1D)


def _find_runfiles_lib() -> str | None:
    """Try to find libr1cs_solver.so in Bazel runfiles."""
    runfiles_dir = os.environ.get("RUNFILES_DIR")
    if runfiles_dir:
        candidate = os.path.join(runfiles_dir, _RUNFILES_LIB_PATH)
        if os.path.isfile(candidate):
            return candidate
    return None


def _load_library() -> ctypes.CDLL:
    """Load the r1cs_solver shared library (cached)."""
    global _lib_cache
    if _lib_cache is not None:
        return _lib_cache

    lib_path = os.environ.get("R1CS_SOLVER_LIB")
    if not lib_path:
        lib_path = _find_runfiles_lib() or str(_DEFAULT_LIB_PATH)
    _lib_cache = ctypes.CDLL(lib_path)

    # compute_abc: 13 memrefs + 1 scalar
    _lib_cache._mlir_ciface_compute_abc.restype = None
    _lib_cache._mlir_ciface_compute_abc.argtypes = [
        _REF_PTR,  # witness
        _REF_PTR,
        _REF_PTR,
        _REF_PTR,  # A CSR
        _REF_PTR,
        _REF_PTR,
        _REF_PTR,  # B CSR
        _REF_PTR,
        _REF_PTR,
        _REF_PTR,  # C CSR
        _REF_PTR,
        _REF_PTR,
        _REF_PTR,  # az, bz, cz
        ctypes.c_int64,  # num_constraints
    ]

    # solve_r1cs: 22 memrefs + 2 scalars
    _lib_cache._mlir_ciface_solve_r1cs.restype = None
    _lib_cache._mlir_ciface_solve_r1cs.argtypes = [
        _REF_PTR,  # witness (in/out)
        _REF_PTR,
        _REF_PTR,
        _REF_PTR,  # A CSR
        _REF_PTR,
        _REF_PTR,
        _REF_PTR,  # B CSR
        _REF_PTR,
        _REF_PTR,
        _REF_PTR,  # C CSR
        _REF_PTR,
        _REF_PTR,
        _REF_PTR,  # unk_side, unk_inv_coeff, target_idx
        _REF_PTR,
        _REF_PTR,
        _REF_PTR,  # level_cids, level_offsets, level_sizes
        _REF_PTR,
        _REF_PTR,
        _REF_PTR,
        _REF_PTR,  # hint_ids, output_starts, num_outputs, num_inputs
        _REF_PTR,  # hint_le_offsets
        _REF_PTR,
        _REF_PTR,  # hint_term_coeff_ids, hint_term_wire_ids
        _REF_PTR,  # hint_input_le_offsets
        _REF_PTR,  # hint_level_offsets
        _REF_PTR,  # coefficients
        _REF_PTR,  # hint_scratch
        ctypes.c_int64,  # max_hint_inputs
        ctypes.c_int64,  # num_levels
    ]

    return _lib_cache


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CSRMatrices:
    """CSR representation of R1CS A, B, C matrices."""

    a_indptr: np.ndarray  # (num_constraints + 1,) int64
    a_indices: np.ndarray  # (nnz_a,) int32
    a_values: np.ndarray  # (nnz_a, 32) uint8 — raw field element bytes

    b_indptr: np.ndarray
    b_indices: np.ndarray
    b_values: np.ndarray

    c_indptr: np.ndarray
    c_indices: np.ndarray
    c_values: np.ndarray


@dataclass
class SolverData:
    """All data needed for native R1CS witness solving + Az/Bz computation."""

    csr: CSRMatrices

    # Unknown info per constraint
    unk_side: np.ndarray  # (num_constraints,) uint8
    unk_inv_coeff: np.ndarray  # (num_constraints, 32) uint8
    target_idx: np.ndarray  # (num_constraints,) int32

    # Level info
    level_cids: np.ndarray  # (num_constraints,) int32
    level_offsets: np.ndarray  # (num_levels,) int32
    level_sizes: np.ndarray  # (num_levels,) int32

    # Hint arrays (flattened)
    hint_ids: np.ndarray  # (num_hints,) int32
    hint_output_starts: np.ndarray  # (num_hints,) int32
    hint_num_outputs: np.ndarray  # (num_hints,) int32
    hint_num_inputs: np.ndarray  # (num_hints,) int32
    hint_le_offsets: np.ndarray  # (num_hints + 1,) int64
    hint_term_coeff_ids: np.ndarray  # (total_terms,) int32
    hint_term_wire_ids: np.ndarray  # (total_terms,) int32
    hint_input_le_offsets: np.ndarray  # (total_inputs + 1,) int64
    hint_level_offsets: np.ndarray  # (num_levels + 1,) int32
    coefficients: np.ndarray  # (num_coefficients, 32) uint8

    # Scalars
    max_hint_inputs: int
    num_levels: int
    num_wires: int
    num_public: int
    num_secret: int
    num_constraints: int
    domain_size: int


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
        np.array(hint_ids, dtype=np.int32),
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


# ---------------------------------------------------------------------------
# Native calls
# ---------------------------------------------------------------------------


def solve_witness(
    witness_inputs: np.ndarray,
    solver: SolverData,
) -> np.ndarray:
    """Solve the R1CS witness from public+secret inputs.

    Takes the first (num_public + num_secret) wire values and fills in all
    internal wires by calling the native ``solve_r1cs``.

    Args:
        witness_inputs: (num_public + num_secret,) bn254_sf_mont numpy array,
            or full (num_wires,) array (only first num_public+num_secret used).
        solver: Solver data loaded by ``load_solver_data``.

    Returns:
        (num_wires,) bn254_sf_mont numpy array with all wires solved.
    """
    lib = _load_library()
    csr = solver.csr
    n_input = solver.num_public + solver.num_secret

    # Build witness buffer: copy all provided wire values (public+secret at
    # minimum; full witness if available), then let solver fill in the rest.
    witness_buf = np.zeros(solver.num_wires * FIELD_ELEM_SIZE, dtype=np.uint8)
    input_bytes = witness_inputs.view(np.uint8).ravel()
    n_copy = min(len(witness_inputs), solver.num_wires)
    witness_buf[: n_copy * FIELD_ELEM_SIZE] = input_bytes[: n_copy * FIELD_ELEM_SIZE]

    # Compute max_hints_per_level for scratch buffer sizing
    counts = solver.hint_level_offsets[1:] - solver.hint_level_offsets[:-1]
    max_hints_per_level = int(np.max(counts, initial=0))

    scratch_size = max(max_hints_per_level * solver.max_hint_inputs, 1)
    hint_scratch = np.zeros(scratch_size * FIELD_ELEM_SIZE, dtype=np.uint8)

    # Build all memrefs
    mr_w = _make_memref(witness_buf)
    mr_a_indptr = _make_memref(csr.a_indptr)
    mr_a_indices = _make_memref(csr.a_indices)
    mr_a_values = _make_memref(csr.a_values.view(np.uint8).ravel())
    mr_b_indptr = _make_memref(csr.b_indptr)
    mr_b_indices = _make_memref(csr.b_indices)
    mr_b_values = _make_memref(csr.b_values.view(np.uint8).ravel())
    mr_c_indptr = _make_memref(csr.c_indptr)
    mr_c_indices = _make_memref(csr.c_indices)
    mr_c_values = _make_memref(csr.c_values.view(np.uint8).ravel())
    mr_unk_side = _make_memref(solver.unk_side)
    mr_unk_inv_coeff = _make_memref(solver.unk_inv_coeff.view(np.uint8).ravel())
    mr_target_idx = _make_memref(solver.target_idx)
    mr_level_cids = _make_memref(solver.level_cids)
    mr_level_offsets = _make_memref(solver.level_offsets)
    mr_level_sizes = _make_memref(solver.level_sizes)
    mr_hint_ids = _make_memref(solver.hint_ids)
    mr_hint_output_starts = _make_memref(solver.hint_output_starts)
    mr_hint_num_outputs = _make_memref(solver.hint_num_outputs)
    mr_hint_num_inputs = _make_memref(solver.hint_num_inputs)
    mr_hint_le_offsets = _make_memref(solver.hint_le_offsets)
    mr_hint_term_coeff_ids = _make_memref(solver.hint_term_coeff_ids)
    mr_hint_term_wire_ids = _make_memref(solver.hint_term_wire_ids)
    mr_hint_input_le_offsets = _make_memref(solver.hint_input_le_offsets)
    mr_hint_level_offsets = _make_memref(solver.hint_level_offsets)
    mr_coefficients = _make_memref(
        solver.coefficients.view(np.uint8).ravel()
        if solver.coefficients.size > 0
        else np.zeros(FIELD_ELEM_SIZE, dtype=np.uint8)
    )
    mr_hint_scratch = _make_memref(hint_scratch)

    lib._mlir_ciface_solve_r1cs(
        ctypes.byref(mr_w),
        ctypes.byref(mr_a_indptr),
        ctypes.byref(mr_a_indices),
        ctypes.byref(mr_a_values),
        ctypes.byref(mr_b_indptr),
        ctypes.byref(mr_b_indices),
        ctypes.byref(mr_b_values),
        ctypes.byref(mr_c_indptr),
        ctypes.byref(mr_c_indices),
        ctypes.byref(mr_c_values),
        ctypes.byref(mr_unk_side),
        ctypes.byref(mr_unk_inv_coeff),
        ctypes.byref(mr_target_idx),
        ctypes.byref(mr_level_cids),
        ctypes.byref(mr_level_offsets),
        ctypes.byref(mr_level_sizes),
        ctypes.byref(mr_hint_ids),
        ctypes.byref(mr_hint_output_starts),
        ctypes.byref(mr_hint_num_outputs),
        ctypes.byref(mr_hint_num_inputs),
        ctypes.byref(mr_hint_le_offsets),
        ctypes.byref(mr_hint_term_coeff_ids),
        ctypes.byref(mr_hint_term_wire_ids),
        ctypes.byref(mr_hint_input_le_offsets),
        ctypes.byref(mr_hint_level_offsets),
        ctypes.byref(mr_coefficients),
        ctypes.byref(mr_hint_scratch),
        ctypes.c_int64(solver.max_hint_inputs),
        ctypes.c_int64(solver.num_levels),
    )

    return witness_buf.view(_MONT_DT)


def compute_abc(
    witness: np.ndarray,
    csr: CSRMatrices,
    num_constraints: int,
    domain_size: int,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute Az, Bz via native CSR SpMV and return as padded JAX arrays.

    Args:
        witness: (num_wires,) bn254_sf_mont numpy array.
        csr: CSR matrices loaded by ``load_csr_matrices``.
        num_constraints: Number of R1CS constraints.
        domain_size: NTT domain size (power of 2, >= num_constraints).

    Returns:
        Tuple of (az, bz) JAX arrays of shape (domain_size,) in Montgomery form.
    """
    lib = _load_library()

    # Reinterpret witness raw bytes for native call (32B per element)
    w_flat = np.ascontiguousarray(witness.view(np.uint8)).ravel()

    # Allocate output buffers (num_constraints elements, 32B each)
    az_buf = np.zeros(num_constraints * FIELD_ELEM_SIZE, dtype=np.uint8)
    bz_buf = np.zeros(num_constraints * FIELD_ELEM_SIZE, dtype=np.uint8)
    cz_buf = np.zeros(num_constraints * FIELD_ELEM_SIZE, dtype=np.uint8)

    # Build memrefs
    mr_w = _make_memref(w_flat)
    mr_a_indptr = _make_memref(csr.a_indptr)
    mr_a_indices = _make_memref(csr.a_indices)
    mr_a_values = _make_memref(csr.a_values.view(np.uint8).ravel())
    mr_b_indptr = _make_memref(csr.b_indptr)
    mr_b_indices = _make_memref(csr.b_indices)
    mr_b_values = _make_memref(csr.b_values.view(np.uint8).ravel())
    mr_c_indptr = _make_memref(csr.c_indptr)
    mr_c_indices = _make_memref(csr.c_indices)
    mr_c_values = _make_memref(csr.c_values.view(np.uint8).ravel())
    mr_az = _make_memref(az_buf)
    mr_bz = _make_memref(bz_buf)
    mr_cz = _make_memref(cz_buf)

    lib._mlir_ciface_compute_abc(
        ctypes.byref(mr_w),
        ctypes.byref(mr_a_indptr),
        ctypes.byref(mr_a_indices),
        ctypes.byref(mr_a_values),
        ctypes.byref(mr_b_indptr),
        ctypes.byref(mr_b_indices),
        ctypes.byref(mr_b_values),
        ctypes.byref(mr_c_indptr),
        ctypes.byref(mr_c_indices),
        ctypes.byref(mr_c_values),
        ctypes.byref(mr_az),
        ctypes.byref(mr_bz),
        ctypes.byref(mr_cz),
        ctypes.c_int64(num_constraints),
    )

    # Reinterpret output as bn254_sf_mont and pad to domain_size
    az_mont = az_buf.view(_MONT_DT)
    bz_mont = bz_buf.view(_MONT_DT)

    if num_constraints < domain_size:
        pad = np.zeros(domain_size - num_constraints, dtype=_MONT_DT)
        az_mont = np.concatenate([az_mont, pad])
        bz_mont = np.concatenate([bz_mont, pad])

    return (
        jnp.array(az_mont.tolist(), dtype=bn254_sf_mont),
        jnp.array(bz_mont.tolist(), dtype=bn254_sf_mont),
    )


def solve_and_compute(
    witness_inputs: np.ndarray,
    solver: SolverData,
) -> tuple[np.ndarray, jnp.ndarray, jnp.ndarray]:
    """Solve witness + compute Az/Bz in one call. Replaces Go solver entirely.

    Args:
        witness_inputs: (num_wires,) or (num_public + num_secret,)
            bn254_sf_mont numpy array.
        solver: Solver data loaded by ``load_solver_data``.

    Returns:
        (witness_full, az, bz) where witness_full is the solved numpy array
        and az/bz are padded JAX arrays.
    """
    witness_full = solve_witness(witness_inputs, solver)
    az, bz = compute_abc(
        witness_full,
        solver.csr,
        solver.num_constraints,
        solver.domain_size,
    )
    return witness_full, az, bz
