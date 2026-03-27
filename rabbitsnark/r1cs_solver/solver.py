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
prime-ir compiled r1cs-solver library.  Shared by both circom and gnark paths.
"""

from __future__ import annotations

import ctypes
import os
from dataclasses import dataclass
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from zk_dtypes import bn254_sf_mont

FIELD_ELEM_SIZE = 32  # 256-bit BN254 scalar = 32 bytes

_MONT_DT = np.dtype(bn254_sf_mont)

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
    if not runfiles_dir:
        # Infer from __file__ (without resolve — runfiles are symlinks).
        for parent in Path(__file__).absolute().parents:
            if parent.name.endswith(".runfiles"):
                runfiles_dir = str(parent)
                break
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
        jnp.array(az_mont),
        jnp.array(bz_mont),
    )


def solve_and_compute(
    witness_inputs: np.ndarray,
    solver: SolverData,
) -> tuple[np.ndarray, jnp.ndarray, jnp.ndarray]:
    """Solve witness + compute Az/Bz in one call. Replaces Go solver entirely.

    The Montgomery→standard conversion for z_std happens here (solve time),
    not at prove time, so that the prover receives standard form directly.

    Args:
        witness_inputs: (num_wires,) or (num_public + num_secret,)
            bn254_sf_mont numpy array.
        solver: Solver data loaded by ``load_solver_data``.

    Returns:
        (z_std, az, bz) where z_std is the solved witness in standard form
        and az/bz are padded JAX arrays in Montgomery form.
    """
    from jax import lax
    from zk_dtypes import bn254_sf

    if len(witness_inputs) >= solver.num_wires:
        # Full witness already provided (e.g. from gnark export) — skip solving.
        witness_full = witness_inputs
    else:
        witness_full = solve_witness(witness_inputs, solver)
    az, bz = compute_abc(
        witness_full,
        solver.csr,
        solver.num_constraints,
        solver.domain_size,
    )
    # Convert witness from Montgomery to standard form at solve time.
    z_mont = jnp.array(witness_full, dtype=bn254_sf_mont)
    z_std = np.asarray(lax.convert_element_type(z_mont, bn254_sf))
    return z_std, az, bz
