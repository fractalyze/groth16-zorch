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
import time
from pathlib import Path

import numpy as np
from zk_dtypes import (
    bn254_g1_affine,
    bn254_g1_affine_mont,
    bn254_g2_affine,
    bn254_g2_affine_mont,
    bn254_sf_mont,
)

from .types import GnarkProvingData

FIELD_ELEM_SIZE = 32  # bytes per BN254 scalar field element
G1_POINT_SIZE = 64  # 2 × 32B (X, Y)
G2_POINT_SIZE = 128  # 4 × 32B (X.A0, X.A1, Y.A0, Y.A1)

_G1_DT = np.dtype(bn254_g1_affine_mont)
_G2_DT = np.dtype(bn254_g2_affine_mont)


def load_gnark_export(
    export_dir: str | Path,
) -> GnarkProvingData:
    """Load all exported gnark data from a directory.

    Args:
        export_dir: Path to the export directory produced by cmd/export.

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

    # Constraint evaluations A·z, B·z — gnark's Solve already computed these
    # (solution.A / solution.B). Load them straight, zero-padded to the NTT
    # domain, so no Az/Bz recomputation is needed on the Python side.
    az_mont = _read_solution(d / "solution_a.bin", num_constraints, domain_size)
    bz_mont = _read_solution(d / "solution_b.bin", num_constraints, domain_size)

    t = time.perf_counter()
    pk_a_g1 = _read_g1_native(d / "pk_a_g1.bin")
    pk_b_g1 = _read_g1_native(d / "pk_b_g1.bin")
    pk_b_g2 = _read_g2_native(d / "pk_b_g2.bin")
    pk_k_g1 = _read_g1_native(d / "pk_k_g1.bin")
    pk_z_g1 = _read_g1_native(d / "pk_z_g1.bin")
    pk_delta_g1 = _read_g1_native(d / "pk_delta_g1.bin")
    pk_delta_g2 = _read_g2_native(d / "pk_delta_g2.bin")
    print(f"  PK points: {time.perf_counter() - t:.1f}s")

    infinity_a = np.fromfile(str(d / "infinity_a.bin"), dtype=np.uint8).astype(bool)
    infinity_b = np.fromfile(str(d / "infinity_b.bin"), dtype=np.uint8).astype(bool)

    vk_alpha_g1 = _read_g1_native(d / "vk_alpha_g1.bin")
    vk_beta_g1 = _read_g1_native(d / "vk_beta_g1.bin")
    vk_beta_g2 = _read_g2_native(d / "vk_beta_g2.bin")
    vk_gamma_g2 = _read_g2_native(d / "vk_gamma_g2.bin")
    vk_ic = _read_g1_native(d / "vk_ic.bin")

    return GnarkProvingData(
        num_wires=num_wires,
        num_public=num_public,
        num_secret=num_secret,
        num_internal=num_internal,
        num_constraints=num_constraints,
        domain_size=domain_size,
        witness_full=witness_full,
        az_mont=az_mont,
        bz_mont=bz_mont,
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
    )


def _read_field_elements_native(path: Path, count: int) -> np.ndarray:
    """Read 32-byte LE Montgomery-form field elements as bn254_sf_mont."""
    raw = np.fromfile(str(path), dtype=np.uint8)
    assert (
        raw.size == count * FIELD_ELEM_SIZE
    ), f"Expected {count * FIELD_ELEM_SIZE} bytes, got {raw.size} in {path}"
    return raw.view(np.dtype(bn254_sf_mont))


def _read_solution(path: Path, num_constraints: int, domain_size: int) -> np.ndarray:
    """Read a solution vector (A·z or B·z) and zero-pad to the NTT domain.

    On disk the vector has ``num_constraints`` Montgomery-form elements; the
    prover expects length ``domain_size`` (a power of two ≥ num_constraints).
    """
    out = np.zeros(domain_size, dtype=np.dtype(bn254_sf_mont))
    out[:num_constraints] = _read_field_elements_native(path, num_constraints)
    return out


def _read_g1_native(path: Path) -> np.ndarray:
    """Read G1 affine points as bn254_g1_affine numpy array.

    Disk format is Montgomery form (same bytes); we view as non-mont
    because the prover operates in non-mont EC point space.
    """
    raw = np.fromfile(str(path), dtype=np.uint8)
    return raw.view(_G1_DT).view(np.dtype(bn254_g1_affine))


def _read_g2_native(path: Path) -> np.ndarray:
    """Read G2 affine points as bn254_g2_affine numpy array.

    Same byte reinterpretation as G1 — see _read_g1_native.
    """
    raw = np.fromfile(str(path), dtype=np.uint8)
    return raw.view(_G2_DT).view(np.dtype(bn254_g2_affine))
