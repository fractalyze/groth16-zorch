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
from zk_dtypes import bn254_sf_mont

from .types import GnarkProvingData

FIELD_ELEM_SIZE = 32  # bytes per BN254 scalar field element
G1_POINT_SIZE = 64  # 2 × 32B (X, Y)
G2_POINT_SIZE = 128  # 4 × 32B (X.A0, X.A1, Y.A0, Y.A1)


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

    return GnarkProvingData(
        num_wires=num_wires,
        num_public=num_public,
        num_secret=num_secret,
        num_internal=num_internal,
        num_constraints=num_constraints,
        domain_size=domain_size,
        witness_full=witness_full,
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
