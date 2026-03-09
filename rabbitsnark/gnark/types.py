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

"""Data types for gnark Groth16 exported data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class GnarkProvingData:
    """All data needed to generate a gnark Groth16 proof.

    Loaded from the binary export produced by cmd/export.
    Field elements and coordinates are in Montgomery form (raw bytes on disk).
    All points are affine coordinates.
    """

    # Sizes
    num_wires: int
    num_public: int
    num_secret: int
    num_internal: int
    num_constraints: int
    domain_size: int

    # Witness (from Go solver)
    witness_full: np.ndarray  # (num_wires,) bn254_sf_mont

    # Solution vectors (from Go solver)
    solution_a: np.ndarray  # (num_constraints,) int objects
    solution_b: np.ndarray  # (num_constraints,) int objects
    solution_c: np.ndarray  # (num_constraints,) int objects

    # PK points — G1: (x, y) tuples, G2: ((x0, x1), (y0, y1)) tuples
    # All uncompacted (zeros at infinity positions)
    pk_a_g1: list[tuple[int, int]]  # (num_wires,)
    pk_b_g1: list[tuple[int, int]]  # (num_wires,)
    pk_b_g2: list[tuple[tuple[int, int], tuple[int, int]]]  # (num_wires,)
    pk_k_g1: list[tuple[int, int]]  # (num_k,) — private wires only
    pk_z_g1: list[tuple[int, int]]  # (num_z,) — H polynomial
    pk_delta_g1: tuple[int, int]
    pk_delta_g2: tuple[tuple[int, int], tuple[int, int]]

    # Infinity masks
    infinity_a: np.ndarray  # (num_wires,) bool
    infinity_b: np.ndarray  # (num_wires,) bool

    # VK points
    vk_alpha_g1: tuple[int, int]
    vk_beta_g1: tuple[int, int]
    vk_beta_g2: tuple[tuple[int, int], tuple[int, int]]
    vk_gamma_g2: tuple[tuple[int, int], tuple[int, int]]
    vk_ic: list[tuple[int, int]]  # (num_vk_ic,) — gnark vk.G1.K

    # R1CS matrices in COO format (rows uint32, cols uint32, vals bn254_sf)
    r1cs_a: tuple[np.ndarray, np.ndarray, np.ndarray]
    r1cs_b: tuple[np.ndarray, np.ndarray, np.ndarray]
    r1cs_c: tuple[np.ndarray, np.ndarray, np.ndarray]

    # R1CS levels (for GPU constraint solver)
    level_sizes: np.ndarray  # (num_levels,) uint32
    level_order: (
        np.ndarray
    )  # (num_constraints,) uint32 — constraint indices in solve order
    level_unknowns: tuple[np.ndarray, np.ndarray]  # (sides uint8, wire_ids uint32)

    # Hint data (optional — None if hint export files not present)
    hint_data: HintData | None = None


@dataclass
class HintInstruction:
    """A single hint instruction from the gnark R1CS circuit.

    Each hint evaluates a function on linear combinations of wire values
    and writes the results into a contiguous output wire range.
    """

    hint_id: int  # FNV32a hash of Go function name
    level_idx: int  # which level this hint belongs to
    inputs: list[list[tuple[int, int]]]  # list of LinearExpressions,
    # each = list of (coeff_id, wire_id)
    output_start: int  # first output wire ID
    num_outputs: int  # number of output wires


@dataclass
class HintData:
    """All hint-related data for integrated R1CS + hint solving."""

    instructions: list[HintInstruction]  # all hints, in level order
    coefficients: np.ndarray  # (num_coeffs,) bn254_sf
    level_offsets: np.ndarray  # (num_levels + 1,) uint32
