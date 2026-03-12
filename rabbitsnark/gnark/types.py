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
