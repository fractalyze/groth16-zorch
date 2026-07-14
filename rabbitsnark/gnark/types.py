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

    Loaded from the binary export produced by gnark's ``r1csTyped.Solve``
    (see tests/gnark/gen_fixture). Field elements and coordinates are in
    Montgomery form (raw bytes on disk). All points are affine coordinates.
    """

    # Sizes
    num_wires: int
    num_public: int
    num_secret: int
    num_internal: int
    num_constraints: int
    domain_size: int

    # Witness + constraint evaluations, all produced by gnark's Go solver.
    witness_full: np.ndarray  # (num_wires,) bn254_sf_mont
    az_mont: np.ndarray  # (domain_size,) bn254_sf_mont — A·z, zero-padded
    bz_mont: np.ndarray  # (domain_size,) bn254_sf_mont — B·z, zero-padded

    # PK points — numpy arrays with native ZK dtypes
    # All uncompacted (zeros at infinity positions)
    pk_a_g1: np.ndarray  # (num_wires,) bn254_g1_affine
    pk_b_g1: np.ndarray  # (num_wires,) bn254_g1_affine
    pk_b_g2: np.ndarray  # (num_wires,) bn254_g2_affine
    pk_k_g1: np.ndarray  # (num_k,) bn254_g1_affine
    pk_z_g1: np.ndarray  # (num_z,) bn254_g1_affine
    pk_delta_g1: np.ndarray  # (1,) bn254_g1_affine
    pk_delta_g2: np.ndarray  # (1,) bn254_g2_affine

    # Infinity masks
    infinity_a: np.ndarray  # (num_wires,) bool
    infinity_b: np.ndarray  # (num_wires,) bool

    # VK points — numpy arrays with native ZK dtypes
    vk_alpha_g1: np.ndarray  # (1,) bn254_g1_affine
    vk_beta_g1: np.ndarray  # (1,) bn254_g1_affine
    vk_beta_g2: np.ndarray  # (1,) bn254_g2_affine
    vk_gamma_g2: np.ndarray  # (1,) bn254_g2_affine
    vk_ic: np.ndarray  # (num_vk_ic,) bn254_g1_affine
