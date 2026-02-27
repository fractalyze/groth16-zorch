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

"""Groth16 proof verification using pairing check.

Verification equation (rearranged for multi-pairing check = 1):

    e(-A, B) * e(alpha, beta) * e(vk_x, gamma) * e(C, delta) = 1

Where vk_x = IC[0] + sum_i(pub[i] * IC[i+1])
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import jax.numpy as jnp
import numpy as np
from jax import lax
from zk_dtypes import bn254_g1_affine, bn254_g2_affine, bn254_sf

from rabbitsnark.circom.zkey.verifying_key import G1Point, G2Point
from rabbitsnark.msm import MSMBn254

from .proof import Groth16Proof

if TYPE_CHECKING:
    from rabbitsnark.circom.zkey.zkey import ZKeyV1

BN254_FQ_MODULUS = (
    21888242871839275222246405745257275088696311157297823662689037894645226208583
)


@dataclass
class VerificationKey:
    """Groth16 verification key for pairing-based proof verification.

    Parsed from snarkjs ``verification_key.json`` or extracted from a zkey.
    """

    alpha_g1: G1Point
    beta_g2: G2Point
    gamma_g2: G2Point
    delta_g2: G2Point
    ic: list[G1Point]

    @classmethod
    def from_json(cls, data: dict) -> VerificationKey:
        """Parse from snarkjs verification_key.json dict."""
        alpha_g1 = _parse_g1(data["vk_alpha_1"])
        beta_g2 = _parse_g2(data["vk_beta_2"])
        gamma_g2 = _parse_g2(data["vk_gamma_2"])
        delta_g2 = _parse_g2(data["vk_delta_2"])
        ic = [_parse_g1(pt) for pt in data["IC"]]
        return cls(
            alpha_g1=alpha_g1,
            beta_g2=beta_g2,
            gamma_g2=gamma_g2,
            delta_g2=delta_g2,
            ic=ic,
        )

    @classmethod
    def from_file(cls, path: str | Path) -> VerificationKey:
        """Load from a snarkjs verification_key.json file."""
        with open(path) as f:
            return cls.from_json(json.load(f))

    @classmethod
    def from_zkey(cls, zkey: ZKeyV1) -> VerificationKey:
        """Extract verification key from a parsed zkey."""
        vk = zkey.verifying_key
        return cls(
            alpha_g1=vk.alpha_g1,
            beta_g2=vk.beta_g2,
            gamma_g2=vk.gamma_g2,
            delta_g2=vk.delta_g2,
            ic=zkey.ic,
        )


def verify(
    vk: VerificationKey,
    proof: Groth16Proof | dict,
    public_signals: list[str],
) -> bool:
    """Verify a Groth16 proof using multi-pairing check.

    Args:
        vk: Verification key.
        proof: Groth16 proof (``Groth16Proof`` or snarkjs JSON dict).
        public_signals: Public signal values as decimal strings.

    Returns:
        True if the proof is valid.
    """
    # Parse proof
    if isinstance(proof, dict):
        pi_a = _parse_g1(proof["pi_a"])
        pi_b = _parse_g2(proof["pi_b"])
        pi_c = _parse_g1(proof["pi_c"])
    else:
        proof_json = proof.to_json()
        pi_a = _parse_g1(proof_json["pi_a"])
        pi_b = _parse_g2(proof_json["pi_b"])
        pi_c = _parse_g1(proof_json["pi_c"])

    # Compute vk_x = IC[0] + sum_i(pub[i] * IC[i+1]) via MSM
    pub_scalars = [int(s) for s in public_signals]
    msm = MSMBn254()
    msm_scalars = jnp.array([1] + pub_scalars, dtype=bn254_sf)
    msm_points = jnp.array(
        [bn254_g1_affine((pt.x, pt.y)) for pt in vk.ic],
        dtype=bn254_g1_affine,
    )
    vk_x_xyzz = msm.compute(msm_scalars, msm_points)
    vk_x_affine = lax.convert_element_type(vk_x_xyzz, bn254_g1_affine)

    # Extract vk_x coordinates from JAX result
    vk_x_np = np.array(vk_x_affine).item()
    vk_x_coords = vk_x_np.raw
    vk_x_x, vk_x_y = int(vk_x_coords[0]), int(vk_x_coords[1])

    # Negate pi_a: (x, p - y)
    neg_pi_a_x = pi_a.x
    neg_pi_a_y = (BN254_FQ_MODULUS - pi_a.y) % BN254_FQ_MODULUS

    # Build G1 and G2 arrays for pairing check:
    # e(-A, B) * e(alpha, beta) * e(vk_x, gamma) * e(C, delta) = 1
    g1_points = jnp.array(
        [
            bn254_g1_affine((neg_pi_a_x, neg_pi_a_y)),
            bn254_g1_affine((vk.alpha_g1.x, vk.alpha_g1.y)),
            bn254_g1_affine((vk_x_x, vk_x_y)),
            bn254_g1_affine((pi_c.x, pi_c.y)),
        ],
        dtype=bn254_g1_affine,
    )
    g2_points = jnp.array(
        [
            bn254_g2_affine((pi_b.x, pi_b.y)),
            bn254_g2_affine((vk.beta_g2.x, vk.beta_g2.y)),
            bn254_g2_affine((vk.gamma_g2.x, vk.gamma_g2.y)),
            bn254_g2_affine((vk.delta_g2.x, vk.delta_g2.y)),
        ],
        dtype=bn254_g2_affine,
    )

    result = lax.pairing_check(g1_points, g2_points)
    return bool(result)


# ---------------------------------------------------------------------------
# snarkjs JSON parsing helpers
# ---------------------------------------------------------------------------


def _parse_g1(coords: list[str]) -> G1Point:
    """Parse a G1 point from snarkjs JSON format [x, y, "1"]."""
    return G1Point.from_ints(int(coords[0]), int(coords[1]))


def _parse_g2(coords: list[list[str]]) -> G2Point:
    """Parse a G2 point from snarkjs JSON format [[x0,x1],[y0,y1],["1","0"]]."""
    x = (int(coords[0][0]), int(coords[0][1]))
    y = (int(coords[1][0]), int(coords[1][1]))
    return G2Point.from_ints(x, y)
