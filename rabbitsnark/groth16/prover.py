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

"""Groth16 prover implementation.

Generates a Groth16 proof from a zkey (proving key) and wtns (witness),
wiring together SpMV, NTT, and MSM primitives.

Algorithm:
    1. SpMV:   Az = A·z,  Bz = B·z
    2. Hadamard: Cz = Az * Bz
    3. IFFT:   a_poly, b_poly, c_poly
    4. Coset NTT (shift = omega_{2n})
    5. Quotient: h_evals = a_coset * b_coset - c_coset
    6. MSMs for proof components
    7. Assemble: pi_A, pi_B2, pi_C
"""

from __future__ import annotations

import math
import secrets
from typing import TYPE_CHECKING

import jax.numpy as jnp
from jax import lax
from zk_dtypes import (
    bn254_g1_affine,
    bn254_g1_xyzz,
    bn254_g2_affine,
    bn254_g2_xyzz,
    bn254_sf,
    bn254_sf_mont,
)

from rabbitsnark.msm import MSMBn254, MSMBn254G2
from rabbitsnark.ntt import BN254_FR_ROOT_OF_UNITY, NTT, coset_ntt
from rabbitsnark.spmv import build_r1cs_matrices, spmv, witness_to_montgomery

from .proof import Groth16Proof, write_public_signals  # noqa: F401

if TYPE_CHECKING:
    from jax import Array

    from rabbitsnark.circom.wtns.wtns import WtnsV2
    from rabbitsnark.circom.zkey.verifying_key import G1Point, G2Point
    from rabbitsnark.circom.zkey.zkey import ZKeyV1

BN254_TWO_ADIC_BITS = 28
BN254_FR_MODULUS = (
    21888242871839275222246405745257275088548364400416034343698204186575808495617
)


def prove(
    zkey: ZKeyV1,
    wtns: WtnsV2,
    *,
    no_zk: bool = False,
) -> tuple[Groth16Proof, list[str]]:
    """Generate a Groth16 proof.

    Args:
        zkey: Parsed proving key (ZKeyV1).
        wtns: Parsed witness (WtnsV2).
        no_zk: If True, use r=s=0 for a deterministic (non-ZK) proof.

    Returns:
        Tuple of (proof, public_signals).
    """
    p = zkey.header_groth.r.to_int()
    num_vars = zkey.header_groth.num_vars
    num_public = zkey.header_groth.num_public_inputs
    domain_size = zkey.domain_size
    vk = zkey.verifying_key

    l = num_public  # public inputs count
    m = num_vars  # total variables

    # --- Randomness ---
    if no_zk:
        r, s = 0, 0
    else:
        r = secrets.randbelow(p)
        s = secrets.randbelow(p)

    # --- Step 1: SpMV  (Az = A·z, Bz = B·z) ---
    matrix_a, matrix_b = build_r1cs_matrices(zkey, bn254_sf_mont)
    z_mont = witness_to_montgomery(wtns.witnesses, bn254_sf_mont, p)
    az = spmv(matrix_a, z_mont)
    bz = spmv(matrix_b, z_mont)

    # --- Step 2–5: Quotient polynomial h_evals ---
    h_evals_mont = _compute_h_evals(az, bz, domain_size, p)

    # --- Step 6: Prepare MSM scalars (standard form) ---
    z_std = jnp.array([int(w) for w in wtns.witnesses], dtype=bn254_sf)
    h_evals_std = jnp.array([int(v) for v in h_evals_mont], dtype=bn254_sf)

    # --- Step 7: MSMs ---
    msm_g1 = MSMBn254()
    msm_g2 = MSMBn254G2()

    pa1 = _g1_points_to_array(zkey.points_a1)
    pb1 = _g1_points_to_array(zkey.points_b1)
    pb2 = _g2_points_to_array(zkey.points_b2)
    pc1 = _g1_points_to_array(zkey.points_c1)
    ph1 = _g1_points_to_array(zkey.points_h1)

    msm_1 = msm_g1.compute(z_std[:m], pa1)  # G1
    msm_2 = msm_g1.compute(z_std[:m], pb1)  # G1
    msm_3 = msm_g2.compute(z_std[:m], pb2)  # G2
    msm_4 = msm_g1.compute(z_std[l + 1 : m], pc1)  # G1 (witness only)
    msm_5 = msm_g1.compute(h_evals_std, ph1)  # G1

    # --- Step 8: Assemble proof ---
    alpha_1 = _g1_to_xyzz(vk.alpha_g1)
    beta_1 = _g1_to_xyzz(vk.beta_g1)
    beta_2 = _g2_to_xyzz(vk.beta_g2)

    pi_a = alpha_1 + msm_1
    pi_b1 = beta_1 + msm_2
    pi_b2 = beta_2 + msm_3
    pi_c = msm_4 + msm_5

    if not no_zk:
        r_delta_1 = _ec_scalar_mul_g1(r, vk.delta_g1, msm_g1)
        s_delta_1 = _ec_scalar_mul_g1(s, vk.delta_g1, msm_g1)
        s_delta_2 = _ec_scalar_mul_g2(s, vk.delta_g2, msm_g2)

        # Compute s·π_A and r·π_B₁ on the UNBLINDED values before updating
        s_pi_a = _scalar_mul_xyzz_g1(s, pi_a, msm_g1)
        r_pi_b1 = _scalar_mul_xyzz_g1(r, pi_b1, msm_g1)

        pi_a = pi_a + r_delta_1
        pi_b1 = pi_b1 + s_delta_1
        pi_b2 = pi_b2 + s_delta_2

        # -(rs)·δ₁ = (p - rs)·δ₁
        neg_rs = (p - (r * s) % p) % p
        neg_rs_delta_1 = _ec_scalar_mul_g1(neg_rs, vk.delta_g1, msm_g1)

        pi_c = pi_c + s_pi_a + r_pi_b1 + neg_rs_delta_1

    pi_a = lax.convert_element_type(pi_a, bn254_g1_affine)
    pi_b2 = lax.convert_element_type(pi_b2, bn254_g2_affine)
    pi_c = lax.convert_element_type(pi_c, bn254_g1_affine)

    proof = Groth16Proof(pi_a=pi_a, pi_b=pi_b2, pi_c=pi_c)
    public_signals = write_public_signals(wtns.witnesses, num_public)

    return proof, public_signals


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _compute_h_evals(
    az: Array,
    bz: Array,
    domain_size: int,
    modulus: int,
) -> Array:
    """Compute quotient polynomial evaluations on coset (Montgomery form).

    Steps: Hadamard → IFFT → coset NTT → quotient.

    Args:
        az: A·z evaluations (Montgomery).
        bz: B·z evaluations (Montgomery).
        domain_size: Domain size n (power of 2).
        modulus: Scalar field modulus p.

    Returns:
        h_evals in Montgomery form, shape (n,).
    """
    # Step 2: Hadamard product
    cz = az * bz

    # Step 3: IFFT
    ntt = NTT(bn254_sf_mont, BN254_FR_ROOT_OF_UNITY)
    a_poly = ntt.inverse(az)
    b_poly = ntt.inverse(bz)
    c_poly = ntt.inverse(cz)

    # Step 4: Coset NTT with shift = omega_{2n}
    log_n = int(math.log2(domain_size))
    omega_2n_int = pow(
        BN254_FR_ROOT_OF_UNITY,
        1 << (BN254_TWO_ADIC_BITS - log_n - 1),
        modulus,
    )
    coset_shift = jnp.array(omega_2n_int, dtype=bn254_sf_mont)

    a_coset = coset_ntt(ntt, a_poly, coset_shift)
    b_coset = coset_ntt(ntt, b_poly, coset_shift)
    c_coset = coset_ntt(ntt, c_poly, coset_shift)

    # Step 5: Quotient h = a * b - c on coset
    return a_coset * b_coset - c_coset


def _g1_to_xyzz(point: G1Point) -> Array:
    """Convert a single G1Point to a JAX XYZZ point."""
    return jnp.array(bn254_g1_xyzz((point.x, point.y, 1, 1)), dtype=bn254_g1_xyzz)


def _g2_to_xyzz(point: G2Point) -> Array:
    """Convert a single G2Point to a JAX XYZZ point."""
    return jnp.array(bn254_g2_xyzz((point.x, point.y, 1, 1)), dtype=bn254_g2_xyzz)


def _g1_points_to_array(points: list[G1Point]) -> Array:
    """Convert a list of G1Points to a JAX array for MSM."""
    return jnp.array(
        [bn254_g1_affine((p.x, p.y)) for p in points],
        dtype=bn254_g1_affine,
    )


def _g2_points_to_array(points: list[G2Point]) -> Array:
    """Convert a list of G2Points to a JAX array for MSM."""
    return jnp.array(
        [bn254_g2_affine((p.x, p.y)) for p in points],
        dtype=bn254_g2_affine,
    )


def _ec_scalar_mul_g1(scalar: int, point: G1Point, msm: MSMBn254) -> Array:
    """Multiply a G1 point by a scalar via 1-element MSM."""
    s = jnp.array([scalar], dtype=bn254_sf)
    p = jnp.array([bn254_g1_affine((point.x, point.y))], dtype=bn254_g1_affine)
    return msm.compute(s, p)


def _ec_scalar_mul_g2(scalar: int, point: G2Point, msm: MSMBn254G2) -> Array:
    """Multiply a G2 point by a scalar via 1-element MSM."""
    s = jnp.array([scalar], dtype=bn254_sf)
    p = jnp.array([bn254_g2_affine((point.x, point.y))], dtype=bn254_g2_affine)
    return msm.compute(s, p)


def _scalar_mul_xyzz_g1(scalar: int, xyzz_point: Array, msm: MSMBn254) -> Array:
    """Multiply a G1 XYZZ point by a scalar (affine round-trip)."""
    affine = lax.convert_element_type(xyzz_point, bn254_g1_affine)
    s = jnp.array([scalar], dtype=bn254_sf)
    p = jnp.array([affine], dtype=bn254_g1_affine)
    return msm.compute(s, p)
