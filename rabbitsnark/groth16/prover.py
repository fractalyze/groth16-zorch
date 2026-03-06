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

"""Groth16 prover implementation — compile + prove split.

Separates one-time circuit compilation from per-proof computation:

    compiled = compile(zkey)                         # parse zkey (one-time)
    proof, signals = compiled.prove(wtns, az, bz)    # generate proof (per-witness)

Architecture:
    compile(zkey) -> CompiledProver
    |-- NTT twiddle arrays
    |-- Coset shift powers
    |-- Point arrays (affine -> XYZZ)
    |-- EC zero pre-allocation
    +-- Delta points

    CompiledProver.prove(wtns, az_mont, bz_mont)
    |-- Input preparation: z_std, r, s, neg_rs  (per-proof)
    +-- _prove_core(config, ...)  <- single JIT dispatch
       |-- Cz = Az ⊙ Bz (Hadamard)
       |-- IFFT x 3 (stage twiddles via static strided slicing)
       |-- Coset NTT x 3
       |-- Quotient: h_evals_mont = a_coset * b_coset - c_coset
       |-- h_evals_std = bitcast_convert_type(h_evals_mont, bn254_sf)
       |-- Scalar decomposition via _decompose_scalars_jit
       |  (bitcast bn254_sf -> uint8 -> int32 window indices)
       |-- MSMs 1-5
       |-- EC assembly + ZK blinding
       +-- XYZZ -> Affine
"""

from __future__ import annotations

import math
import secrets
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, NamedTuple

import jax
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

from rabbitsnark.msm import (
    _affine_to_xyzz,
    _decompose_scalars_jit,
    _ec_zeros,
    _estimate_optimal_window_bits,
    _pippenger_msm,
    _to_xyzz_dtype,
)
from rabbitsnark.ntt import BN254_FR_ROOT_OF_UNITY, NTT

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
BN254_SCALAR_BITS = 254


class ProveConfig(NamedTuple):
    """Static configuration for _prove_core (compile-time constants)."""

    # Arithmetic
    log_n: int
    # Scalar decomposition / MSM
    scalar_bits: int
    wb_main: int  # window_bits for MSMs 1-4
    wb_h: int  # window_bits for MSM 5 (h_evals)
    wb_1elem: int  # window_bits for 1-element ZK MSMs
    num_public: int  # l -- for z[l+1:m] slicing


@dataclass
class CompiledProver:
    """Pre-compiled proving key -- reusable across proofs.

    Created by ``compile(zkey)``. Call ``prove(wtns, az_mont, bz_mont)``
    to generate proofs.
    """

    config: ProveConfig
    # NTT arrays (per-stage twiddles, pre-extracted)
    fwd_stage_twiddles: tuple[Array, ...]
    inv_stage_twiddles: tuple[Array, ...]
    inv_n: Array
    shift_powers: Array
    # Point arrays (XYZZ)
    pa1_xyzz: Array
    pb1_xyzz: Array
    pb2_xyzz: Array
    pc1_xyzz: Array
    ph1_xyzz: Array
    # VK points
    alpha1_xyzz: Array
    beta1_xyzz: Array
    beta2_xyzz: Array
    # EC zeros
    g1_zero: Array
    g2_zero: Array
    g1_ws_main: Array
    g1_bk_main: Array
    g2_ws_main: Array
    g2_bk_main: Array
    g1_ws_h: Array
    g1_bk_h: Array
    g1_ws_1elem: Array
    g1_bk_1elem: Array
    g2_ws_1elem: Array
    g2_bk_1elem: Array
    # Delta points
    delta_g1_xyzz: Array
    delta_g2_xyzz: Array

    def prove(
        self,
        wtns: WtnsV2,
        az_mont: Array,
        bz_mont: Array,
        *,
        no_zk: bool = False,
    ) -> tuple[Groth16Proof, list[str]]:
        """Generate a Groth16 proof from a witness.

        Args:
            wtns: Parsed witness (WtnsV2).
            az_mont: Pre-computed A*z in Montgomery form (bn254_sf_mont).
            bz_mont: Pre-computed B*z in Montgomery form (bn254_sf_mont).
            no_zk: If True, use r=s=0 for a deterministic (non-ZK) proof.

        Returns:
            Tuple of (proof, public_signals).
        """
        z_std = jnp.array([int(w) for w in wtns.witnesses], dtype=bn254_sf)

        if no_zk:
            r_int, s_int = 0, 0
        else:
            r_int = secrets.randbelow(BN254_FR_MODULUS)
            s_int = secrets.randbelow(BN254_FR_MODULUS)

        r_arr = jnp.array([r_int], dtype=bn254_sf)
        s_arr = jnp.array([s_int], dtype=bn254_sf)
        neg_rs = -(bn254_sf(r_int) * bn254_sf(s_int))
        neg_rs_arr = jnp.array([neg_rs], dtype=bn254_sf)

        pi_a, pi_b2, pi_c = _prove_core(
            self.config,
            z_std,
            r_arr,
            s_arr,
            neg_rs_arr,
            az_mont,
            bz_mont,
            # NTT arrays (per-stage twiddles)
            self.fwd_stage_twiddles,
            self.inv_stage_twiddles,
            self.inv_n,
            self.shift_powers,
            # Point arrays (XYZZ)
            self.pa1_xyzz,
            self.pb1_xyzz,
            self.pb2_xyzz,
            self.pc1_xyzz,
            self.ph1_xyzz,
            # VK points
            self.alpha1_xyzz,
            self.beta1_xyzz,
            self.beta2_xyzz,
            # EC zeros
            self.g1_zero,
            self.g2_zero,
            self.g1_ws_main,
            self.g1_bk_main,
            self.g2_ws_main,
            self.g2_bk_main,
            self.g1_ws_h,
            self.g1_bk_h,
            self.g1_ws_1elem,
            self.g1_bk_1elem,
            self.g2_ws_1elem,
            self.g2_bk_1elem,
            # Delta points
            self.delta_g1_xyzz,
            self.delta_g2_xyzz,
        )

        proof = Groth16Proof(pi_a=pi_a, pi_b=pi_b2, pi_c=pi_c)
        public_signals = write_public_signals(wtns.witnesses, self.config.num_public)

        return proof, public_signals


def compile(zkey: ZKeyV1) -> CompiledProver:
    """Compile a proving key into a reusable prover.

    Pre-computes all circuit-constant data (NTT twiddles, point arrays,
    EC zeros, delta points).  The returned ``CompiledProver`` can generate
    multiple proofs via ``prove(wtns, az_mont, bz_mont)``.

    Args:
        zkey: Parsed proving key (ZKeyV1).

    Returns:
        Compiled prover ready for proof generation.
    """
    num_vars = zkey.header_groth.num_vars
    num_public = zkey.header_groth.num_public_inputs
    domain_size = zkey.domain_size
    log_n = int(math.log2(domain_size))
    vk = zkey.verifying_key

    m = num_vars

    # NTT per-stage twiddle arrays
    ntt = NTT(bn254_sf_mont, BN254_FR_ROOT_OF_UNITY)
    fwd_stage_twiddles, inv_stage_twiddles, inv_n = ntt.get_stage_twiddles(log_n)

    # Coset shift powers: [1, g, g², ..., g^(n-1)] where g = omega_{2n}
    # Pre-computed outside JIT (256-bit constants can't be created during JIT)
    coset_shift = jnp.array(
        bn254_sf_mont(BN254_FR_ROOT_OF_UNITY)
        ** (1 << (BN254_TWO_ADIC_BITS - log_n - 1)),
        dtype=bn254_sf_mont,
    )
    shift_powers = _build_shift_powers(coset_shift, log_n)

    # Point arrays: affine -> XYZZ (for MSM)
    g1_xyzz_dtype = _to_xyzz_dtype(bn254_g1_affine)
    g2_xyzz_dtype = _to_xyzz_dtype(bn254_g2_affine)

    pa1_xyzz = _affine_to_xyzz(_g1_points_to_array(zkey.points_a1), g1_xyzz_dtype)
    pb1_xyzz = _affine_to_xyzz(_g1_points_to_array(zkey.points_b1), g1_xyzz_dtype)
    pb2_xyzz = _affine_to_xyzz(_g2_points_to_array(zkey.points_b2), g2_xyzz_dtype)
    pc1_xyzz = _affine_to_xyzz(_g1_points_to_array(zkey.points_c1), g1_xyzz_dtype)
    ph1_xyzz = _affine_to_xyzz(_g1_points_to_array(zkey.points_h1), g1_xyzz_dtype)

    # VK points -> XYZZ
    alpha1_xyzz = _g1_to_xyzz(vk.alpha_g1)
    beta1_xyzz = _g1_to_xyzz(vk.beta_g1)
    beta2_xyzz = _g2_to_xyzz(vk.beta_g2)

    # Window bits estimation
    wb_main = _estimate_optimal_window_bits(BN254_SCALAR_BITS, m)
    wb_h = _estimate_optimal_window_bits(BN254_SCALAR_BITS, domain_size)
    wb_1elem = min(16, BN254_SCALAR_BITS)

    # EC zeros for MSM (pre-allocated, not baked as JIT constants)
    num_bk_main = (1 << wb_main) - 1
    num_bk_h = (1 << wb_h) - 1
    num_bk_1elem = (1 << wb_1elem) - 1
    num_w_main = (BN254_SCALAR_BITS + wb_main - 1) // wb_main
    num_w_h = (BN254_SCALAR_BITS + wb_h - 1) // wb_h
    num_w_1elem = (BN254_SCALAR_BITS + wb_1elem - 1) // wb_1elem

    g1_zero = _ec_zeros((), g1_xyzz_dtype)
    g2_zero = _ec_zeros((), g2_xyzz_dtype)
    g1_ws_main = _ec_zeros(num_w_main, g1_xyzz_dtype)
    g1_bk_main = _ec_zeros(num_bk_main, g1_xyzz_dtype)
    g2_ws_main = _ec_zeros(num_w_main, g2_xyzz_dtype)
    g2_bk_main = _ec_zeros(num_bk_main, g2_xyzz_dtype)
    g1_ws_h = _ec_zeros(num_w_h, g1_xyzz_dtype)
    g1_bk_h = _ec_zeros(num_bk_h, g1_xyzz_dtype)
    g1_ws_1elem = _ec_zeros(num_w_1elem, g1_xyzz_dtype)
    g1_bk_1elem = _ec_zeros(num_bk_1elem, g1_xyzz_dtype)
    g2_ws_1elem = _ec_zeros(num_w_1elem, g2_xyzz_dtype)
    g2_bk_1elem = _ec_zeros(num_bk_1elem, g2_xyzz_dtype)

    # Delta points for ZK blinding (1-element XYZZ arrays)
    delta_g1_xyzz = _affine_to_xyzz(
        jnp.array(
            [bn254_g1_affine((vk.delta_g1.x, vk.delta_g1.y))],
            dtype=bn254_g1_affine,
        ),
        g1_xyzz_dtype,
    )
    delta_g2_xyzz = _affine_to_xyzz(
        jnp.array(
            [bn254_g2_affine((vk.delta_g2.x, vk.delta_g2.y))],
            dtype=bn254_g2_affine,
        ),
        g2_xyzz_dtype,
    )

    config = ProveConfig(
        log_n=log_n,
        scalar_bits=BN254_SCALAR_BITS,
        wb_main=wb_main,
        wb_h=wb_h,
        wb_1elem=wb_1elem,
        num_public=num_public,
    )

    return CompiledProver(
        config=config,
        fwd_stage_twiddles=fwd_stage_twiddles,
        inv_stage_twiddles=inv_stage_twiddles,
        inv_n=inv_n,
        shift_powers=shift_powers,
        pa1_xyzz=pa1_xyzz,
        pb1_xyzz=pb1_xyzz,
        pb2_xyzz=pb2_xyzz,
        pc1_xyzz=pc1_xyzz,
        ph1_xyzz=ph1_xyzz,
        alpha1_xyzz=alpha1_xyzz,
        beta1_xyzz=beta1_xyzz,
        beta2_xyzz=beta2_xyzz,
        g1_zero=g1_zero,
        g2_zero=g2_zero,
        g1_ws_main=g1_ws_main,
        g1_bk_main=g1_bk_main,
        g2_ws_main=g2_ws_main,
        g2_bk_main=g2_bk_main,
        g1_ws_h=g1_ws_h,
        g1_bk_h=g1_bk_h,
        g1_ws_1elem=g1_ws_1elem,
        g1_bk_1elem=g1_bk_1elem,
        g2_ws_1elem=g2_ws_1elem,
        g2_bk_1elem=g2_bk_1elem,
        delta_g1_xyzz=delta_g1_xyzz,
        delta_g2_xyzz=delta_g2_xyzz,
    )


# ---------------------------------------------------------------------------
# Single JIT dispatch: Arithmetic + Decomposition + MSMs
# ---------------------------------------------------------------------------


@partial(jax.jit, static_argnums=(0,))
def _prove_core(
    config: ProveConfig,
    z_std: Array,
    r_arr: Array,
    s_arr: Array,
    neg_rs_arr: Array,
    az_mont: Array,
    bz_mont: Array,
    # NTT arrays (per-stage twiddles)
    fwd_stage_tw: tuple[Array, ...],
    inv_stage_tw: tuple[Array, ...],
    inv_n: Array,
    shift_powers: Array,
    # Point arrays (XYZZ)
    pa1: Array,
    pb1: Array,
    pb2: Array,
    pc1: Array,
    ph1: Array,
    # VK points
    alpha1: Array,
    beta1: Array,
    beta2: Array,
    # EC zeros
    g1_zero: Array,
    g2_zero: Array,
    g1_ws_main: Array,
    g1_bk_main: Array,
    g2_ws_main: Array,
    g2_bk_main: Array,
    g1_ws_h: Array,
    g1_bk_h: Array,
    g1_ws_1elem: Array,
    g1_bk_1elem: Array,
    g2_ws_1elem: Array,
    g2_bk_1elem: Array,
    # Delta points
    delta_g1: Array,
    delta_g2: Array,
) -> tuple[Array, Array, Array]:
    """Single JIT kernel: Hadamard + NTT + Decomposition + MSMs -> Affine.

    Accepts pre-computed Az, Bz (Montgomery form) from external SpMV.
    ZK blinding is always computed.  When r=s=0 the blinding MSMs produce
    identity (zero windows -> Pippenger returns identity) so the result is
    equivalent to a non-ZK proof with no compile-time branching.
    """
    log_n = config.log_n
    sb = config.scalar_bits
    wb_main = config.wb_main
    wb_h = config.wb_h
    wb_1elem = config.wb_1elem
    l = config.num_public  # noqa: E741

    # ---------------------------------------------------------------
    # Stage 1: Arithmetic (Hadamard + NTT + quotient) in Montgomery form
    # ---------------------------------------------------------------

    # Hadamard: Cz = Az ⊙ Bz
    cz = az_mont * bz_mont

    # IFFT x 3
    a_poly = NTT.inverse_ntt(az_mont, inv_n, log_n, *inv_stage_tw)
    b_poly = NTT.inverse_ntt(bz_mont, inv_n, log_n, *inv_stage_tw)
    c_poly = NTT.inverse_ntt(cz, inv_n, log_n, *inv_stage_tw)

    # Coset NTT x 3 (shift_powers pre-computed outside JIT)
    a_coset = NTT.forward_ntt(a_poly * shift_powers, log_n, *fwd_stage_tw)
    b_coset = NTT.forward_ntt(b_poly * shift_powers, log_n, *fwd_stage_tw)
    c_coset = NTT.forward_ntt(c_poly * shift_powers, log_n, *fwd_stage_tw)

    # Quotient: h = a * b - c on coset
    h_evals_mont = a_coset * b_coset - c_coset

    # ---------------------------------------------------------------
    # Stage 2: Scalar decomposition (all inside JIT)
    # ---------------------------------------------------------------
    h_evals_std = lax.bitcast_convert_type(h_evals_mont, bn254_sf)

    wi_z_main = _decompose_scalars_jit(z_std, sb, wb_main)
    wi_zw = _decompose_scalars_jit(z_std[l + 1 :], sb, wb_main)
    wi_h = _decompose_scalars_jit(h_evals_std, sb, wb_h)

    wi_r = _decompose_scalars_jit(r_arr, sb, wb_1elem)
    wi_s = _decompose_scalars_jit(s_arr, sb, wb_1elem)
    wi_neg_rs = _decompose_scalars_jit(neg_rs_arr, sb, wb_1elem)

    # ---------------------------------------------------------------
    # Stage 3: MSMs + EC assembly + ZK blinding + affine conversion
    # ---------------------------------------------------------------

    # MSMs 1-5
    msm_1 = _pippenger_msm(wi_z_main, pa1, g1_zero, g1_ws_main, g1_bk_main, sb, wb_main)
    msm_2 = _pippenger_msm(wi_z_main, pb1, g1_zero, g1_ws_main, g1_bk_main, sb, wb_main)
    msm_3 = _pippenger_msm(wi_z_main, pb2, g2_zero, g2_ws_main, g2_bk_main, sb, wb_main)
    msm_4 = _pippenger_msm(wi_zw, pc1, g1_zero, g1_ws_main, g1_bk_main, sb, wb_main)
    msm_5 = _pippenger_msm(wi_h, ph1, g1_zero, g1_ws_h, g1_bk_h, sb, wb_h)

    # EC assembly
    pi_a = alpha1 + msm_1
    pi_b1 = beta1 + msm_2
    pi_b2 = beta2 + msm_3
    pi_c = msm_4 + msm_5

    # ZK blinding (identity when r=s=0)
    # r*delta_1, s*delta_2
    r_delta1 = _pippenger_msm(
        wi_r,
        delta_g1,
        g1_zero,
        g1_ws_1elem,
        g1_bk_1elem,
        sb,
        wb_1elem,
    )
    s_delta2 = _pippenger_msm(
        wi_s,
        delta_g2,
        g2_zero,
        g2_ws_1elem,
        g2_bk_1elem,
        sb,
        wb_1elem,
    )

    # s*pi_A and r*pi_B1 on UNBLINDED values (no affine round-trip)
    s_pi_a = _pippenger_msm(
        wi_s,
        jnp.expand_dims(pi_a, 0),
        g1_zero,
        g1_ws_1elem,
        g1_bk_1elem,
        sb,
        wb_1elem,
    )
    r_pi_b1 = _pippenger_msm(
        wi_r,
        jnp.expand_dims(pi_b1, 0),
        g1_zero,
        g1_ws_1elem,
        g1_bk_1elem,
        sb,
        wb_1elem,
    )

    # -(rs)*delta_1
    neg_rs_delta1 = _pippenger_msm(
        wi_neg_rs,
        delta_g1,
        g1_zero,
        g1_ws_1elem,
        g1_bk_1elem,
        sb,
        wb_1elem,
    )

    pi_a = pi_a + r_delta1
    pi_b2 = pi_b2 + s_delta2
    pi_c = pi_c + s_pi_a + r_pi_b1 + neg_rs_delta1

    # XYZZ -> Affine
    pi_a = lax.convert_element_type(pi_a, bn254_g1_affine)
    pi_b2 = lax.convert_element_type(pi_b2, bn254_g2_affine)
    pi_c = lax.convert_element_type(pi_c, bn254_g1_affine)

    return pi_a, pi_b2, pi_c


# ---------------------------------------------------------------------------
# JIT-internal helpers (called during trace of _prove_core)
# ---------------------------------------------------------------------------


def _build_shift_powers(shift: Array, log_n: int) -> Array:
    """Build coset shift powers [1, g, g², ..., g^(n-1)] via O(log n) doubling."""
    dtype = shift.dtype
    one = dtype.type(1)
    powers = jnp.array([one], dtype=dtype)
    step_mul = shift
    for _ in range(log_n):
        powers = jnp.concatenate([powers, powers * step_mul])
        step_mul = step_mul * step_mul
    return powers


# ---------------------------------------------------------------------------
# Preprocessing helpers (called outside JIT)
# ---------------------------------------------------------------------------


def _g1_to_xyzz(point: G1Point) -> Array:
    """Convert a single G1Point to a JAX XYZZ scalar."""
    return jnp.array(bn254_g1_xyzz((point.x, point.y, 1, 1)), dtype=bn254_g1_xyzz)


def _g2_to_xyzz(point: G2Point) -> Array:
    """Convert a single G2Point to a JAX XYZZ scalar."""
    return jnp.array(bn254_g2_xyzz((point.x, point.y, 1, 1)), dtype=bn254_g2_xyzz)


def _g1_points_to_array(points: list[G1Point]) -> Array:
    """Convert a list of G1Points to a JAX affine array."""
    return jnp.array(
        [bn254_g1_affine((p.x, p.y)) for p in points],
        dtype=bn254_g1_affine,
    )


def _g2_points_to_array(points: list[G2Point]) -> Array:
    """Convert a list of G2Points to a JAX affine array."""
    return jnp.array(
        [bn254_g2_affine((p.x, p.y)) for p in points],
        dtype=bn254_g2_affine,
    )
