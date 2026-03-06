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
       |-- Scalar decomposition via MSM.decompose_scalars
       |  (bitcast bn254_sf -> uint8 -> int32 window indices)
       |-- MSMs 1-5
       |-- EC assembly + ZK blinding
       +-- XYZZ -> Affine
"""

from __future__ import annotations

import math
import os
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

from rabbitsnark.msm import MSM
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
    num_public: int  # l -- for z[l+1:m] slicing
    # Data-level parallelism
    num_parts_main: int  # partitions for MSMs 1-3 (N=m)
    chunk_size_main: int
    num_parts_w: int  # partitions for MSM 4 (N=m-l-1)
    chunk_size_w: int
    num_parts_h: int  # partitions for MSM 5 (N=domain_size)
    chunk_size_h: int


@dataclass
class CompiledProver:
    """Pre-compiled proving key -- reusable across proofs.

    Created by ``compile(zkey)``. Call ``prove(wtns)`` to generate proofs.
    """

    config: ProveConfig
    # NTT arrays (per-stage twiddles, pre-extracted)
    fwd_stage_twiddles: tuple[Array, ...]
    inv_stage_twiddles: tuple[Array, ...]
    inv_n: Array
    shift_powers: Array
    # Point arrays (XYZZ, padded for partition alignment)
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
    # 2D bucket/window_sums for parallel MSMs 1-3 (N=m)
    g1_bk_main: Array  # [P_main, num_bk_main]
    g2_bk_main: Array  # [P_main, num_bk_main]
    g1_ws_main: Array  # [P_main, num_w_main]
    g2_ws_main: Array  # [P_main, num_w_main]
    # 2D bucket/window_sums for parallel MSM 4 (N=m-l-1)
    g1_bk_w: Array  # [P_w, num_bk_main]
    g1_ws_w: Array  # [P_w, num_w_main]
    # 2D bucket/window_sums for parallel MSM 5 (N=domain_size)
    g1_bk_h: Array  # [P_h, num_bk_h]
    g1_ws_h: Array  # [P_h, num_w_h]
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
        neg_rs_int = (
            BN254_FR_MODULUS - (r_int * s_int) % BN254_FR_MODULUS
        ) % BN254_FR_MODULUS
        neg_rs_arr = jnp.array([neg_rs_int], dtype=bn254_sf)

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
            # Point arrays (XYZZ, padded)
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
            # 2D parallel MSM arrays
            self.g1_bk_main,
            self.g2_bk_main,
            self.g1_ws_main,
            self.g2_ws_main,
            self.g1_bk_w,
            self.g1_ws_w,
            self.g1_bk_h,
            self.g1_ws_h,
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
    p = BN254_FR_MODULUS

    # NTT per-stage twiddle arrays
    ntt = NTT(bn254_sf_mont, BN254_FR_ROOT_OF_UNITY)
    fwd_stage_twiddles, inv_stage_twiddles, inv_n = ntt.get_stage_twiddles(log_n)

    # Coset shift powers: [1, g, g^2, ..., g^(n-1)] where g = omega_{2n}
    # Pre-computed outside JIT (256-bit constants can't be created during JIT)
    omega_2n_int = pow(
        BN254_FR_ROOT_OF_UNITY,
        1 << (BN254_TWO_ADIC_BITS - log_n - 1),
        p,
    )
    coset_shift = jnp.array(omega_2n_int, dtype=bn254_sf_mont)
    shift_powers = _build_shift_powers(coset_shift, log_n)

    # Point arrays: affine -> XYZZ (for MSM)
    g1_xyzz_dtype = MSM.to_xyzz_dtype(bn254_g1_affine)
    g2_xyzz_dtype = MSM.to_xyzz_dtype(bn254_g2_affine)

    pa1_xyzz = MSM.affine_to_xyzz(_g1_points_to_array(zkey.points_a1), g1_xyzz_dtype)
    pb1_xyzz = MSM.affine_to_xyzz(_g1_points_to_array(zkey.points_b1), g1_xyzz_dtype)
    pb2_xyzz = MSM.affine_to_xyzz(_g2_points_to_array(zkey.points_b2), g2_xyzz_dtype)
    pc1_xyzz = MSM.affine_to_xyzz(_g1_points_to_array(zkey.points_c1), g1_xyzz_dtype)
    ph1_xyzz = MSM.affine_to_xyzz(_g1_points_to_array(zkey.points_h1), g1_xyzz_dtype)

    # VK points -> XYZZ
    alpha1_xyzz = _g1_to_xyzz(vk.alpha_g1)
    beta1_xyzz = _g1_to_xyzz(vk.beta_g1)
    beta2_xyzz = _g2_to_xyzz(vk.beta_g2)

    # Window bits estimation
    wb_main = MSM.estimate_optimal_window_bits(BN254_SCALAR_BITS, m)
    wb_h = MSM.estimate_optimal_window_bits(BN254_SCALAR_BITS, domain_size)

    num_bk_main = 1 << wb_main
    num_bk_h = 1 << wb_h
    num_w_main = (BN254_SCALAR_BITS + wb_main - 1) // wb_main
    num_w_h = (BN254_SCALAR_BITS + wb_h - 1) // wb_h

    # --- Data-level parallelism: partition config ---
    max_parts = os.cpu_count() or 1
    n_w = m - num_public - 1  # witness-only count for MSM 4

    # Minimum elements per partition to amortize scatter/reduce overhead.
    # Each partition generates W windows of scatter+fori_loop in HLO;
    # too many partitions on small inputs causes IR bloat and slow LLVM compile.
    min_chunk = 64

    p_main = max(1, min(m // min_chunk, max_parts))
    chunk_main = math.ceil(m / p_main)
    padded_main = chunk_main * p_main

    p_w = max(1, min(n_w // min_chunk, max_parts))
    chunk_w = math.ceil(n_w / p_w)
    padded_w = chunk_w * p_w

    p_h = max(1, min(domain_size // min_chunk, max_parts))
    chunk_h = math.ceil(domain_size / p_h)
    padded_h = chunk_h * p_h

    # Pad point arrays to partition-aligned sizes (zero-index scatters → dummy bucket)
    if padded_main > m:
        pad_g1 = MSM.ec_zeros(padded_main - m, g1_xyzz_dtype)
        pa1_xyzz = jnp.concatenate([pa1_xyzz, pad_g1])
        pb1_xyzz = jnp.concatenate([pb1_xyzz, pad_g1])
        pad_g2 = MSM.ec_zeros(padded_main - m, g2_xyzz_dtype)
        pb2_xyzz = jnp.concatenate([pb2_xyzz, pad_g2])
    if padded_w > n_w:
        pc1_xyzz = jnp.concatenate(
            [pc1_xyzz, MSM.ec_zeros(padded_w - n_w, g1_xyzz_dtype)]
        )
    if padded_h > domain_size:
        ph1_xyzz = jnp.concatenate(
            [ph1_xyzz, MSM.ec_zeros(padded_h - domain_size, g1_xyzz_dtype)]
        )

    # EC zeros
    g1_zero = MSM.ec_zeros((), g1_xyzz_dtype)
    g2_zero = MSM.ec_zeros((), g2_xyzz_dtype)

    # 2D bucket/window_sums for parallel MSMs
    g1_bk_main = MSM.ec_zeros((p_main, num_bk_main), g1_xyzz_dtype)
    g2_bk_main = MSM.ec_zeros((p_main, num_bk_main), g2_xyzz_dtype)
    g1_ws_main = MSM.ec_zeros((p_main, num_w_main), g1_xyzz_dtype)
    g2_ws_main = MSM.ec_zeros((p_main, num_w_main), g2_xyzz_dtype)
    g1_bk_w = MSM.ec_zeros((p_w, num_bk_main), g1_xyzz_dtype)
    g1_ws_w = MSM.ec_zeros((p_w, num_w_main), g1_xyzz_dtype)
    g1_bk_h = MSM.ec_zeros((p_h, num_bk_h), g1_xyzz_dtype)
    g1_ws_h = MSM.ec_zeros((p_h, num_w_h), g1_xyzz_dtype)

    # Delta points for ZK blinding (1-element XYZZ arrays)
    delta_g1_xyzz = MSM.affine_to_xyzz(
        jnp.array(
            [bn254_g1_affine((vk.delta_g1.x, vk.delta_g1.y))],
            dtype=bn254_g1_affine,
        ),
        g1_xyzz_dtype,
    )
    delta_g2_xyzz = MSM.affine_to_xyzz(
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
        num_public=num_public,
        num_parts_main=p_main,
        chunk_size_main=chunk_main,
        num_parts_w=p_w,
        chunk_size_w=chunk_w,
        num_parts_h=p_h,
        chunk_size_h=chunk_h,
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
        g1_bk_main=g1_bk_main,
        g2_bk_main=g2_bk_main,
        g1_ws_main=g1_ws_main,
        g2_ws_main=g2_ws_main,
        g1_bk_w=g1_bk_w,
        g1_ws_w=g1_ws_w,
        g1_bk_h=g1_bk_h,
        g1_ws_h=g1_ws_h,
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
    # Point arrays (XYZZ, padded for partition alignment)
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
    # 2D parallel MSM arrays
    g1_bk_main: Array,
    g2_bk_main: Array,
    g1_ws_main: Array,
    g2_ws_main: Array,
    g1_bk_w: Array,
    g1_ws_w: Array,
    g1_bk_h: Array,
    g1_ws_h: Array,
    # Delta points
    delta_g1: Array,
    delta_g2: Array,
) -> tuple[Array, Array, Array]:
    """Single JIT kernel: Hadamard + NTT + Decomposition + MSMs -> Affine.

    Accepts pre-computed Az, Bz (Montgomery form) from external SpMV.
    MSMs 1-5 use data-level parallelism via ``MSM.pippenger``:
    each MSM is split into P independent partitions that ThunkExecutor
    can schedule in parallel.  ZK blinding uses direct scalar
    multiplication via ``*`` operator.
    """
    log_n = config.log_n
    sb = config.scalar_bits
    wb_main = config.wb_main
    wb_h = config.wb_h
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

    wi_z_main = MSM.decompose_scalars(z_std, sb, wb_main)
    wi_zw = MSM.decompose_scalars(z_std[l + 1 :], sb, wb_main)
    wi_h = MSM.decompose_scalars(h_evals_std, sb, wb_h)

    # Pad window indices to partition-aligned sizes (zeros → bucket 0 dummy)
    pad_main = config.chunk_size_main * config.num_parts_main - wi_z_main.shape[1]
    if pad_main > 0:
        wi_z_main = jnp.pad(wi_z_main, ((0, 0), (0, pad_main)))
    pad_w = config.chunk_size_w * config.num_parts_w - wi_zw.shape[1]
    if pad_w > 0:
        wi_zw = jnp.pad(wi_zw, ((0, 0), (0, pad_w)))
    pad_h = config.chunk_size_h * config.num_parts_h - wi_h.shape[1]
    if pad_h > 0:
        wi_h = jnp.pad(wi_h, ((0, 0), (0, pad_h)))

    # ---------------------------------------------------------------
    # Stage 3: MSMs + EC assembly + ZK blinding + affine conversion
    # ---------------------------------------------------------------
    P_main = config.num_parts_main
    C_main = config.chunk_size_main
    P_w = config.num_parts_w
    C_w = config.chunk_size_w
    P_h = config.num_parts_h
    C_h = config.chunk_size_h

    # Parallel MSMs 1-5
    msm_1 = MSM.pippenger(
        wi_z_main,
        pa1,
        g1_zero,
        g1_bk_main,
        g1_ws_main,
        sb,
        wb_main,
        P_main,
        C_main,
    )
    msm_2 = MSM.pippenger(
        wi_z_main,
        pb1,
        g1_zero,
        g1_bk_main,
        g1_ws_main,
        sb,
        wb_main,
        P_main,
        C_main,
    )
    msm_3 = MSM.pippenger(
        wi_z_main,
        pb2,
        g2_zero,
        g2_bk_main,
        g2_ws_main,
        sb,
        wb_main,
        P_main,
        C_main,
    )
    msm_4 = MSM.pippenger(
        wi_zw,
        pc1,
        g1_zero,
        g1_bk_w,
        g1_ws_w,
        sb,
        wb_main,
        P_w,
        C_w,
    )
    msm_5 = MSM.pippenger(
        wi_h,
        ph1,
        g1_zero,
        g1_bk_h,
        g1_ws_h,
        sb,
        wb_h,
        P_h,
        C_h,
    )

    # EC assembly
    pi_a = alpha1 + msm_1
    pi_b1 = beta1 + msm_2
    pi_b2 = beta2 + msm_3
    pi_c = msm_4 + msm_5

    # ZK blinding (identity when r=s=0) — scalar multiplication via *
    r = r_arr[0]
    s = s_arr[0]
    neg_rs = neg_rs_arr[0]

    # s*pi_A and r*pi_B1 on UNBLINDED values (before blinding below)
    s_pi_a = s * pi_a
    r_pi_b1 = r * pi_b1

    pi_a = pi_a + r * delta_g1[0]
    pi_b2 = pi_b2 + s * delta_g2[0]
    pi_c = pi_c + s_pi_a + r_pi_b1 + neg_rs * delta_g1[0]

    # XYZZ -> Affine
    pi_a = lax.convert_element_type(pi_a, bn254_g1_affine)
    pi_b2 = lax.convert_element_type(pi_b2, bn254_g2_affine)
    pi_c = lax.convert_element_type(pi_c, bn254_g1_affine)

    return pi_a, pi_b2, pi_c


# ---------------------------------------------------------------------------
# JIT-internal helpers (called during trace of _prove_core)
# ---------------------------------------------------------------------------


def _build_shift_powers(shift: Array, log_n: int) -> Array:
    """Build coset shift powers [1, g, g^2, ..., g^(n-1)] via O(log n) doubling."""
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
