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

"""Groth16 prover implementation — two JIT dispatches.

Generates a Groth16 proof from a zkey (proving key) and wtns (witness),
executing the proof computation in two ``@jax.jit`` calls with a Python
scalar decomposition bridge between them (17+ dispatches → 2).

Architecture:
    prove(zkey, wtns)
    ├─ Input preparation: z_std, z_mont, r, s  (per-proof parameters)
    ├─ Circuit constants: R1CS ELL arrays, NTT twiddles, point arrays, ...
    │  (pre-computed once per circuit, reusable across proofs)
    │
    ├─ _prove_arith(config, z_mont, ...)                  ← JIT dispatch 1
    │  ├─ SpMV × 2, Hadamard
    │  ├─ IFFT × 3 (stage twiddles via static strided slicing)
    │  ├─ Coset NTT × 3
    │  └─ Quotient: h_evals = a_coset * b_coset - c_coset
    │
    ├─ Scalar decomposition (Python bridge):
    │  ├─ h_evals_mont → h_evals_std via int() extraction
    │  └─ _decompose_scalars for z, h, r, s, neg_rs
    │
    ├─ _prove_msm(config, wi_z, wi_zw, wi_h, ...)        ← JIT dispatch 2
    │  ├─ MSMs 1-5
    │  ├─ EC assembly + ZK blinding
    │  └─ XYZZ → Affine conversion
    │
    └─ Groth16Proof(pi_a, pi_b, pi_c)
"""

from __future__ import annotations

import math
import secrets
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
    _decompose_scalars,
    _ec_zeros,
    _estimate_optimal_window_bits,
    _pippenger_msm,
    _to_xyzz_dtype,
)
from rabbitsnark.ntt import BN254_FR_ROOT_OF_UNITY, NTT, _forward_ntt, _inverse_ntt
from rabbitsnark.spmv import build_r1cs_matrices, witness_to_montgomery
from rabbitsnark.spmv.spmv import _spmv_kernel

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


class ArithConfig(NamedTuple):
    """Static configuration for _prove_arith (compile-time constants)."""

    n_rows_a: int
    max_nnz_a: int
    n_rows_b: int
    max_nnz_b: int
    log_n: int


class MSMConfig(NamedTuple):
    """Static configuration for _prove_msm (compile-time constants)."""

    scalar_bits: int
    wb_main: int  # window_bits for MSMs 1-4
    wb_h: int  # window_bits for MSM 5 (h_evals)
    wb_1elem: int  # window_bits for 1-element ZK MSMs
    no_zk: bool


def prove(
    zkey: ZKeyV1,
    wtns: WtnsV2,
    *,
    no_zk: bool = False,
) -> tuple[Groth16Proof, list[str]]:
    """Generate a Groth16 proof via two JIT dispatches.

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
    log_n = int(math.log2(domain_size))
    vk = zkey.verifying_key

    l = num_public  # noqa: E741
    m = num_vars

    # --- Input preparation (per-proof parameters) ---
    z_std = jnp.array([int(w) for w in wtns.witnesses], dtype=bn254_sf)
    z_mont = witness_to_montgomery(wtns.witnesses, bn254_sf_mont, p)
    if no_zk:
        r_int, s_int = 0, 0
    else:
        r_int = secrets.randbelow(p)
        s_int = secrets.randbelow(p)

    # --- Circuit constants (pre-computed once per circuit) ---

    # R1CS matrices → ELL arrays
    matrix_a, matrix_b = build_r1cs_matrices(zkey, bn254_sf_mont)

    # NTT twiddle arrays
    ntt = NTT(bn254_sf_mont, BN254_FR_ROOT_OF_UNITY)
    fwd_roots, inv_roots, inv_n = ntt.get_twiddle_arrays(domain_size)

    # Coset shift powers: [1, g, g², ..., g^(n-1)] where g = omega_{2n}
    # Pre-computed outside JIT (256-bit constants can't be created during JIT)
    omega_2n_int = pow(
        BN254_FR_ROOT_OF_UNITY,
        1 << (BN254_TWO_ADIC_BITS - log_n - 1),
        p,
    )
    coset_shift = jnp.array(omega_2n_int, dtype=bn254_sf_mont)
    shift_powers = _build_shift_powers(coset_shift, log_n)

    # Point arrays: affine → XYZZ (for MSM)
    g1_xyzz_dtype = _to_xyzz_dtype(bn254_g1_affine)
    g2_xyzz_dtype = _to_xyzz_dtype(bn254_g2_affine)

    pa1_xyzz = _affine_to_xyzz(_g1_points_to_array(zkey.points_a1), g1_xyzz_dtype)
    pb1_xyzz = _affine_to_xyzz(_g1_points_to_array(zkey.points_b1), g1_xyzz_dtype)
    pb2_xyzz = _affine_to_xyzz(_g2_points_to_array(zkey.points_b2), g2_xyzz_dtype)
    pc1_xyzz = _affine_to_xyzz(_g1_points_to_array(zkey.points_c1), g1_xyzz_dtype)
    ph1_xyzz = _affine_to_xyzz(_g1_points_to_array(zkey.points_h1), g1_xyzz_dtype)

    # VK points → XYZZ
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

    # =================================================================
    # JIT dispatch 1: Arithmetic (SpMV + NTT + quotient)
    # =================================================================
    arith_config = ArithConfig(
        n_rows_a=matrix_a.n_rows,
        max_nnz_a=matrix_a.max_nnz_per_row,
        n_rows_b=matrix_b.n_rows,
        max_nnz_b=matrix_b.max_nnz_per_row,
        log_n=log_n,
    )

    h_evals_mont = _prove_arith(
        arith_config,
        z_mont,
        matrix_a.ell_col_indices,
        matrix_a.ell_values,
        matrix_b.ell_col_indices,
        matrix_b.ell_values,
        fwd_roots,
        inv_roots,
        inv_n,
        shift_powers,
    )

    # =================================================================
    # Python bridge: scalar decomposition
    # =================================================================
    h_evals_std = jnp.array([int(v) for v in h_evals_mont], dtype=bn254_sf)

    wi_z_main = _decompose_scalars(z_std[:m], BN254_SCALAR_BITS, wb_main)
    wi_zw = _decompose_scalars(z_std[l + 1 : m], BN254_SCALAR_BITS, wb_main)
    wi_h = _decompose_scalars(h_evals_std, BN254_SCALAR_BITS, wb_h)

    if not no_zk:
        r_arr = jnp.array([r_int], dtype=bn254_sf)
        s_arr = jnp.array([s_int], dtype=bn254_sf)
        neg_rs_int = (p - (r_int * s_int) % p) % p
        neg_rs_arr = jnp.array([neg_rs_int], dtype=bn254_sf)

        wi_r = _decompose_scalars(r_arr, BN254_SCALAR_BITS, wb_1elem)
        wi_s = _decompose_scalars(s_arr, BN254_SCALAR_BITS, wb_1elem)
        wi_neg_rs = _decompose_scalars(neg_rs_arr, BN254_SCALAR_BITS, wb_1elem)
    else:
        # Placeholders (not used when no_zk=True, but needed for call signature)
        wi_r = jnp.zeros((num_w_1elem, 1), dtype=jnp.int32)
        wi_s = jnp.zeros((num_w_1elem, 1), dtype=jnp.int32)
        wi_neg_rs = jnp.zeros((num_w_1elem, 1), dtype=jnp.int32)

    # =================================================================
    # JIT dispatch 2: MSMs + EC assembly + affine conversion
    # =================================================================
    msm_config = MSMConfig(
        scalar_bits=BN254_SCALAR_BITS,
        wb_main=wb_main,
        wb_h=wb_h,
        wb_1elem=wb_1elem,
        no_zk=no_zk,
    )

    pi_a, pi_b2, pi_c = _prove_msm(
        msm_config,
        # Window indices
        wi_z_main,
        wi_zw,
        wi_h,
        wi_r,
        wi_s,
        wi_neg_rs,
        # Point arrays (XYZZ)
        pa1_xyzz,
        pb1_xyzz,
        pb2_xyzz,
        pc1_xyzz,
        ph1_xyzz,
        # VK points
        alpha1_xyzz,
        beta1_xyzz,
        beta2_xyzz,
        # EC zeros
        g1_zero,
        g2_zero,
        g1_ws_main,
        g1_bk_main,
        g2_ws_main,
        g2_bk_main,
        g1_ws_h,
        g1_bk_h,
        g1_ws_1elem,
        g1_bk_1elem,
        g2_ws_1elem,
        g2_bk_1elem,
        # Delta points
        delta_g1_xyzz,
        delta_g2_xyzz,
    )

    proof = Groth16Proof(pi_a=pi_a, pi_b=pi_b2, pi_c=pi_c)
    public_signals = write_public_signals(wtns.witnesses, num_public)

    return proof, public_signals


# ---------------------------------------------------------------------------
# JIT dispatch 1: Arithmetic
# ---------------------------------------------------------------------------


@partial(jax.jit, static_argnums=(0,))
def _prove_arith(
    config: ArithConfig,
    z_mont: Array,
    # R1CS ELL arrays
    ell_col_a: Array,
    ell_val_a: Array,
    ell_col_b: Array,
    ell_val_b: Array,
    # NTT arrays
    fwd_roots: Array,
    inv_roots: Array,
    inv_n: Array,
    shift_powers: Array,
) -> Array:
    """JIT kernel 1: SpMV → Hadamard → IFFT → Coset NTT → Quotient.

    Returns h_evals in Montgomery form.
    """
    log_n = config.log_n
    n = 1 << log_n

    # SpMV: Az = A·z, Bz = B·z
    az = _spmv_kernel(ell_col_a, ell_val_a, z_mont, config.n_rows_a, config.max_nnz_a)
    bz = _spmv_kernel(ell_col_b, ell_val_b, z_mont, config.n_rows_b, config.max_nnz_b)

    # Hadamard: Cz = Az * Bz
    cz = az * bz

    # IFFT × 3
    inv_stage_tw = _extract_inv_stage_twiddles(inv_roots, n, log_n)
    a_poly = _inverse_ntt(az, inv_n, log_n, *inv_stage_tw)
    b_poly = _inverse_ntt(bz, inv_n, log_n, *inv_stage_tw)
    c_poly = _inverse_ntt(cz, inv_n, log_n, *inv_stage_tw)

    # Coset NTT × 3 (shift_powers pre-computed outside JIT)
    fwd_stage_tw = _extract_fwd_stage_twiddles(fwd_roots, n, log_n)
    a_coset = _forward_ntt(a_poly * shift_powers, log_n, *fwd_stage_tw)
    b_coset = _forward_ntt(b_poly * shift_powers, log_n, *fwd_stage_tw)
    c_coset = _forward_ntt(c_poly * shift_powers, log_n, *fwd_stage_tw)

    # Quotient: h = a * b - c on coset
    return a_coset * b_coset - c_coset


# ---------------------------------------------------------------------------
# JIT dispatch 2: MSMs + EC assembly
# ---------------------------------------------------------------------------


@partial(jax.jit, static_argnums=(0,))
def _prove_msm(
    config: MSMConfig,
    # Window indices (int32)
    wi_z_main: Array,
    wi_zw: Array,
    wi_h: Array,
    wi_r: Array,
    wi_s: Array,
    wi_neg_rs: Array,
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
    """JIT kernel 2: MSMs → EC assembly → ZK blinding → affine conversion."""
    sb = config.scalar_bits
    wb_main = config.wb_main
    wb_h = config.wb_h
    wb_1elem = config.wb_1elem

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

    # ZK blinding (compile-time branch via static config)
    if not config.no_zk:
        # r·δ₁, s·δ₁, s·δ₂
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

        # s·π_A and r·π_B₁ on UNBLINDED values (no affine round-trip)
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

        # -(rs)·δ₁
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

    # XYZZ → Affine
    pi_a = lax.convert_element_type(pi_a, bn254_g1_affine)
    pi_b2 = lax.convert_element_type(pi_b2, bn254_g2_affine)
    pi_c = lax.convert_element_type(pi_c, bn254_g1_affine)

    return pi_a, pi_b2, pi_c


# ---------------------------------------------------------------------------
# JIT-internal helpers (called during trace of _prove_arith)
# ---------------------------------------------------------------------------


def _extract_fwd_stage_twiddles(roots: Array, n: int, log_n: int) -> tuple[Array, ...]:
    """Extract per-stage forward twiddles via static strided slicing.

    Stage s needs roots[::stride][:half_m] where stride = n // 2^(s+1).
    Uses Slice HLO, not gather.
    """
    stage_twiddles = []
    for s in range(log_n):
        half_m = 1 << s
        stride = n // (2 * half_m)
        tw = roots[::stride][:half_m]
        stage_twiddles.append(tw)
    return tuple(stage_twiddles)


def _extract_inv_stage_twiddles(
    inv_roots: Array, n: int, log_n: int
) -> tuple[Array, ...]:
    """Extract per-stage inverse twiddles via static strided slicing."""
    stage_twiddles = []
    for s in range(log_n):
        actual_stage = log_n - 1 - s
        half_m = 1 << actual_stage
        stride = n // (2 * half_m)
        tw = inv_roots[::stride][:half_m]
        stage_twiddles.append(tw)
    return tuple(stage_twiddles)


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
