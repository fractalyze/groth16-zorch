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

"""Groth16 prover implementation — compile + prove.

Separates one-time circuit compilation from per-proof computation:

    compiled = compile_circom(zkey)                    # parse zkey (one-time)
    proof, signals = compiled.prove_circom(wtns, az, bz)  # per-witness

Architecture:
    compile_circom(zkey) -> CompiledProver
    |-- NTT twiddle arrays
    |-- Coset shift powers
    |-- Point arrays (affine)
    +-- VK / Delta points (affine scalars)

    CompiledProver.prove_circom(wtns, az_mont, bz_mont)
    |-- Input preparation: z_std, r, s, r*s  (per-proof)
    +-- Phase 1: _prove_ntt(config, ...)  <- JIT
    |  |-- Cz = Az ⊙ Bz (Hadamard)
    |  |-- IFFT x 3 → Coset NTT x 3
    |  |-- Quotient: h = a * b - c on coset
    |  +-- h-polynomial → scalars for MSM
    +-- Phase 2: 5x MSM via lax.msm (GPU memory managed by MsmChunkSplit)
    +-- Phase 3: _prove_phase3(...)  <- JIT (CPU-only)
       |-- EC assembly + ZK blinding via scalar *
       +-- Convert to affine

    Coset generator:
      circom: omega_{2n} (primitive 2n-th root of unity)
      gnark:  Fr multiplicative generator = 5
"""

from __future__ import annotations

import math
import secrets
from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from zk_dtypes import (
    bn254_g1_affine,
    bn254_g1_jacobian,
    bn254_g2_affine,
    bn254_g2_jacobian,
    bn254_sf,
    bn254_sf_mont,
)

from rabbitsnark.ntt import BN254_FR_ROOT_OF_UNITY, NTT

from .proof import Groth16Proof, write_public_signals  # noqa: F401

if TYPE_CHECKING:
    import numpy as np
    from jax import Array

    from rabbitsnark.circom.wtns.wtns import WtnsV2
    from rabbitsnark.circom.zkey.verifying_key import G1Point, G2Point
    from rabbitsnark.circom.zkey.zkey import ZKeyV1
    from rabbitsnark.gnark.types import GnarkProvingData

BN254_TWO_ADIC_BITS = 28
BN254_FR_MODULUS = (
    21888242871839275222246405745257275088548364400416034343698204186575808495617
)
# gnark uses the Fr multiplicative generator (= 5) as coset shift,
# NOT omega_{2n} which circom/snarkjs uses.
GNARK_COSET_GEN = 5

# Fixed non-zero blinding factors for deterministic benchmarking.
# Arbitrary values; chosen so that ZK blinding EC scalar multiplies execute
# with realistic non-zero operands (unlike no_zk where r=s=0 skips work).
_DETERMINISTIC_R = 7
_DETERMINISTIC_S = 11
# gnark uses 5^((p - 1) / 2²⁸) as the primitive 2²⁸-th root of unity.
# BN254_FR_ROOT_OF_UNITY (from the NTT module) uses generator 7 instead of 5,
# giving a different primitive root.  The prover MUST use the same root as the
# CRS / proving key to get correct h-polynomial coefficients.
GNARK_FR_ROOT_OF_UNITY = (
    19103219067921713944291392827692070036145651957329286315305642004821462161904
)


class ProveConfig(NamedTuple):
    """Static configuration for proving (compile-time constants)."""

    log_n: int
    num_public: int  # l -- for z[l+1:m] slicing
    is_circom: bool = True


@dataclass
class CompiledProver:
    """Pre-compiled proving key -- reusable across proofs.

    Created by ``compile_circom(zkey)`` or ``compile_gnark(data)``.
    Call ``prove_circom()`` or ``prove_gnark()`` to generate proofs.
    """

    config: ProveConfig
    # NTT arrays (per-stage twiddles, pre-extracted)
    fwd_stage_twiddles: tuple[Array, ...]
    inv_stage_twiddles: tuple[Array, ...]
    inv_n: Array
    shift_powers: Array
    # Point arrays (affine)
    pa1: Array  # bn254_g1_affine [m]
    pb1: Array  # bn254_g1_affine [m]
    pb2: Array  # bn254_g2_affine [m]
    pc1: Array  # bn254_g1_affine [m-l-1]
    ph1: Array  # bn254_g1_affine [domain_size] (circom) or [domain_size-1] (gnark)
    # VK points (affine scalars)
    alpha1: Array  # bn254_g1_affine scalar
    beta1: Array  # bn254_g1_affine scalar
    beta2: Array  # bn254_g2_affine scalar
    # Delta points (affine scalars)
    delta_g1: Array  # bn254_g1_affine scalar
    delta_g2: Array  # bn254_g2_affine scalar
    # Gnark-specific (None for circom — not computed)
    den: Array | None = None  # 1 / (g^n - 1), scalar bn254_sf_mont
    inv_shift_powers: Array | None = None  # [1, g⁻¹, g⁻², ...] bn254_sf_mont [n]

    def _run_prove(
        self,
        z_std: Array,
        az_mont: Array,
        bz_mont: Array,
        r_val: Array,
        s_val: Array,
        rs_val: Array,
    ) -> tuple[Array, Array, Array]:
        """Run the proving computation.

        Phase 1+2 (NTT + MSMs) runs on the default device (GPU if available).
        Phase 3 (EC assembly) always runs on CPU because EC scalar multiply
        fusions create kernels with ~68KB stack frames that cause
        CUDA_ERROR_OUT_OF_MEMORY on GPU.
        """
        # Gnark-specific arrays: None for circom → dummy scalar placeholder
        # (never traced thanks to config.is_circom static branching)
        den = self.den if self.den is not None else jnp.array(0, dtype=bn254_sf_mont)
        inv_sp = (
            self.inv_shift_powers
            if self.inv_shift_powers is not None
            else jnp.array(0, dtype=bn254_sf_mont)
        )

        # Phase 1: NTT + h-polynomial (JIT).
        h_scalars = _prove_ntt(
            self.config,
            az_mont,
            bz_mont,
            self.fwd_stage_twiddles,
            self.inv_stage_twiddles,
            self.inv_n,
            self.shift_powers,
            den,
            inv_sp,
        )
        # Phase 2: MSMs (MsmChunkSplit in ZKX handles GPU memory management).
        l = self.config.num_public  # noqa: E741
        private_start = l + 1 if self.config.is_circom else l
        msm_1 = lax.msm(z_std, self.pa1)
        msm_2 = lax.msm(z_std, self.pb1)
        msm_3 = lax.msm(z_std, self.pb2)
        msm_4 = lax.msm(z_std[private_start:], self.pc1)
        msm_5 = lax.msm(h_scalars, self.ph1)
        # Phase 3: EC assembly on CPU.
        cpu = jax.devices("cpu")[0]
        _to_cpu = lambda x: jnp.array(np.array(x), dtype=x.dtype)
        with jax.default_device(cpu):
            return _prove_phase3(
                _to_cpu(msm_1),
                _to_cpu(msm_2),
                _to_cpu(msm_3),
                _to_cpu(msm_4),
                _to_cpu(msm_5),
                _to_cpu(r_val),
                _to_cpu(s_val),
                _to_cpu(rs_val),
                _to_cpu(self.alpha1),
                _to_cpu(self.beta1),
                _to_cpu(self.beta2),
                _to_cpu(self.delta_g1),
                _to_cpu(self.delta_g2),
            )

    def prove_circom(
        self,
        wtns: WtnsV2,
        az_mont: Array,
        bz_mont: Array,
        *,
        no_zk: bool = False,
        deterministic: bool = False,
    ) -> tuple[Groth16Proof, list[str]]:
        """Generate a Groth16 proof from a witness.

        Args:
            wtns: Parsed witness (WtnsV2).
            az_mont: Pre-computed A*z in Montgomery form (bn254_sf_mont).
            bz_mont: Pre-computed B*z in Montgomery form (bn254_sf_mont).
            no_zk: If True, use r=s=0 (no ZK blinding, eliminates EC muls).
            deterministic: If True, use fixed non-zero r, s for reproducible
                proofs that still exercise full ZK blinding computation.

        Returns:
            Tuple of (proof, public_signals).
        """
        z_std = jnp.array([int(w) for w in wtns.witnesses], dtype=bn254_sf)

        if no_zk:
            r_int, s_int = 0, 0
        elif deterministic:
            r_int, s_int = _DETERMINISTIC_R, _DETERMINISTIC_S
        else:
            r_int = secrets.randbelow(BN254_FR_MODULUS)
            s_int = secrets.randbelow(BN254_FR_MODULUS)

        r_val = jnp.array(r_int, dtype=bn254_sf)
        s_val = jnp.array(s_int, dtype=bn254_sf)
        rs = bn254_sf(r_int) * bn254_sf(s_int)
        rs_val = jnp.array(rs, dtype=bn254_sf)

        pi_a, pi_b2, pi_c = self._run_prove(
            z_std,
            az_mont,
            bz_mont,
            r_val,
            s_val,
            rs_val,
        )

        proof = Groth16Proof(pi_a=pi_a, pi_b=pi_b2, pi_c=pi_c)
        public_signals = write_public_signals(wtns.witnesses, self.config.num_public)

        return proof, public_signals

    def prove_gnark(
        self,
        witness_mont: np.ndarray,
        az_mont: Array,
        bz_mont: Array,
        *,
        no_zk: bool = False,
        deterministic: bool = False,
    ) -> tuple[Groth16Proof, list[str]]:
        """Generate a Groth16 proof from gnark solved witness + pre-computed Az/Bz.

        Args:
            witness_mont: Full solved witness as bn254_sf_mont numpy array
                (raw Montgomery form from Go exporter), shape (num_wires,).
            az_mont: Pre-computed A*z in Montgomery form (bn254_sf_mont).
            bz_mont: Pre-computed B*z in Montgomery form (bn254_sf_mont).
            no_zk: If True, use r=s=0 (no ZK blinding, eliminates EC muls).
            deterministic: If True, use fixed non-zero r, s for reproducible
                proofs that still exercise full ZK blinding computation.

        Returns:
            Tuple of (proof, public_signals).
        """
        # Convert witness from Montgomery to standard form for MSM scalars.
        # The Go exporter writes raw Montgomery bytes; the caller reinterprets
        # them as bn254_sf_mont.  lax.convert_element_type performs Montgomery
        # reduction (multiply by R⁻¹ mod p).
        z_mont = jnp.array(witness_mont, dtype=bn254_sf_mont)
        z_std = lax.convert_element_type(z_mont, bn254_sf)

        if no_zk:
            r_int, s_int = 0, 0
        elif deterministic:
            r_int, s_int = _DETERMINISTIC_R, _DETERMINISTIC_S
        else:
            r_int = secrets.randbelow(BN254_FR_MODULUS)
            s_int = secrets.randbelow(BN254_FR_MODULUS)

        r_val = jnp.array(r_int, dtype=bn254_sf)
        s_val = jnp.array(s_int, dtype=bn254_sf)
        rs = bn254_sf(r_int) * bn254_sf(s_int)
        rs_val = jnp.array(rs, dtype=bn254_sf)

        pi_a, pi_b2, pi_c = self._run_prove(
            z_std,
            az_mont,
            bz_mont,
            r_val,
            s_val,
            rs_val,
        )

        proof = Groth16Proof(pi_a=pi_a, pi_b=pi_b2, pi_c=pi_c)
        # gnark: public inputs at wire indices 0..num_public-1.
        # witness_mont contains Montgomery values; int() on bn254_sf_mont
        # returns the standard-form value (Montgomery reduction applied).
        public_signals = [
            str(int(witness_mont[i])) for i in range(self.config.num_public)
        ]

        return proof, public_signals


def compile_circom(zkey: ZKeyV1) -> CompiledProver:
    """Compile a proving key into a reusable prover.

    Pre-computes all circuit-constant data (NTT twiddles, point arrays,
    VK/delta points).  The returned ``CompiledProver`` can generate
    multiple proofs via ``prove(wtns, az_mont, bz_mont)``.

    Args:
        zkey: Parsed proving key (ZKeyV1).

    Returns:
        Compiled prover ready for proof generation.
    """
    num_public = zkey.header_groth.num_public_inputs
    domain_size = zkey.domain_size
    log_n = int(math.log2(domain_size))
    vk = zkey.verifying_key

    # NTT per-stage twiddle arrays
    ntt = NTT(bn254_sf_mont, BN254_FR_ROOT_OF_UNITY)
    fwd_stage_twiddles, inv_stage_twiddles, inv_n = ntt.get_stage_twiddles(log_n)

    # Coset shift powers: [1, g, g², ..., g^(n-1)] where g = omega_{2n}
    coset_shift = jnp.array(
        bn254_sf_mont(BN254_FR_ROOT_OF_UNITY)
        ** (1 << (BN254_TWO_ADIC_BITS - log_n - 1)),
        dtype=bn254_sf_mont,
    )
    shift_powers = _build_shift_powers(coset_shift, log_n)

    # Point arrays (affine — lax.msm takes affine directly)
    pa1 = _g1_points_to_array(zkey.points_a1)
    pb1 = _g1_points_to_array(zkey.points_b1)
    pb2 = _g2_points_to_array(zkey.points_b2)
    pc1 = _g1_points_to_array(zkey.points_c1)
    ph1 = _g1_points_to_array(zkey.points_h1)

    # VK points (affine scalars)
    alpha1 = _g1_to_affine(vk.alpha_g1)
    beta1 = _g1_to_affine(vk.beta_g1)
    beta2 = _g2_to_affine(vk.beta_g2)

    # Delta points (affine scalars)
    delta_g1 = _g1_to_affine(vk.delta_g1)
    delta_g2 = _g2_to_affine(vk.delta_g2)

    config = ProveConfig(
        log_n=log_n,
        num_public=num_public,
        is_circom=True,
    )

    return CompiledProver(
        config=config,
        fwd_stage_twiddles=fwd_stage_twiddles,
        inv_stage_twiddles=inv_stage_twiddles,
        inv_n=inv_n,
        shift_powers=shift_powers,
        pa1=pa1,
        pb1=pb1,
        pb2=pb2,
        pc1=pc1,
        ph1=ph1,
        alpha1=alpha1,
        beta1=beta1,
        beta2=beta2,
        delta_g1=delta_g1,
        delta_g2=delta_g2,
        # den / inv_shift_powers not needed for circom
    )


def compile_gnark(data: GnarkProvingData) -> CompiledProver:
    """Compile gnark exported proving data into a reusable prover.

    Converts gnark's tuple-based point arrays into JAX affine arrays
    suitable for ``lax.msm``.  No XYZZ conversion or window_bits needed.

    Args:
        data: Loaded gnark export data.

    Returns:
        Compiled prover ready for proof generation via ``prove_gnark()``.
    """
    num_public = data.num_public
    domain_size = data.domain_size
    log_n = int(math.log2(domain_size))

    # NTT per-stage twiddle arrays.
    # Must use gnark's root of unity (generator 5) to match the CRS evaluation
    # domain — using a different primitive root gives wrong h-polynomial.
    ntt = NTT(bn254_sf_mont, GNARK_FR_ROOT_OF_UNITY)
    fwd_stage_twiddles, inv_stage_twiddles, inv_n = ntt.get_stage_twiddles(log_n)

    # Coset shift powers: [1, g, g², ..., g^(n-1)]
    # gnark uses Fr multiplicative generator (= 5) as coset shift,
    # NOT omega_{2n} which circom/snarkjs uses.
    coset_shift = jnp.array(
        bn254_sf_mont(GNARK_COSET_GEN),
        dtype=bn254_sf_mont,
    )
    shift_powers = _build_shift_powers(coset_shift, log_n)

    # Inverse coset shift powers: [1, g⁻¹, g⁻², ..., g⁻⁽ⁿ⁻¹⁾]
    coset_gen_inv = pow(GNARK_COSET_GEN, BN254_FR_MODULUS - 2, BN254_FR_MODULUS)
    coset_shift_inv = jnp.array(
        bn254_sf_mont(coset_gen_inv),
        dtype=bn254_sf_mont,
    )
    inv_shift_powers = _build_shift_powers(coset_shift_inv, log_n)

    # Vanishing polynomial denominator: den = 1 / (g^n - 1)
    g_pow_n = pow(GNARK_COSET_GEN, domain_size, BN254_FR_MODULUS)
    den_int = pow(g_pow_n - 1, BN254_FR_MODULUS - 2, BN254_FR_MODULUS)
    den = jnp.array(bn254_sf_mont(den_int), dtype=bn254_sf_mont)

    # Point arrays (affine — lax.msm takes affine directly)
    pa1 = _g1_tuples_to_array(data.pk_a_g1)
    pb1 = _g1_tuples_to_array(data.pk_b_g1)
    pb2 = _g2_tuples_to_array(data.pk_b_g2)
    pc1 = _g1_tuples_to_array(data.pk_k_g1)
    ph1 = _g1_tuples_to_array(data.pk_z_g1)  # n-1 points (gnark)

    # VK points (affine scalars)
    alpha1 = jnp.array(
        bn254_g1_affine(data.vk_alpha_g1),
        dtype=bn254_g1_affine,
    )
    beta1 = jnp.array(
        bn254_g1_affine(data.vk_beta_g1),
        dtype=bn254_g1_affine,
    )
    beta2 = jnp.array(
        bn254_g2_affine(data.vk_beta_g2),
        dtype=bn254_g2_affine,
    )

    # Delta points (affine scalars)
    delta_g1 = jnp.array(
        bn254_g1_affine(data.pk_delta_g1),
        dtype=bn254_g1_affine,
    )
    delta_g2 = jnp.array(
        bn254_g2_affine(data.pk_delta_g2),
        dtype=bn254_g2_affine,
    )

    config = ProveConfig(
        log_n=log_n,
        num_public=num_public,
        is_circom=False,
    )

    return CompiledProver(
        config=config,
        fwd_stage_twiddles=fwd_stage_twiddles,
        inv_stage_twiddles=inv_stage_twiddles,
        inv_n=inv_n,
        shift_powers=shift_powers,
        pa1=pa1,
        pb1=pb1,
        pb2=pb2,
        pc1=pc1,
        ph1=ph1,
        alpha1=alpha1,
        beta1=beta1,
        beta2=beta2,
        delta_g1=delta_g1,
        delta_g2=delta_g2,
        den=den,
        inv_shift_powers=inv_shift_powers,
    )


# ---------------------------------------------------------------------------
# Phase 1+2 JIT: NTT + field arithmetic + MSMs (GPU-safe)
# ---------------------------------------------------------------------------


@partial(jax.jit, static_argnums=(0,))
def _prove_ntt(
    config: ProveConfig,
    az_mont: Array,
    bz_mont: Array,
    # NTT arrays (per-stage twiddles)
    fwd_stage_tw: tuple[Array, ...],
    inv_stage_tw: tuple[Array, ...],
    inv_n: Array,
    shift_powers: Array,
    # Gnark-specific (unused dummy for circom — never traced)
    den: Array,
    inv_shift_powers: Array,
) -> Array:
    """Phase 1: NTT + field arithmetic → h-polynomial scalars.

    GPU-safe JIT function.  Returns h_scalars (bn254_sf) for the MSM phase.
    """
    log_n = config.log_n
    n = 1 << log_n

    # Hadamard: Cz = Az ⊙ Bz
    cz = az_mont * bz_mont

    # IFFT x 3
    a_poly = NTT.inverse_ntt(az_mont, inv_n, log_n, *inv_stage_tw)
    b_poly = NTT.inverse_ntt(bz_mont, inv_n, log_n, *inv_stage_tw)
    c_poly = NTT.inverse_ntt(cz, inv_n, log_n, *inv_stage_tw)

    # Coset NTT x 3
    a_coset = NTT.forward_ntt(a_poly * shift_powers, log_n, *fwd_stage_tw)
    b_coset = NTT.forward_ntt(b_poly * shift_powers, log_n, *fwd_stage_tw)
    c_coset = NTT.forward_ntt(c_poly * shift_powers, log_n, *fwd_stage_tw)

    # Quotient: h = a * b - c on coset
    h_evals_mont = a_coset * b_coset - c_coset

    if config.is_circom:
        return lax.convert_element_type(h_evals_mont, bn254_sf)
    else:
        h_evals_mont = h_evals_mont * den
        h_poly = NTT.inverse_ntt(h_evals_mont, inv_n, log_n, *inv_stage_tw)
        h_coeffs = h_poly * inv_shift_powers
        h_coeffs = lax.bit_reverse(h_coeffs, dimensions=[0])
        return lax.convert_element_type(h_coeffs[: n - 1], bn254_sf)


# ---------------------------------------------------------------------------
# Phase 3 JIT: EC assembly + ZK blinding (CPU-only)
# ---------------------------------------------------------------------------


@partial(jax.jit, static_argnums=())
def _prove_phase3(
    msm_1: Array,
    msm_2: Array,
    msm_3: Array,
    msm_4: Array,
    msm_5: Array,
    r_val: Array,
    s_val: Array,
    rs_val: Array,
    # VK points (affine scalars)
    alpha1: Array,
    beta1: Array,
    beta2: Array,
    # Delta points (affine scalars)
    delta_g1: Array,
    delta_g2: Array,
) -> tuple[Array, Array, Array]:
    """Phase 3: EC assembly + ZK blinding + affine conversion.

    CPU-only — EC scalar multiply (r * delta, s * pi_a, etc.) triggers
    horizontal fusion that generates `.b256` ptxas errors on GPU.

    Accepts MSM results from _prove_phase12.
    ZK blinding is always computed.  When r=s=0 the blinding EC muls
    produce identity so the result is equivalent to a non-ZK proof with
    no compile-time branching.
    """
    # EC assembly: convert all points to jacobian for addition, then back
    # to affine at the end. StableHLO only defines EC add for jacobian.
    _j1 = lambda p: lax.convert_element_type(p, bn254_g1_jacobian)
    _j2 = lambda p: lax.convert_element_type(p, bn254_g2_jacobian)

    pi_a = _j1(alpha1) + _j1(msm_1)
    pi_b1 = _j1(beta1) + _j1(msm_2)
    pi_b2 = _j2(beta2) + _j2(msm_3)
    pi_c = _j1(msm_4) + _j1(msm_5)

    # ZK blinding via EC scalar multiply (returns jacobian already)
    r_delta1 = r_val * delta_g1
    s_delta2 = s_val * delta_g2
    s_pi_a = s_val * pi_a
    r_pi_b1 = r_val * pi_b1
    rs_delta1 = rs_val * delta_g1

    pi_a = pi_a + r_delta1
    pi_b2 = pi_b2 + s_delta2
    pi_c = pi_c + s_pi_a + r_pi_b1 + rs_delta1

    # Convert to affine for output
    pi_a = lax.convert_element_type(pi_a, bn254_g1_affine)
    pi_b2 = lax.convert_element_type(pi_b2, bn254_g2_affine)
    pi_c = lax.convert_element_type(pi_c, bn254_g1_affine)

    return pi_a, pi_b2, pi_c


# ---------------------------------------------------------------------------
# JIT-internal helpers (called during trace of _prove_ntt / _prove_phase12)
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


def _g1_to_affine(point: G1Point) -> Array:
    """Convert a single G1Point to a JAX affine scalar."""
    return jnp.array(bn254_g1_affine((point.x, point.y)), dtype=bn254_g1_affine)


def _g2_to_affine(point: G2Point) -> Array:
    """Convert a single G2Point to a JAX affine scalar."""
    return jnp.array(bn254_g2_affine((point.x, point.y)), dtype=bn254_g2_affine)


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


def _g1_tuples_to_array(points: list[tuple[int, int]]) -> Array:
    """Convert a list of (x, y) tuples to a JAX G1 affine array."""
    return jnp.array(
        [bn254_g1_affine(p) for p in points],
        dtype=bn254_g1_affine,
    )


def _g2_tuples_to_array(
    points: list[tuple[tuple[int, int], tuple[int, int]]],
) -> Array:
    """Convert a list of ((x0,x1),(y0,y1)) tuples to a JAX G2 affine array."""
    return jnp.array(
        [bn254_g2_affine(p) for p in points],
        dtype=bn254_g2_affine,
    )
