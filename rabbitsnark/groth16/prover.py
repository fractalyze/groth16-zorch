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

    compiled = compile_circom(zkey)                    # parse zkey (one-time)
    proof, signals = compiled.prove_circom(wtns, az, bz)  # per-witness

Architecture:
    compile_circom(zkey) -> CompiledProver
    |-- NTT twiddle arrays
    |-- Coset shift powers
    |-- Point arrays (affine)
    +-- VK / Delta points (affine scalars)

    CompiledProver.prove_circom(wtns, az_mont, bz_mont)
    |-- Input preparation: z_std, r, s, neg_rs  (per-proof)
    +-- Phase 1+2: _prove_phase12(config, ...)  <- JIT (GPU-safe)
    |  |-- Cz = Az ⊙ Bz (Hadamard)
    |  |-- IFFT x 3 (stage twiddles via static strided slicing)
    |  |-- Coset NTT x 3
    |  |-- Quotient: h_evals_mont = a_coset * b_coset - c_coset
    |  |-- h-polynomial (static branch on config.is_circom):
    |  |     circom: convert mont→std, MSM with n eval points
    |  |     gnark:  * den → IFFT → * inv_shift_powers, MSM with n-1 coefficients
    |  +-- MSMs 1-5 via lax.msm
    +-- Phase 3: _prove_phase3(config, ...)  <- JIT (CPU-only)
       |-- EC assembly + ZK blinding via scalar *
       +-- Convert to affine

    Coset generator:
      circom: omega_{2n} (primitive 2n-th root of unity)
      gnark:  Fr multiplicative generator = 5
"""

from __future__ import annotations

import json
import math
import secrets
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import zk_dtypes
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
    """Static configuration for _prove_core (compile-time constants)."""

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
        neg_rs_val: Array,
        *,
        split: bool = False,
    ) -> tuple[Array, Array, Array]:
        """Run the proving computation (combined or split phase).

        Args:
            split: If True, run phase 1+2 and phase 3 as separate JIT
                dispatches. This allows phase 1+2 (NTT + MSMs) to run on
                GPU while phase 3 (EC assembly) runs on CPU.
        """
        # Gnark-specific arrays: None for circom → dummy scalar placeholder
        # (never traced thanks to config.is_circom static branching)
        den = self.den if self.den is not None else jnp.array(0, dtype=bn254_sf_mont)
        inv_sp = (
            self.inv_shift_powers
            if self.inv_shift_powers is not None
            else jnp.array(0, dtype=bn254_sf_mont)
        )

        if split:
            # Phase 1: NTT + h-polynomial on GPU (JIT).
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
            # Phase 2: MSMs on GPU (chunked to work around ICICLE bug
            # where arrays > ~11M elements produce incorrect results).
            l = self.config.num_public  # noqa: E741
            private_start = l + 1 if self.config.is_circom else l
            msm_1 = _chunked_msm_g1(z_std, self.pa1)
            msm_2 = _chunked_msm_g1(z_std, self.pb1)
            msm_3 = _chunked_msm_g2(z_std, self.pb2)
            msm_4 = _chunked_msm_g1(z_std[private_start:], self.pc1)
            msm_5 = _chunked_msm_g1(h_scalars, self.ph1)
            # Phase 3 must run on CPU — EC scalar multiply generates .b256
            # PTX instructions that ptxas cannot compile on GPU.
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
                    _to_cpu(neg_rs_val),
                    _to_cpu(self.alpha1),
                    _to_cpu(self.beta1),
                    _to_cpu(self.beta2),
                    _to_cpu(self.delta_g1),
                    _to_cpu(self.delta_g2),
                )
        return _prove_core(
            self.config,
            z_std,
            r_val,
            s_val,
            neg_rs_val,
            az_mont,
            bz_mont,
            self.fwd_stage_twiddles,
            self.inv_stage_twiddles,
            self.inv_n,
            self.shift_powers,
            self.pa1,
            self.pb1,
            self.pb2,
            self.pc1,
            self.ph1,
            self.alpha1,
            self.beta1,
            self.beta2,
            self.delta_g1,
            self.delta_g2,
            den,
            inv_sp,
        )

    def prove_circom(
        self,
        wtns: WtnsV2,
        az_mont: Array,
        bz_mont: Array,
        *,
        no_zk: bool = False,
        deterministic: bool = False,
        split: bool = False,
    ) -> tuple[Groth16Proof, list[str]]:
        """Generate a Groth16 proof from a witness.

        Args:
            wtns: Parsed witness (WtnsV2).
            az_mont: Pre-computed A*z in Montgomery form (bn254_sf_mont).
            bz_mont: Pre-computed B*z in Montgomery form (bn254_sf_mont).
            no_zk: If True, use r=s=0 (no ZK blinding, eliminates EC muls).
            deterministic: If True, use fixed non-zero r, s for reproducible
                proofs that still exercise full ZK blinding computation.
            split: If True, use separate JIT for phase 1+2 (GPU) and
                phase 3 (CPU).

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
        neg_rs = -(bn254_sf(r_int) * bn254_sf(s_int))
        neg_rs_val = jnp.array(neg_rs, dtype=bn254_sf)

        pi_a, pi_b2, pi_c = self._run_prove(
            z_std,
            az_mont,
            bz_mont,
            r_val,
            s_val,
            neg_rs_val,
            split=split,
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
        split: bool = False,
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
            split: If True, use separate JIT for phase 1+2 (GPU) and
                phase 3 (CPU).

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
        neg_rs = -(bn254_sf(r_int) * bn254_sf(s_int))
        neg_rs_val = jnp.array(neg_rs, dtype=bn254_sf)

        pi_a, pi_b2, pi_c = self._run_prove(
            z_std,
            az_mont,
            bz_mont,
            r_val,
            s_val,
            neg_rs_val,
            split=split,
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


@partial(jax.jit, static_argnums=(0,))
def _prove_phase12(
    config: ProveConfig,
    z_std: Array,
    az_mont: Array,
    bz_mont: Array,
    # NTT arrays (per-stage twiddles)
    fwd_stage_tw: tuple[Array, ...],
    inv_stage_tw: tuple[Array, ...],
    inv_n: Array,
    shift_powers: Array,
    # Point arrays (affine)
    pa1: Array,
    pb1: Array,
    pb2: Array,
    pc1: Array,
    ph1: Array,
    # Gnark-specific (unused dummy for circom — never traced)
    den: Array,
    inv_shift_powers: Array,
) -> tuple[Array, Array, Array, Array, Array]:
    """Phase 1+2: NTT + field arithmetic + MSMs.

    GPU-safe — contains only field ops and MSMs, no EC scalar multiply
    which triggers horizontal fusion `.b256` ptxas errors on GPU.

    Circom vs gnark h-polynomial handling (static branch on config.is_circom):
      circom: h_evals (evaluation form, mont→std) → MSM with n points
      gnark:  h_evals * den → IFFT → * inv_shift_powers (coefficient form)
              → MSM with n-1 points

    Returns (msm_1, msm_2, msm_3, msm_4, msm_5) as affine points.
    """
    log_n = config.log_n
    n = 1 << log_n
    l = config.num_public  # noqa: E741

    # ---------------------------------------------------------------
    # Phase 1: Arithmetic (Hadamard + NTT + quotient) in Montgomery form
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
    # Phase 2: h-polynomial processing + MSMs via lax.msm
    # ---------------------------------------------------------------

    if config.is_circom:
        # Circom/snarkjs: h in evaluation form, convert mont→std for MSM.
        # pk.h_g1 has n points matching n evaluation values.
        h_scalars = lax.convert_element_type(h_evals_mont, bn254_sf)
    else:
        # Gnark (standard Groth16): recover h in coefficient form.
        #   1) Divide by vanishing polynomial: h_evals *= den
        #   2) IFFT → coefficient domain
        #   3) Undo coset shift: h_poly *= inv_shift_powers
        #   4) Bit-reverse to match gnark's pk.G1.Z order
        # gnark's setup.go bit-reverses Z bases (line 247) so that
        # Z[i] = [τ^(bit_reverse(i)) / δ]₁. computeH returns h in
        # DIF bit-reversed order. Our IFFT returns natural order,
        # so we must bit-reverse before the MSM to match Z's ordering.
        h_evals_mont = h_evals_mont * den
        h_poly = NTT.inverse_ntt(h_evals_mont, inv_n, log_n, *inv_stage_tw)
        h_coeffs = h_poly * inv_shift_powers
        h_coeffs = lax.bit_reverse(h_coeffs, dimensions=[0])
        h_scalars = lax.convert_element_type(h_coeffs[: n - 1], bn254_sf)

    msm_1 = lax.msm(z_std, pa1)
    msm_2 = lax.msm(z_std, pb1)
    msm_3 = lax.msm(z_std, pb2)
    # circom: private wires start at l+1 (skip ONE wire + l public inputs)
    # gnark:  private wires start at l (no ONE wire, l public inputs)
    private_start = l + 1 if config.is_circom else l
    msm_4 = lax.msm(z_std[private_start:], pc1)
    msm_5 = lax.msm(h_scalars, ph1)

    return msm_1, msm_2, msm_3, msm_4, msm_5


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
    neg_rs_val: Array,
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

    # ZK blinding (Groth16 §3.2): blind A and B₁ first, then compute C
    # with blinded values.
    #   A' = A + r*δ₁
    #   B₁' = B₁ + s*δ₁
    #   B₂' = B₂ + s*δ₂
    #   C' = C + s*A' + r*B₁' - rs*δ₁
    r_delta1 = r_val * delta_g1
    s_delta1 = s_val * delta_g1
    s_delta2 = s_val * delta_g2

    pi_a = pi_a + r_delta1          # A' = A + r*δ₁
    pi_b1 = pi_b1 + s_delta1        # B₁' = B₁ + s*δ₁
    pi_b2 = pi_b2 + s_delta2        # B₂' = B₂ + s*δ₂

    s_pi_a = s_val * pi_a           # s * A' (blinded)
    r_pi_b1 = r_val * pi_b1         # r * B₁' (blinded)
    neg_rs_delta1 = neg_rs_val * delta_g1  # -rs * δ₁

    pi_c = pi_c + s_pi_a + r_pi_b1 + neg_rs_delta1

    # Convert to affine for output
    pi_a = lax.convert_element_type(pi_a, bn254_g1_affine)
    pi_b2 = lax.convert_element_type(pi_b2, bn254_g2_affine)
    pi_c = lax.convert_element_type(pi_c, bn254_g1_affine)

    return pi_a, pi_b2, pi_c


# ---------------------------------------------------------------------------
# Combined _prove_core: Phase 1+2 → Phase 3 (for single-dispatch use)
# ---------------------------------------------------------------------------


@partial(jax.jit, static_argnums=(0,))
def _prove_core(
    config: ProveConfig,
    z_std: Array,
    r_val: Array,
    s_val: Array,
    neg_rs_val: Array,
    az_mont: Array,
    bz_mont: Array,
    # NTT arrays (per-stage twiddles)
    fwd_stage_tw: tuple[Array, ...],
    inv_stage_tw: tuple[Array, ...],
    inv_n: Array,
    shift_powers: Array,
    # Point arrays (affine)
    pa1: Array,
    pb1: Array,
    pb2: Array,
    pc1: Array,
    ph1: Array,
    # VK points (affine scalars)
    alpha1: Array,
    beta1: Array,
    beta2: Array,
    # Delta points (affine scalars)
    delta_g1: Array,
    delta_g2: Array,
    # Gnark-specific (unused dummy for circom)
    den: Array,
    inv_shift_powers: Array,
) -> tuple[Array, Array, Array]:
    """Combined kernel: Phase 1+2 (NTT + MSMs) → Phase 3 (EC assembly).

    Single JIT dispatch for CPU-only execution. For GPU/CPU split, use
    _prove_phase12 and _prove_phase3 separately.
    """
    msm_1, msm_2, msm_3, msm_4, msm_5 = _prove_phase12(
        config,
        z_std,
        az_mont,
        bz_mont,
        fwd_stage_tw,
        inv_stage_tw,
        inv_n,
        shift_powers,
        pa1,
        pb1,
        pb2,
        pc1,
        ph1,
        den,
        inv_shift_powers,
    )
    return _prove_phase3(
        msm_1,
        msm_2,
        msm_3,
        msm_4,
        msm_5,
        r_val,
        s_val,
        neg_rs_val,
        alpha1,
        beta1,
        beta2,
        delta_g1,
        delta_g2,
    )


# ---------------------------------------------------------------------------
# JIT-internal helpers (called during trace of _prove_core)
# ---------------------------------------------------------------------------

# ICICLE MSM produces incorrect results when GPU memory is insufficient
# for the optimal bucket window.  The threshold depends on total GPU memory
# pressure (input arrays + ICICLE scratch space), so multiple sequential
# MSMs fail at lower per-call sizes.  2M chunks are safe on RTX 5090.
_MSM_CHUNK = 2_000_000


def _chunked_msm_g1(scalars: Array, points: Array) -> Array:
    """Multi-scalar multiplication on G1 with chunking for large arrays.

    Splits into chunks of _MSM_CHUNK to work around ICICLE GPU memory leak
    across sequential MSM calls within the same process.  Each chunk runs as
    a separate HLO execution so ICICLE scratch space is freed between calls.
    """
    n = scalars.shape[0]
    if n <= _MSM_CHUNK:
        return lax.msm(scalars, points)
    chunks = []
    for offset in range(0, n, _MSM_CHUNK):
        end = min(offset + _MSM_CHUNK, n)
        chunks.append(lax.msm(scalars[offset:end], points[offset:end]))
    cpu = jax.devices("cpu")[0]
    _to_cpu = lambda x: jnp.array(np.array(x), dtype=x.dtype)
    with jax.default_device(cpu):
        acc = lax.convert_element_type(_to_cpu(chunks[0]), bn254_g1_jacobian)
        for chunk in chunks[1:]:
            acc = acc + lax.convert_element_type(_to_cpu(chunk), bn254_g1_jacobian)
        return lax.convert_element_type(acc, bn254_g1_affine)


def _chunked_msm_g2(scalars: Array, points: Array) -> Array:
    """Multi-scalar multiplication on G2 with chunking for large arrays."""
    n = scalars.shape[0]
    if n <= _MSM_CHUNK:
        return lax.msm(scalars, points)
    chunks = []
    for offset in range(0, n, _MSM_CHUNK):
        end = min(offset + _MSM_CHUNK, n)
        chunks.append(lax.msm(scalars[offset:end], points[offset:end]))
    cpu = jax.devices("cpu")[0]
    _to_cpu = lambda x: jnp.array(np.array(x), dtype=x.dtype)
    with jax.default_device(cpu):
        acc = lax.convert_element_type(_to_cpu(chunks[0]), bn254_g2_jacobian)
        for chunk in chunks[1:]:
            acc = acc + lax.convert_element_type(_to_cpu(chunk), bn254_g2_jacobian)
        return lax.convert_element_type(acc, bn254_g2_affine)


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


def compile_gnark_native(export_dir: str | Path) -> CompiledProver:
    """Compile gnark export directly from binary files (zero-copy).

    Bypasses Python int intermediaries by loading PK point arrays as
    native zk_dtypes numpy arrays and converting to JAX arrays directly.

    ~10x faster than ``load_gnark_export`` + ``compile_gnark`` for PK points
    (mmap + view vs Python int parsing + list comprehension).

    Args:
        export_dir: Path to the gnark export directory.

    Returns:
        Compiled prover ready for proof generation via ``prove_gnark()``.
    """
    from rabbitsnark.gnark.loader import (
        _read_g1_points_native,
        _read_g2_points_native,
    )

    d = Path(export_dir)
    t_start = time.perf_counter()

    # Metadata
    with open(d / "metadata.json") as f:
        meta = json.load(f)
    num_public = meta["num_public"]
    domain_size = meta["domain_size"]
    log_n = int(math.log2(domain_size))

    # NTT twiddle arrays
    t = time.perf_counter()
    ntt = NTT(bn254_sf_mont, GNARK_FR_ROOT_OF_UNITY)
    fwd_stage_twiddles, inv_stage_twiddles, inv_n = ntt.get_stage_twiddles(log_n)
    print(f"  twiddles: {time.perf_counter() - t:.1f}s")

    # Coset shift powers
    t = time.perf_counter()
    coset_shift = jnp.array(bn254_sf_mont(GNARK_COSET_GEN), dtype=bn254_sf_mont)
    shift_powers = _build_shift_powers(coset_shift, log_n)
    coset_gen_inv = pow(GNARK_COSET_GEN, BN254_FR_MODULUS - 2, BN254_FR_MODULUS)
    coset_shift_inv = jnp.array(bn254_sf_mont(coset_gen_inv), dtype=bn254_sf_mont)
    inv_shift_powers = _build_shift_powers(coset_shift_inv, log_n)
    g_pow_n = pow(GNARK_COSET_GEN, domain_size, BN254_FR_MODULUS)
    den_int = pow(g_pow_n - 1, BN254_FR_MODULUS - 2, BN254_FR_MODULUS)
    den = jnp.array(bn254_sf_mont(den_int), dtype=bn254_sf_mont)
    print(f"  shift/den: {time.perf_counter() - t:.1f}s")

    # PK point arrays — zero-copy native loading
    t = time.perf_counter()
    pa1_np = _read_g1_points_native(d / "pk_a_g1.bin")
    pb1_np = _read_g1_points_native(d / "pk_b_g1.bin")
    pb2_np = _read_g2_points_native(d / "pk_b_g2.bin")
    pc1_np = _read_g1_points_native(d / "pk_k_g1.bin")
    ph1_np = _read_g1_points_native(d / "pk_z_g1.bin")
    t_read = time.perf_counter() - t
    print(f"  PK read (native): {t_read:.1f}s")

    # numpy → JAX array conversion (direct — no tolist needed)
    t = time.perf_counter()
    pa1 = jnp.array(pa1_np, dtype=bn254_g1_affine)
    pb1 = jnp.array(pb1_np, dtype=bn254_g1_affine)
    pb2 = jnp.array(pb2_np, dtype=bn254_g2_affine)
    pc1 = jnp.array(pc1_np, dtype=bn254_g1_affine)
    ph1 = jnp.array(ph1_np, dtype=bn254_g1_affine)
    t_jax = time.perf_counter() - t
    print(f"  PK to JAX: {t_jax:.1f}s")

    # VK / Delta points (small — Python int path is fine)
    vk_g1 = _read_g1_points_native(d / "vk_alpha_g1.bin")
    vk_beta_g1 = _read_g1_points_native(d / "vk_beta_g1.bin")
    vk_beta_g2 = _read_g2_points_native(d / "vk_beta_g2.bin")
    vk_gamma_g2 = _read_g2_points_native(d / "vk_gamma_g2.bin")
    delta_g1_np = _read_g1_points_native(d / "pk_delta_g1.bin")
    delta_g2_np = _read_g2_points_native(d / "pk_delta_g2.bin")

    alpha1 = jnp.array(vk_g1[0], dtype=bn254_g1_affine)
    beta1 = jnp.array(vk_beta_g1[0], dtype=bn254_g1_affine)
    beta2 = jnp.array(vk_beta_g2[0], dtype=bn254_g2_affine)
    delta_g1 = jnp.array(delta_g1_np[0], dtype=bn254_g1_affine)
    delta_g2 = jnp.array(delta_g2_np[0], dtype=bn254_g2_affine)

    print(f"  total compile: {time.perf_counter() - t_start:.1f}s")

    return CompiledProver(
        config=ProveConfig(log_n=log_n, num_public=num_public, is_circom=False),
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


_ZK_DTYPE_MAP: dict[str, type] = {}


def _resolve_zk_dtype(name: str):
    """Resolve a zk_dtypes type name to its dtype class."""
    if not _ZK_DTYPE_MAP:
        for attr in dir(zk_dtypes):
            obj = getattr(zk_dtypes, attr)
            if isinstance(obj, type):
                try:
                    _ZK_DTYPE_MAP[np.dtype(obj).name] = obj
                except TypeError:
                    pass
    return _ZK_DTYPE_MAP[name]


def _save_zk_array(path: Path, arr) -> None:
    """Save a JAX/numpy array with zk_dtypes as raw bytes + dtype name."""
    np_arr = np.array(arr)
    np_arr.tofile(str(path))
    (path.parent / (path.stem + ".dtype")).write_text(np_arr.dtype.name)


def _load_zk_array(path: Path):
    """Load a raw binary file as numpy array with zk_dtypes."""
    dtype_name = (path.parent / (path.stem + ".dtype")).read_text()
    dtype_cls = _resolve_zk_dtype(dtype_name)
    return np.fromfile(str(path), dtype=np.dtype(dtype_cls))


def save_compiled_prover(prover: CompiledProver, cache_dir: str | Path) -> None:
    """Save compiled prover arrays to disk for fast reloading.

    Writes all JAX arrays as raw binary files with dtype metadata.
    Subsequent loads via ``load_compiled_prover`` skip twiddle computation
    and point array conversion entirely.

    Args:
        prover: A compiled prover to save.
        cache_dir: Directory to write cache files into.
    """
    d = Path(cache_dir)
    d.mkdir(parents=True, exist_ok=True)

    # Config
    with open(d / "config.json", "w") as f:
        json.dump(
            {
                "log_n": prover.config.log_n,
                "num_public": prover.config.num_public,
                "is_circom": prover.config.is_circom,
            },
            f,
        )

    # Scalar / 1D arrays
    for name in (
        "inv_n",
        "shift_powers",
        "alpha1",
        "beta1",
        "beta2",
        "delta_g1",
        "delta_g2",
        "den",
        "inv_shift_powers",
    ):
        val = getattr(prover, name)
        if val is not None:
            _save_zk_array(d / f"{name}.bin", val)

    # Point arrays
    for name in ("pa1", "pb1", "pb2", "pc1", "ph1"):
        _save_zk_array(d / f"{name}.bin", getattr(prover, name))

    # Twiddle stage arrays
    for prefix, twiddles in [
        ("fwd", prover.fwd_stage_twiddles),
        ("inv", prover.inv_stage_twiddles),
    ]:
        for i, tw in enumerate(twiddles):
            _save_zk_array(d / f"{prefix}_tw_{i}.bin", tw)
        (d / f"{prefix}_tw_count.txt").write_text(str(len(twiddles)))


def load_compiled_prover(cache_dir: str | Path) -> CompiledProver:
    """Load a compiled prover from cache (fast — no twiddle computation).

    Args:
        cache_dir: Directory containing files from ``save_compiled_prover``.

    Returns:
        Compiled prover ready for proof generation.
    """
    d = Path(cache_dir)
    t_start = time.perf_counter()

    with open(d / "config.json") as f:
        cfg = json.load(f)
    config = ProveConfig(
        log_n=cfg["log_n"],
        num_public=cfg["num_public"],
        is_circom=cfg["is_circom"],
    )

    def _load(name):
        np_arr = _load_zk_array(d / f"{name}.bin")
        dtype_cls = _resolve_zk_dtype(np_arr.dtype.name)
        return jnp.array(np_arr, dtype=dtype_cls)

    # Twiddle stages
    def _load_stages(prefix):
        count = int((d / f"{prefix}_tw_count.txt").read_text().strip())
        return tuple(_load(f"{prefix}_tw_{i}") for i in range(count))

    t = time.perf_counter()
    fwd_stage_twiddles = _load_stages("fwd")
    inv_stage_twiddles = _load_stages("inv")
    print(f"  twiddles: {time.perf_counter() - t:.1f}s")

    t = time.perf_counter()
    inv_n = _load("inv_n")
    shift_powers = _load("shift_powers")
    den = _load("den")
    inv_shift_powers = _load("inv_shift_powers")
    print(f"  scalars: {time.perf_counter() - t:.1f}s")

    t = time.perf_counter()
    pa1 = _load("pa1")
    pb1 = _load("pb1")
    pb2 = _load("pb2")
    pc1 = _load("pc1")
    ph1 = _load("ph1")
    print(f"  PK points: {time.perf_counter() - t:.1f}s")

    alpha1 = _load("alpha1")
    beta1 = _load("beta1")
    beta2 = _load("beta2")
    delta_g1 = _load("delta_g1")
    delta_g2 = _load("delta_g2")

    print(f"  total load: {time.perf_counter() - t_start:.1f}s")

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


def aot_compile(prover: CompiledProver) -> None:
    """Trigger AOT compilation for _prove_ntt and _prove_phase3.

    Traces each JIT function with abstract shapes matching the prover's
    arrays, then compiles to machine code.  With ``jax_compilation_cache_dir``
    set, the compiled executables are persisted to disk so that subsequent
    ``prove_gnark()`` calls (even in a new process) skip LLVM compilation
    entirely.

    Call this once during the "compile" step::

        compiled = compile_gnark_native(export_dir)
        aot_compile(compiled)           # ← triggers LLVM compilation
        save_compiled_prover(compiled, cache_dir)

    Then in "prove"::

        compiled = load_compiled_prover(cache_dir)
        proof = compiled.prove_gnark(...)  # ← cache hit, no LLVM work
    """
    n = 1 << prover.config.log_n

    # --- AOT compile _prove_ntt (CPU, ~31s cold → cached) ---
    # Use the already-decorated @jax.jit function's .lower() to ensure
    # the cache key matches the actual prove call.
    t = time.perf_counter()
    ntt_lowered = _prove_ntt.lower(
        prover.config,
        jax.ShapeDtypeStruct((n,), bn254_sf_mont),
        jax.ShapeDtypeStruct((n,), bn254_sf_mont),
        prover.fwd_stage_twiddles,
        prover.inv_stage_twiddles,
        prover.inv_n,
        prover.shift_powers,
        prover.den,
        prover.inv_shift_powers,
    )
    ntt_lowered.compile()
    print(f"  AOT _prove_ntt: {time.perf_counter() - t:.1f}s")

    # --- AOT compile _prove_phase3 (CPU, ~5s cold → cached) ---
    t = time.perf_counter()
    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        g1_shape = jax.ShapeDtypeStruct((), bn254_g1_affine)
        g2_shape = jax.ShapeDtypeStruct((), bn254_g2_affine)
        sf_shape = jax.ShapeDtypeStruct((), bn254_sf)
        phase3_lowered = _prove_phase3.lower(
            g1_shape,
            g1_shape,
            g2_shape,
            g1_shape,
            g1_shape,
            sf_shape,
            sf_shape,
            sf_shape,
            prover.alpha1,
            prover.beta1,
            prover.beta2,
            prover.delta_g1,
            prover.delta_g2,
        )
        phase3_lowered.compile()
    print(f"  AOT _prove_phase3: {time.perf_counter() - t:.1f}s")


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
