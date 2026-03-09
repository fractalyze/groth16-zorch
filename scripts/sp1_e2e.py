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

"""SP1 Groth16 end-to-end: load gnark export → compile → prove → verify.

Usage:
    cd jax && python ../rabbitsnark-py/scripts/sp1_e2e.py \
        --export_dir=../sp1-groth16-bench/testdata/export/

The script uses pre-computed solution vectors (Az, Bz) from the Go exporter,
so no R1CS solver is needed.  GPU acceleration via lax.msm when available.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from zk_dtypes import bn254_sf_mont

from rabbitsnark.circom.zkey.verifying_key import G1Point, G2Point
from rabbitsnark.gnark.loader import FIELD_ELEM_SIZE, load_gnark_export
from rabbitsnark.groth16.prover import compile_gnark
from rabbitsnark.groth16.verifier import VerificationKey, verify


def _load_solution_mont(path: Path, domain_size: int) -> jax.Array:
    """Load pre-computed solution vector as padded bn254_sf_mont JAX array.

    The Go exporter writes raw Montgomery form bytes (gnark fr.Element is
    [4]uint64 Montgomery).  We reinterpret the bytes directly as bn254_sf_mont
    to avoid the double-conversion that would occur if we parsed to Python ints
    and then passed them through the bn254_sf_mont constructor (which auto-
    converts standard → Montgomery).

    Args:
        path: Binary file of 32-byte LE Montgomery field elements.
        domain_size: NTT domain size (power of 2, >= num_constraints).

    Returns:
        (domain_size,) JAX array in Montgomery form.
    """
    raw = np.fromfile(str(path), dtype=np.uint8)
    num_constraints = raw.size // FIELD_ELEM_SIZE
    # Pad raw bytes to domain_size with zero bytes (Montgomery(0) = 0).
    if num_constraints < domain_size:
        padding = np.zeros(
            (domain_size - num_constraints) * FIELD_ELEM_SIZE, dtype=np.uint8
        )
        raw = np.concatenate([raw, padding])
    mont_np = raw.view(np.dtype(bn254_sf_mont))
    return jnp.array(mont_np.tolist(), dtype=bn254_sf_mont)


def _build_vk(data) -> VerificationKey:
    """Build VerificationKey from gnark export data."""
    return VerificationKey(
        alpha_g1=G1Point.from_ints(*data.vk_alpha_g1),
        beta_g2=G2Point.from_ints(*data.vk_beta_g2),
        gamma_g2=G2Point.from_ints(*data.vk_gamma_g2),
        delta_g2=G2Point.from_ints(*data.pk_delta_g2),
        ic=[G1Point.from_ints(*p) for p in data.vk_ic],
    )


def main():
    parser = argparse.ArgumentParser(description="SP1 Groth16 E2E prove + verify")
    parser.add_argument(
        "--export_dir",
        type=str,
        default="../sp1-groth16-bench/testdata/export/",
        help="Path to gnark export directory",
    )
    parser.add_argument(
        "--split",
        action="store_true",
        help="Split phase 1+2 (GPU) / phase 3 (CPU)",
    )
    parser.add_argument(
        "--no_zk",
        action="store_true",
        help="Use r=s=0 for deterministic (non-ZK) proof",
    )
    parser.add_argument(
        "--skip_verify",
        action="store_true",
        help="Skip verification step",
    )
    args = parser.parse_args()

    export_dir = Path(args.export_dir)
    print(f"Loading gnark export from {export_dir}")

    # --- Load ---
    t0 = time.perf_counter()
    data = load_gnark_export(export_dir)
    t_load = time.perf_counter() - t0
    print(
        f"Load: {t_load:.1f}s  "
        f"(wires={data.num_wires:,}, constraints={data.num_constraints:,}, "
        f"domain={data.domain_size:,})"
    )

    # --- Compile ---
    print("\nCompiling proving key...")
    t0 = time.perf_counter()
    compiled = compile_gnark(data)
    t_compile = time.perf_counter() - t0
    print(f"Compile: {t_compile:.1f}s")

    # --- Prepare Az/Bz ---
    # Load directly from binary as bn254_sf_mont (raw Montgomery bytes).
    # Using _load_solution_mont instead of the loader's Python-int arrays
    # avoids the double Montgomery conversion bug.
    print("\nPreparing solution vectors (Az, Bz)...")
    t0 = time.perf_counter()
    az_mont = _load_solution_mont(export_dir / "solution_a.bin", data.domain_size)
    bz_mont = _load_solution_mont(export_dir / "solution_b.bin", data.domain_size)
    t_prep = time.perf_counter() - t0
    print(f"Solution prep: {t_prep:.1f}s")

    # --- Prove ---
    witness_mont = data.witness_full  # already bn254_sf_mont from loader
    print(f"\nProving (split={args.split}, no_zk={args.no_zk})...")
    t0 = time.perf_counter()
    proof, public_signals = compiled.prove_gnark(
        witness_mont,
        az_mont,
        bz_mont,
        no_zk=args.no_zk,
        split=args.split,
    )
    t_prove = time.perf_counter() - t0
    print(f"Prove: {t_prove:.1f}s")
    print(f"Public signals ({len(public_signals)}): {public_signals}")

    # --- Verify ---
    if not args.skip_verify:
        print("\nVerifying proof...")
        vk = _build_vk(data)
        t0 = time.perf_counter()
        valid = verify(vk, proof, public_signals)
        t_verify = time.perf_counter() - t0
        print(f"Verify: {t_verify:.1f}s — {'VALID' if valid else 'INVALID'}")
    else:
        print("\nSkipping verification.")

    # --- Summary ---
    print(f"\n{'='*50}")
    print(f"Load:     {t_load:.1f}s")
    print(f"Compile:  {t_compile:.1f}s")
    print(f"Sol prep: {t_prep:.1f}s")
    print(f"Prove:    {t_prove:.1f}s")
    if not args.skip_verify:
        print(f"Verify:   {t_verify:.1f}s")
    print(f"{'='*50}")


if __name__ == "__main__":
    main()
