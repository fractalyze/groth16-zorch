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
"""FFT, IFFT, and MSM G1 benchmarks for BN254 field.

Measures the performance of core cryptographic primitives on BN254:
  - Forward NTT (lax.ntt with gnark generator = 5)
  - Inverse NTT (lax.ntt INTT)
  - Multi-Scalar Multiplication G1 (lax.msm)

Allocates arrays at the maximum requested size once, then slices down
for smaller degrees to avoid repeated data generation overhead.

Usage:
    bazel run //benchmark:primitives_benchmark -- --output=results.json
    bazel run //benchmark:primitives_benchmark -- --sizes=16,18,20 --iterations=10
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from typing import Iterable

import jax.numpy as jnp
import numpy as np
from jax import lax
from zk_dtypes import bn254_g1_affine_mont, bn254_sf, bn254_sf_mont  # noqa: F401
from zkbench import BenchmarkConfig, BenchmarkOp, JaxBenchmark

# gnark uses generator 5 for NTT evaluation domain.
GNARK_GENERATOR = 5

# Hashes are computed in gnark-crypto canonical form so they match SP1
# ref's sp1-groth16-bench/cmd/primitives output byte-for-byte. See
# benchmark/CONTRACT.md ("Canonical output_hash rules") for the spec.
BN254_FR_MODULUS = (
    21888242871839275222246405745257275088548364400416034343698204186575808495617
)
BN254_FP_MODULUS = (
    21888242871839275222246405745257275088696311157297823662689037894645226208583
)
_MONT_R_INV_FP = pow(1 << 256, -1, BN254_FP_MODULUS)

# MSM values match `fractalyze/sp1@ref:sp1-groth16-bench/bin/primitives
# --sizes=… --iterations=1 --warmup=0 --json` on the same (seed=42)
# MT19937 stream — cross-impl identity is asserted by run_local.sh.
# FFT/IFFT values are rabbit-side only: zkx's lax.fft and gnark-crypto's
# fft.Domain.FFT agree on result[0]=sum(input) but diverge on result[i>0]
# (convention difference under investigation; see benchmark/CONTRACT.md).
_EXPECTED_HASHES: dict[tuple[str, int], str] = {
    # degree 4 (smoke test)
    ("fft", 4): "c390d98af36e6df3f0d8f8596f7157f7de7708e89f2f2607480628732fc5f18c",
    ("ifft", 4): "fde8b21addeb7144e7c5df34259f0d7bf41f00418c68ac5b94e708ba286b9cda",
    ("msm_g1", 4): "cd31b79a267287eaf1c0262abb133c5a93532f529173d32c4f1f9449efc57f95",
    # degree 16
    ("fft", 16): "c63b84b3199027b0391f35567400924669d5199736fc4067d336d830922e7ec2",
    ("ifft", 16): "92557e5fe1b5c5c5ac737d54537ed021d1b618b34d8c7bb5a786586c2ef257ca",
    ("msm_g1", 16): "8f8be98cfba31bd2d38589f4a4bcee0f5ba5a1387ae4da7859b3014810d71a24",
    # degree 18
    ("fft", 18): "39047b0ff1aaa1d6064e900a422fb035804b12c25783f6d8057a0b55eaa579b7",
    ("ifft", 18): "d1b4402334099e05783a1fb6434d1c13368573bfdf52bd63fe338afed4f84a38",
    ("msm_g1", 18): "ccba575dd045c22dfcc1a4202c844b48d795896d54a3cfbae864928f3b9b5e8f",
    # degree 20
    ("fft", 20): "d236623b3dfd0a2b1693c036eaac72e04b443bc2966c962550b5b6c62800cd4a",
    ("ifft", 20): "82d21fd0d5603b80e3a77539cf80c34f2eb726e08a7b242782c0ce8df9713351",
    ("msm_g1", 20): "b08e8b8286d767a7f784dcdc12895dca8161778f80d61bd5e8f030f5787e9273",
    # degree 22
    ("fft", 22): "13d38d7606295876aed0201215319146ac78fbca8d370beeb161416b0f9ed349",
    ("ifft", 22): "826db606c7987472977722b85114f37913407e1c8ec6665cdb0505e48abb4ec7",
    ("msm_g1", 22): "38f140969de7ccd3c79a8f3b8f63451dd071b931ad055927257f173e8de63240",
    # degree 24 not pre-populated — regenerate on first --sizes=24 run.
}


def _hash_scalars(arr) -> str:
    """SHA-256 of an fr.Element array, gnark canonical form: standard-form
    big-endian 32 bytes per element, concatenated in array order."""
    arr_np = np.asarray(arr)
    if arr_np.ndim == 0:
        arr_np = arr_np.reshape(1)
    h = hashlib.sha256()
    for i in range(arr_np.shape[0]):
        h.update((int(arr_np[i]) % BN254_FR_MODULUS).to_bytes(32, "big"))
    return h.hexdigest()


def _hash_g1_affine(point) -> str:
    """SHA-256 of a G1Affine point in gnark Marshal() canonical form
    (BN254 uses 64-byte uncompressed: x big-endian || y big-endian, both
    in standard form)."""
    point_np = np.asarray(point)
    item = point_np.item() if point_np.ndim == 0 else point_np[0]
    x_mont, y_mont = item.raw
    x = (int(x_mont) * _MONT_R_INV_FP) % BN254_FP_MODULUS
    y = (int(y_mont) * _MONT_R_INV_FP) % BN254_FP_MODULUS
    return hashlib.sha256(x.to_bytes(32, "big") + y.to_bytes(32, "big")).hexdigest()


def _generate_scalars(n: int, seed: int = 42) -> jnp.ndarray:
    """Generate deterministic BN254 scalar field elements."""
    rng = np.random.RandomState(seed)
    values = [bn254_sf_mont(int(rng.randint(1, 2**32))) for _ in range(n)]
    return jnp.array(values, dtype=bn254_sf_mont)


def _generate_bases(n: int) -> jnp.ndarray:
    """Generate n copies of the BN254 G1 generator point."""
    gen = bn254_g1_affine_mont((1, 2))
    return jnp.array(np.array([gen] * n, dtype=np.dtype(bn254_g1_affine_mont)))


class PrimitivesBenchmark(JaxBenchmark):

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            implementation="rabbitsnark",
            version="0.1.0",
            default_iterations=3,
            default_warmup=1,
        )

    def add_custom_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--sizes",
            type=str,
            default="16,18,20,22",
            help="Comma-separated log2 sizes (default: 16,18,20,22)",
        )

    def get_ops(self, args: argparse.Namespace) -> Iterable[BenchmarkOp]:
        sizes = sorted(int(s) for s in args.sizes.split(","))
        max_n = 1 << sizes[-1]

        # Allocate at maximum size once, slice down for smaller degrees.
        print(f"Generating scalars (2^{sizes[-1]})...")
        scalars_full = _generate_scalars(max_n)
        # MSM needs standard form; convert from Montgomery once.
        msm_scalars_full = lax.convert_element_type(scalars_full, bn254_sf)
        print(f"Generating MSM bases (2^{sizes[-1]})...")
        bases_full = _generate_bases(max_n)

        for log_size in sizes:
            n = 1 << log_size
            meta = {"field": "bn254", "degree": str(log_size)}

            scalars = scalars_full[:n]
            scalars_hash = _hash_scalars(scalars)

            # Mutable containers capture fn() results for hash verification
            # against hardcoded _EXPECTED_HASHES constants.
            fft_last: dict = {}
            ifft_last: dict = {}
            msm_last: dict = {}

            def _fft_fn(s=scalars, sz=n, last=fft_last):
                r = lax.ntt(s, ntt_type="NTT", ntt_length=sz, generator=GNARK_GENERATOR)
                last["result"] = r
                return r

            def _ifft_fn(s=scalars, sz=n, last=ifft_last):
                r = lax.ntt(
                    s, ntt_type="INTT", ntt_length=sz, generator=GNARK_GENERATOR
                )
                last["result"] = r
                return r

            yield BenchmarkOp(
                name="fft",
                fn=_fft_fn,
                metadata={**meta},
                input_hash=scalars_hash,
                output_hash_fn=lambda last=fft_last: _hash_scalars(last["result"]),
                verify_fn=lambda last=fft_last, ls=log_size: (
                    _hash_scalars(last["result"]) == _EXPECTED_HASHES[("fft", ls)]
                ),
            )

            yield BenchmarkOp(
                name="ifft",
                fn=_ifft_fn,
                metadata={**meta},
                input_hash=scalars_hash,
                output_hash_fn=lambda last=ifft_last: _hash_scalars(last["result"]),
                verify_fn=lambda last=ifft_last, ls=log_size: (
                    _hash_scalars(last["result"]) == _EXPECTED_HASHES[("ifft", ls)]
                ),
            )

            msm_s = msm_scalars_full[:n]
            msm_b = bases_full[:n]
            msm_hash = _hash_scalars(msm_s)

            def _msm_fn(sc=msm_s, ba=msm_b, last=msm_last):
                r = lax.msm(sc, ba)
                last["result"] = r
                return r

            yield BenchmarkOp(
                name="msm_g1",
                fn=_msm_fn,
                metadata={**meta},
                input_hash=msm_hash,
                output_hash_fn=lambda last=msm_last: _hash_g1_affine(last["result"]),
                verify_fn=lambda last=msm_last, ls=log_size: (
                    _hash_g1_affine(last["result"]) == _EXPECTED_HASHES[("msm_g1", ls)]
                ),
            )


def main() -> int:
    return PrimitivesBenchmark().run()


if __name__ == "__main__":
    sys.exit(main())
