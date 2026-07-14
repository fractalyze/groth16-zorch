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

# Expected output hashes per (op, degree). Inputs are deterministic (fixed
# seed), so outputs must be identical across runs. If a compiler or runtime
# change alters these, it indicates a correctness regression.
_EXPECTED_HASHES: dict[tuple[str, int], str] = {
    # degree 4 (smoke test)
    ("fft", 4): "e67daa1b3a1ff76c475914f059314edecfc15dd4df5391c2844df3f967c735b8",
    ("ifft", 4): "69d28d58b50137796491499fa884ea1f8b35c11adb2dbef0c2ac9e63a5a7d1a1",
    ("msm_g1", 4): "25fc6d3944e8f815c5b2b7c41deedd04ac113b645228b34d892220a2fd5cbd70",
    # degree 16
    ("fft", 16): "88f3691fddb59bf515970ad73ba91f4953d1d92d05e7673b51ffd0cc173fa0d3",
    ("ifft", 16): "076a19ed238b6f9c14c9481781eb1a0617139f817c57ff1c328cf2ee3a7031c4",
    ("msm_g1", 16): "6938ab273f49cb82f29998ba2456b723b87585962ededdc8a6941855328e8af3",
    # degree 18
    ("fft", 18): "34519dc28351ba5b5e7317e9257c5f8932b5bdb8e6ef3a32557d820ae718a48e",
    ("ifft", 18): "f5b250ad02708816df34554ba69a2830c491a279e8e2582e04137ea9ca134a0f",
    ("msm_g1", 18): "01b9bf7257b9a897554018d856241d310ba9737007a2665ad6849617e4a439ce",
    # degree 20
    ("fft", 20): "b688a07e9347bdf4607dd6fe157f221c002430a63dc4b3d36ed50ca42f26b568",
    ("ifft", 20): "ad80eb4a2d70f1941c4365f8e23109346b6658a8fa5e9b100128bd25d29c4872",
    ("msm_g1", 20): "968ce09676a34b4783f9ff84b4d9041e1aa5db10e7bbd06bd8a87412d0eca06b",
    # degree 22
    ("fft", 22): "50167109d9443f63464c2ed63e650786a3813d3e94ad8762500f108b68dab723",
    ("ifft", 22): "c73328a6bca72f650635446cfc86fdb0178a8d29c970b57af189a1091dcdd709",
    ("msm_g1", 22): "ce3289dd5429b30bc9b72d34fa6a5aabdcf8f141691b5743477f0dedeaed1b5b",
    # degree 24
    ("fft", 24): "5c41c9122b6873a1a9173310bb7d1a65012c8e9c89b9f6c587bb5615e7cd4f46",
    ("ifft", 24): "c79e15b2cb4560df4753b4a84404cdc628f0e5e8a437cbddbebb2ba57d61ba2a",
    ("msm_g1", 24): "0d292d795f431adf9692b8471adc4418a34784969a7b22906d2731dda11d5c42",
}


def _hash_array(arr: jnp.ndarray) -> str:
    """Compute SHA-256 hash of a JAX array's raw bytes."""
    arr_np = np.array(arr)
    if arr_np.ndim == 0:
        arr_np = arr_np.reshape(1)
    raw = arr_np.view(np.uint8).tobytes()
    return hashlib.sha256(raw).hexdigest()


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
            default="16,18,20,22,24",
            help="Comma-separated log2 sizes (default: 16,18,20,22,24)",
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
            scalars_hash = _hash_array(scalars)

            # Mutable containers capture fn() results for hash verification
            # against hardcoded _EXPECTED_HASHES constants.
            fft_last: dict = {}
            ifft_last: dict = {}
            msm_last: dict = {}

            def _fft_fn(s=scalars, sz=n, last=fft_last):
                r = lax.ntt(
                    s,
                    ntt_type=lax.NttType.NTT,
                    ntt_length=sz,
                    generator=GNARK_GENERATOR,
                )
                last["result"] = r
                return r

            def _ifft_fn(s=scalars, sz=n, last=ifft_last):
                r = lax.ntt(
                    s,
                    ntt_type=lax.NttType.INTT,
                    ntt_length=sz,
                    generator=GNARK_GENERATOR,
                )
                last["result"] = r
                return r

            yield BenchmarkOp(
                name="fft",
                fn=_fft_fn,
                metadata={**meta},
                input_hash=scalars_hash,
                output_hash_fn=lambda last=fft_last: _hash_array(last["result"]),
                verify_fn=lambda last=fft_last, ls=log_size: (
                    _hash_array(last["result"]) == _EXPECTED_HASHES[("fft", ls)]
                ),
            )

            yield BenchmarkOp(
                name="ifft",
                fn=_ifft_fn,
                metadata={**meta},
                input_hash=scalars_hash,
                output_hash_fn=lambda last=ifft_last: _hash_array(last["result"]),
                verify_fn=lambda last=ifft_last, ls=log_size: (
                    _hash_array(last["result"]) == _EXPECTED_HASHES[("ifft", ls)]
                ),
            )

            msm_s = msm_scalars_full[:n]
            msm_b = bases_full[:n]
            msm_hash = _hash_array(msm_s)

            def _msm_fn(sc=msm_s, ba=msm_b, last=msm_last):
                r = lax.msm(sc, ba)
                last["result"] = r
                return r

            yield BenchmarkOp(
                name="msm_g1",
                fn=_msm_fn,
                metadata={**meta},
                input_hash=msm_hash,
                output_hash_fn=lambda last=msm_last: _hash_array(last["result"]),
                verify_fn=lambda last=msm_last, ls=log_size: (
                    _hash_array(last["result"]) == _EXPECTED_HASHES[("msm_g1", ls)]
                ),
            )


def main() -> int:
    return PrimitivesBenchmark().run()


if __name__ == "__main__":
    sys.exit(main())
