#!/usr/bin/env python3
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

"""Export NTT forward/inverse as StableHLO MLIR for stablehlo_runner benchmarking.

Usage:
    cd /path/to/jax && python /path/to/export_ntt_stablehlo.py [--log_n=20]

Outputs:
    ntt_forward.stablehlo.mlir  - Forward NTT StableHLO module
    ntt_inverse.stablehlo.mlir  - Inverse NTT StableHLO module
"""

import argparse
import time

import jax.numpy as jnp
from zk_dtypes import bn254_sf_mont

from rabbitsnark.ntt import BN254_FR_ROOT_OF_UNITY, NTT


def export_ntt(log_n: int, output_dir: str = ".") -> None:
    """Export NTT forward and inverse as StableHLO MLIR files."""
    n = 1 << log_n
    print(f"Exporting NTT for n = 2^{log_n} = {n}")

    # Create NTT instance and pre-compute twiddles
    print("Creating NTT instance and computing twiddles...")
    t0 = time.perf_counter()
    ntt = NTT(bn254_sf_mont, BN254_FR_ROOT_OF_UNITY)
    fwd_tw, inv_tw, inv_n = ntt.get_stage_twiddles(log_n)
    t_init = time.perf_counter() - t0
    print(f"  NTT init: {t_init:.3f}s")

    # Create dummy input for tracing
    dummy = jnp.zeros(n, dtype=bn254_sf_mont)

    # --- Forward NTT ---
    print("\nExporting forward NTT...")
    t0 = time.perf_counter()
    lowered = NTT.forward_ntt.lower(dummy, log_n, *fwd_tw)
    t_lower = time.perf_counter() - t0
    print(f"  Trace + lower: {t_lower:.3f}s")

    t0 = time.perf_counter()
    stablehlo_text = lowered.as_text()
    t_text = time.perf_counter() - t0
    print(f"  as_text(): {t_text:.3f}s")
    print(f"  StableHLO size: {len(stablehlo_text)} chars")

    forward_path = f"{output_dir}/ntt_forward_{log_n}.stablehlo.mlir"
    with open(forward_path, "w") as f:
        f.write(stablehlo_text)
    print(f"  Written to: {forward_path}")

    # --- Inverse NTT ---
    print("\nExporting inverse NTT...")
    t0 = time.perf_counter()
    lowered_inv = NTT.inverse_ntt.lower(dummy, inv_n, log_n, *inv_tw)
    t_lower_inv = time.perf_counter() - t0
    print(f"  Trace + lower: {t_lower_inv:.3f}s")

    t0 = time.perf_counter()
    stablehlo_text_inv = lowered_inv.as_text()
    t_text_inv = time.perf_counter() - t0
    print(f"  as_text(): {t_text_inv:.3f}s")
    print(f"  StableHLO size: {len(stablehlo_text_inv)} chars")

    inverse_path = f"{output_dir}/ntt_inverse_{log_n}.stablehlo.mlir"
    with open(inverse_path, "w") as f:
        f.write(stablehlo_text_inv)
    print(f"  Written to: {inverse_path}")

    # --- Also time actual execution for comparison ---
    print("\nTiming JAX execution (for comparison with stablehlo_runner)...")

    # Warmup
    result = NTT.forward_ntt(dummy, log_n, *fwd_tw)
    result.block_until_ready()

    # Timed run
    iters = 10
    t0 = time.perf_counter()
    for _ in range(iters):
        result = NTT.forward_ntt(dummy, log_n, *fwd_tw)
        result.block_until_ready()
    t_exec = time.perf_counter() - t0
    print(
        f"  Forward NTT: {iters} iterations in {t_exec:.3f}s "
        f"({t_exec / iters * 1000:.1f}ms/iter)"
    )

    # Warmup inverse
    result = NTT.inverse_ntt(dummy, inv_n, log_n, *inv_tw)
    result.block_until_ready()

    t0 = time.perf_counter()
    for _ in range(iters):
        result = NTT.inverse_ntt(dummy, inv_n, log_n, *inv_tw)
        result.block_until_ready()
    t_exec_inv = time.perf_counter() - t0
    print(
        f"  Inverse NTT: {iters} iterations in {t_exec_inv:.3f}s "
        f"({t_exec_inv / iters * 1000:.1f}ms/iter)"
    )

    print("\nDone!")


def main():
    parser = argparse.ArgumentParser(description="Export NTT as StableHLO MLIR")
    parser.add_argument(
        "--log_n", type=int, default=20, help="Log2 of NTT size (default: 20)"
    )
    parser.add_argument(
        "--output_dir", type=str, default=".", help="Output directory for MLIR files"
    )
    args = parser.parse_args()
    export_ntt(args.log_n, args.output_dir)


if __name__ == "__main__":
    main()
