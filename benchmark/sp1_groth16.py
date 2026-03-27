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
"""SP1 Groth16 benchmark using JaxBenchmark.

Loads a gnark export, compiles the proving key, solves the witness, then
benchmarks prove iterations.  Verification runs after timing via verify_fn.

Usage:
    bazel run //benchmark:sp1_groth16 -- \
        --export_dir=/data/testdata/sp1-groth16/ \
        --deterministic --iterations=3 --warmup=1 \
        --output=benchmark_results.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from pathlib import Path
from typing import Iterable

import jax.numpy as jnp
from jax import lax
from zk_dtypes import bn254_sf, bn254_sf_mont
from zkbench import BenchmarkConfig, BenchmarkOp, JaxBenchmark

from rabbitsnark.gnark.compute_abc import load_solver_data, solve_and_compute
from rabbitsnark.gnark.loader import load_gnark_export
from rabbitsnark.groth16.prover import compile_gnark
from rabbitsnark.groth16.verifier import VerificationKey, verify


def _hash_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _proof_hash(proof) -> str:
    proof_json = json.dumps(proof.to_json(), sort_keys=True)
    return hashlib.sha256(proof_json.encode()).hexdigest()


class Groth16Benchmark(JaxBenchmark):

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            implementation="rabbitsnark",
            version="0.1.0",
            default_iterations=3,
            default_warmup=1,
        )

    def add_custom_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--export_dir",
            type=str,
            required=True,
            help="Path to gnark binary export directory",
        )
        parser.add_argument(
            "--deterministic",
            action="store_true",
            help="Fixed non-zero r, s for reproducible proofs",
        )
        parser.add_argument(
            "--circuit",
            type=str,
            default="sp1",
            help="Circuit name for metadata (default: sp1)",
        )

    def get_ops(self, args: argparse.Namespace) -> Iterable[BenchmarkOp]:
        export_dir = Path(args.export_dir)

        # --- One-time setup ---
        print(f"Loading gnark export from {export_dir}")
        t0 = time.perf_counter()
        data = load_gnark_export(export_dir)
        solver = load_solver_data(export_dir)
        t_load = time.perf_counter() - t0
        print(
            f"Load: {t_load:.1f}s  "
            f"(wires={data.num_wires:,}, constraints={data.num_constraints:,}, "
            f"domain={data.domain_size:,})"
        )

        print("\nCompiling proving key...")
        t0 = time.perf_counter()
        compiled = compile_gnark(data)
        t_compile = time.perf_counter() - t0
        print(f"Compile: {t_compile:.1f}s")

        print("\nSolving witness + computing Az/Bz...")
        t0 = time.perf_counter()
        witness_full, az_mont, bz_mont = solve_and_compute(data.witness_full, solver)
        t_prep = time.perf_counter() - t0
        print(f"Solution prep: {t_prep:.1f}s")

        z_mont = jnp.array(witness_full, dtype=bn254_sf_mont)
        z_std = lax.convert_element_type(z_mont, bn254_sf)
        public_signals = [
            str(int(witness_full[i])) for i in range(compiled.config.num_public)
        ]
        vk = VerificationKey.from_gnark(data)

        # Mutable container to capture the last proof for verify_fn.
        last = {}

        def prove_fn():
            proof, pub = compiled.prove(
                z_std,
                az_mont,
                bz_mont,
                public_signals,
                deterministic=args.deterministic,
            )
            last["proof"] = proof
            last["pub"] = pub
            return proof

        def verify_fn():
            return verify(vk, last["proof"], last["pub"])

        log_n = int(math.log2(data.domain_size))
        witness_path = export_dir / "witness_full.bin"
        input_hash = _hash_bytes(
            witness_path.read_bytes() if witness_path.exists() else b""
        )

        yield BenchmarkOp(
            name="groth16",
            fn=prove_fn,
            metadata={
                "field": "bn254",
                "degree": str(log_n),
                "circuit": args.circuit,
                "constraints": str(data.num_constraints),
            },
            input_hash=input_hash,
            output_hash_fn=lambda: _proof_hash(last["proof"]),
            verify_fn=verify_fn,
        )


def main() -> int:
    return Groth16Benchmark().run()


if __name__ == "__main__":
    sys.exit(main())
