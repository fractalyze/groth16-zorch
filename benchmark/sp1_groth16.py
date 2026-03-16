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

"""SP1 Groth16 end-to-end: load gnark export -> compile -> prove -> verify.

Usage:
    cd jax && python ../rabbitsnark-py/benchmark/sp1_groth16.py \
        --export_dir=../sp1-groth16-bench/testdata/export/

The script uses pre-computed solution vectors (Az, Bz) from the Go exporter,
so no R1CS solver is needed.  GPU acceleration via lax.msm when available.
"""

from __future__ import annotations

import argparse
import hashlib
import statistics
from pathlib import Path

from benchmark.runner import BenchmarkConfig, GnarkBenchmarkRunner


def _write_zkbench_report(
    prove_times: list[float],
    t_load: float,
    t_compile: float,
    t_prep: float,
    witness_bytes: bytes,
    proof_bytes: bytes,
    verified: bool,
    num_constraints: int,
    circuit: str = "sp1",
):
    """Write benchmark_results.json in zkbench schema."""
    from zkbench.schema import (
        BenchmarkReport,
        BenchmarkResult,
        Metadata,
        MetricValue,
        TestVectors,
    )
    from zkbench.statistics import (
        calculate_confidence_interval,
        calculate_statistics,
    )
    from zkbench.utils import compute_hash

    mean, stdev = calculate_statistics(prove_times)
    lower, upper = calculate_confidence_interval(mean, stdev)

    # Use median for the primary value.
    median = statistics.median(prove_times)

    # Convert s -> ms, round to 2 decimals.
    to_ms = lambda v: round(v * 1000, 2)

    input_hash = compute_hash(witness_bytes)
    output_hash = compute_hash(proof_bytes) if proof_bytes else ""

    load_ms = to_ms(t_load + t_compile + t_prep)
    prove_ms_lower = to_ms(lower)
    prove_ms_upper = to_ms(upper)

    metadata = Metadata.create("sp1-groth16-rabbitsnark", "0.1.0")
    report = BenchmarkReport(metadata=metadata)

    report.benchmarks[f"groth16_{circuit}_prove"] = BenchmarkResult(
        latency=MetricValue(
            value=to_ms(median),
            unit="ms",
            lower_value=prove_ms_lower,
            upper_value=prove_ms_upper,
        ),
        iterations=len(prove_times),
        test_vectors=TestVectors(
            input_hash=input_hash,
            output_hash=output_hash,
            verified=verified,
        ),
        metadata={
            "constraints": num_constraints,
            "mean_ms": to_ms(mean),
            "stdev_ms": to_ms(stdev),
        },
    )

    report.benchmarks[f"groth16_{circuit}_load"] = BenchmarkResult(
        latency=MetricValue(value=load_ms, unit="ms"),
        metadata={
            "load_s": round(t_load, 2),
            "compile_s": round(t_compile, 2),
            "sol_prep_s": round(t_prep, 2),
        },
    )

    report.benchmarks[f"groth16_{circuit}_e2e"] = BenchmarkResult(
        latency=MetricValue(value=round(load_ms + to_ms(median), 2), unit="ms"),
    )

    with open("benchmark_results.json", "w") as f:
        f.write(report.to_json())
    print("\nWrote benchmark_results.json")


def main():
    parser = argparse.ArgumentParser(description="SP1 Groth16 E2E prove + verify")
    parser.add_argument(
        "--export_dir",
        type=str,
        default="../sp1-groth16-bench/testdata/export/",
        help="Path to gnark export directory",
    )
    parser.add_argument(
        "--no_zk",
        action="store_true",
        help="Use r=s=0 (no ZK blinding, eliminates EC muls)",
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use fixed non-zero r, s for reproducible proofs with full blinding",
    )
    parser.add_argument(
        "--skip_verify",
        action="store_true",
        help="Skip verification step",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1,
        help="Number of prove iterations (default: 1)",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=0,
        help="Number of warmup iterations excluded from stats (default: 0)",
    )
    parser.add_argument(
        "--zkbench",
        action="store_true",
        help="Write zkbench JSON to benchmark_results.json",
    )
    parser.add_argument(
        "--circuit",
        type=str,
        default="sp1",
        help="Circuit name for benchmark naming (default: sp1)",
    )
    args = parser.parse_args()

    config = BenchmarkConfig(
        iterations=args.iterations,
        warmup=args.warmup,
        no_zk=args.no_zk,
        deterministic=args.deterministic,
        skip_verify=args.skip_verify,
    )
    runner = GnarkBenchmarkRunner(Path(args.export_dir), config)

    data = runner.load()
    compiled = runner.compile(data)
    az_mont, bz_mont = runner.prepare_solutions(data)

    witness_mont = data.witness_full
    measured_times = runner.run_prove_iterations(
        compiled, witness_mont, az_mont, bz_mont
    )
    print(
        f"\nPublic signals ({len(runner._last_public_signals)}): "
        f"{runner._last_public_signals}"
    )

    # --- Verify ---
    verified = False
    if not args.skip_verify:
        vk = runner.build_vk(data)
        verified = runner.verify_proof(
            vk, runner._last_proof, runner._last_public_signals
        )
    else:
        print("\nSkipping verification.")

    # --- Summary ---
    print(f"\n{'='*50}")
    print(f"Load:     {runner.t_load:.1f}s")
    print(f"Compile:  {runner.t_compile:.1f}s")
    print(f"Sol prep: {runner.t_prep:.1f}s")
    for i, t in enumerate(measured_times):
        print(f"Prove[{i}]: {t:.1f}s")
    if not args.skip_verify:
        print(f"Verify:   verified={verified}")
    print(f"{'='*50}")

    # --- zkbench output ---
    if args.zkbench:
        export_dir = Path(args.export_dir)
        witness_path = export_dir / "groth16_witness.json"
        witness_bytes = witness_path.read_bytes() if witness_path.exists() else b""

        proof_bytes = b""
        if runner._last_proof is not None:
            proof_json = json.dumps(runner._last_proof.to_json(), sort_keys=True)
            proof_bytes = hashlib.sha256(proof_json.encode()).digest()

        _write_zkbench_report(
            prove_times=measured_times,
            t_load=runner.t_load,
            t_compile=runner.t_compile,
            t_prep=runner.t_prep,
            witness_bytes=witness_bytes,
            proof_bytes=proof_bytes,
            verified=verified,
            num_constraints=data.num_constraints,
            circuit=args.circuit,
        )


if __name__ == "__main__":
    main()
