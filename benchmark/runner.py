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

"""Reusable Gnark Groth16 benchmark runner."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import jax

from rabbitsnark.gnark.loader import load_gnark_export
from rabbitsnark.gnark.solutions import load_solutions_mont
from rabbitsnark.groth16.prover import compile_gnark
from rabbitsnark.groth16.verifier import VerificationKey, verify

if TYPE_CHECKING:
    from rabbitsnark.gnark.types import GnarkProvingData
    from rabbitsnark.groth16.prover import CompiledProver


@dataclass
class BenchmarkConfig:
    """Configuration for Gnark Groth16 benchmark runs."""

    iterations: int = 1
    warmup: int = 0
    no_zk: bool = False
    deterministic: bool = False
    skip_verify: bool = False


class GnarkBenchmarkRunner:
    """Reusable runner for Gnark Groth16 prove + verify benchmarks."""

    def __init__(self, export_dir: Path, config: BenchmarkConfig):
        self.export_dir = export_dir
        self.config = config

    def load(self) -> GnarkProvingData:
        """Load gnark export data from the export directory."""
        print(f"Loading gnark export from {self.export_dir}")
        t0 = time.perf_counter()
        data = load_gnark_export(self.export_dir)
        self.t_load = time.perf_counter() - t0
        print(
            f"Load: {self.t_load:.1f}s  "
            f"(wires={data.num_wires:,}, constraints={data.num_constraints:,}, "
            f"domain={data.domain_size:,})"
        )
        return data

    def compile(self, data: GnarkProvingData) -> CompiledProver:
        """Compile the proving key (one-time)."""
        print("\nCompiling proving key...")
        t0 = time.perf_counter()
        compiled = compile_gnark(data)
        self.t_compile = time.perf_counter() - t0
        print(f"Compile: {self.t_compile:.1f}s")
        return compiled

    def prepare_solutions(self, data: GnarkProvingData) -> tuple[jax.Array, jax.Array]:
        """Load pre-computed solution vectors as padded Montgomery-form arrays.

        Uses raw binary files from the export directory to avoid double
        Montgomery conversion.
        """
        print("\nPreparing solution vectors (Az, Bz)...")
        t0 = time.perf_counter()
        az_mont, bz_mont = load_solutions_mont(self.export_dir, data.domain_size)
        self.t_prep = time.perf_counter() - t0
        print(f"Solution prep: {self.t_prep:.1f}s")
        return az_mont, bz_mont

    def run_prove_iterations(
        self,
        compiled: CompiledProver,
        witness_mont: jax.Array,
        az_mont: jax.Array,
        bz_mont: jax.Array,
    ) -> list[float]:
        """Run warmup + measured prove iterations.

        Returns:
            List of measured prove times (excluding warmup).
        """
        total = self.config.warmup + self.config.iterations
        prove_times_all: list[float] = []
        self._last_proof = None
        self._last_public_signals = None

        for i in range(total):
            label = "warmup" if i < self.config.warmup else "measured"
            print(
                f"\nProve iteration {i} "
                f"({label}, "
                f"no_zk={self.config.no_zk}, "
                f"deterministic={self.config.deterministic})..."
            )
            t0 = time.perf_counter()
            proof, public_signals = compiled.prove_gnark(
                witness_mont,
                az_mont,
                bz_mont,
                no_zk=self.config.no_zk,
                deterministic=self.config.deterministic,
            )
            t_prove = time.perf_counter() - t0
            prove_times_all.append(t_prove)
            print(f"Prove: {t_prove:.1f}s")

            self._last_proof = proof
            self._last_public_signals = public_signals

        return prove_times_all[self.config.warmup :]

    def build_vk(self, data: GnarkProvingData) -> VerificationKey:
        """Build VerificationKey from gnark export data."""
        return VerificationKey.from_gnark(data)

    def verify_proof(self, vk: VerificationKey, proof, public_signals) -> bool:
        """Verify a Groth16 proof."""
        print("\nVerifying proof...")
        t0 = time.perf_counter()
        valid = verify(vk, proof, public_signals)
        t_verify = time.perf_counter() - t0
        print(f"Verify: {t_verify:.1f}s — {'VALID' if valid else 'INVALID'}")
        return valid
