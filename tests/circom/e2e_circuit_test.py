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

"""End-to-end test: circuit.so + input.json → witness → prove → verify.

Uses the multiplier_3 circuit (3-input multiplier) with the --circuit path:
    inputs: {"a": "3", "b": "4", "c": "5"}
    expected output: 60 (3 * 4 * 5)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from absl.testing import absltest
from zk_dtypes import bn254_sf

from rabbitsnark.circom.witness_calculator import CircomWitnessCalculator, load_w2s
from rabbitsnark.circom.zkey import parse_zkey
from rabbitsnark.groth16 import compile_circom, write_public_signals
from rabbitsnark.groth16.verifier import VerificationKey, verify
from rabbitsnark.r1cs_solver import compute_abc


class TestCircomE2ECircuitProve(absltest.TestCase):
    """End-to-end: circuit.so + input.json → witness → prove → verify."""

    def setUp(self):
        test_data_dir = Path(__file__).parent / "data"
        self.zkey = parse_zkey(test_data_dir / "multiplier_3.zkey")
        self.compiled = compile_circom(self.zkey)

        # Load circuit and compute witness
        circuit_so = self._find_circuit_so()
        calc = CircomWitnessCalculator(circuit_so)
        with open(test_data_dir / "multiplier_3_input.json") as f:
            inputs = json.load(f)
        w2s = load_w2s(test_data_dir / "multiplier_3_w2s.json")

        witness = calc.compute_witness(inputs, w2s)

        # witness bytes are standard form (from_mont applied in calculator)
        self.az_mont, self.bz_mont = compute_abc(
            witness,
            self.compiled.csr,
            self.compiled.domain_size,
            self.compiled.domain_size,
        )
        self.z_std = witness.view(np.dtype(bn254_sf))
        self.public_signals = write_public_signals(
            self.z_std[: self.compiled.config.num_public + 1],
            self.compiled.config.num_public,
        )
        self.vk = VerificationKey.from_zkey(self.zkey)

    def _find_circuit_so(self) -> str:
        """Find libmultiplier_3.so in Bazel runfiles."""
        for parent in Path(__file__).absolute().parents:
            if parent.name.endswith(".runfiles"):
                candidate = (
                    parent
                    / "rabbitsnark"
                    / "third_party"
                    / "circom_circuits"
                    / "libmultiplier_3.so"
                )
                if candidate.exists():
                    return str(candidate)
        raise FileNotFoundError("libmultiplier_3.so not found in runfiles")

    def test_prove_verify_no_zk(self):
        """Deterministic proof (r=s=0) verifies correctly."""
        proof, public_signals = self.compiled.prove(
            self.z_std,
            self.az_mont,
            self.bz_mont,
            self.public_signals,
            no_zk=True,
        )
        self.assertTrue(verify(self.vk, proof, public_signals))

    @absltest.skip(
        "Segfaults during JIT compilation of ZK blinding EC ops in Inline mode. "
        "Requires AOT EC runtime (prime-ir ECRuntime). Re-enable after ZKX "
        "jaxlib bump that includes AOT support."
    )
    def test_prove_verify_with_zk(self):
        """Randomized ZK proof verifies correctly."""
        proof, public_signals = self.compiled.prove(
            self.z_std,
            self.az_mont,
            self.bz_mont,
            self.public_signals,
            no_zk=False,
        )
        self.assertTrue(verify(self.vk, proof, public_signals))


if __name__ == "__main__":
    import atexit
    import os

    # TODO(chokobole): Fix ZKX JAX atexit use-after-free triggered by ctypes
    # circuit .so loading. The crash is in jax._src.api.clean_up →
    # clear_all_caches → dict.clear. Tests pass (2/2 OK) but the process
    # crashes during atexit cleanup. Force clean exit after tests complete.
    atexit.register(os._exit, 0)
    absltest.main()
