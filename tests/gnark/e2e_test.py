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

"""End-to-end test: gnark load -> compile -> prove -> verify.

Uses the tiny_multiply fixture (x × x == y, x=3, y=9):
    4 wires, 2 constraints, domain_size=2
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from absl.testing import absltest
from jax import lax
from zk_dtypes import bn254_sf, bn254_sf_mont

from groth16_zorch.gnark import load_gnark_export
from groth16_zorch.groth16 import compile_gnark
from groth16_zorch.groth16.verifier import VerificationKey, verify


class TestGnarkE2EProveVerify(absltest.TestCase):
    """End-to-end: load gnark export -> prove -> verify.

    Az/Bz come straight from the export (solution_a/b, computed by gnark's
    Go solver); the witness is converted Montgomery -> standard form here.
    """

    def setUp(self):
        self.data_dir = Path(__file__).parent / "data" / "tiny_multiply"
        self.data = load_gnark_export(self.data_dir)
        self.compiled = compile_gnark(self.data)
        self.az_mont = self.data.az_mont
        self.bz_mont = self.data.bz_mont
        self.z_std = np.asarray(
            lax.convert_element_type(
                np.asarray(self.data.witness_full).view(bn254_sf_mont), bn254_sf
            )
        )
        self.public_signals = [
            str(int(self.z_std[i])) for i in range(self.compiled.config.num_public)
        ]
        self.vk = VerificationKey.from_gnark(self.data)

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

    def test_prove_verify_deterministic(self):
        """Deterministic proof (fixed non-zero r, s) verifies correctly."""
        proof, public_signals = self.compiled.prove(
            self.z_std,
            self.az_mont,
            self.bz_mont,
            self.public_signals,
            deterministic=True,
        )
        self.assertTrue(verify(self.vk, proof, public_signals))

    def test_prove_verify_with_zk(self):
        """Randomized ZK proof verifies correctly."""
        proof, public_signals = self.compiled.prove(
            self.z_std,
            self.az_mont,
            self.bz_mont,
            self.public_signals,
        )
        self.assertTrue(verify(self.vk, proof, public_signals))

    def test_invalid_signal_rejects(self):
        """Public signal modification causes verification failure."""
        proof, public_signals = self.compiled.prove(
            self.z_std,
            self.az_mont,
            self.bz_mont,
            self.public_signals,
            no_zk=True,
        )
        # Tamper: change first public signal
        tampered = ["99"] + list(public_signals[1:])
        self.assertFalse(verify(self.vk, proof, tampered))


if __name__ == "__main__":
    absltest.main()
