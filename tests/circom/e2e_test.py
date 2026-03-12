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

"""End-to-end test: compile -> prove -> verify round-trip (circom).

Uses the multiplier_3 circuit (3-input multiplier):
    z = [1, 60, 3, 4, 5, 12]
    z[0] = 1 (constant), z[1] = 60 (output: 3 * 4 * 5)
    z[2..4] = inputs (3, 4, 5), z[5] = 12 (intermediate: 3 * 4)
"""

from __future__ import annotations

from pathlib import Path

from absl.testing import absltest

from rabbitsnark.circom.wtns import parse_wtns
from rabbitsnark.circom.zkey import parse_zkey
from rabbitsnark.groth16 import compile_circom
from rabbitsnark.groth16.verifier import VerificationKey, verify
from tests.circom.testutil import compute_az_bz


class TestCircomE2EProveVerify(absltest.TestCase):
    """End-to-end: compile_circom -> prove_circom -> verify."""

    def setUp(self):
        test_data_dir = Path(__file__).parent / "data"
        self.zkey = parse_zkey(test_data_dir / "multiplier_3.zkey")
        self.wtns = parse_wtns(test_data_dir / "multiplier_3.wtns")
        self.compiled = compile_circom(self.zkey)
        self.az_mont, self.bz_mont = compute_az_bz(self.zkey, self.wtns)
        self.vk = VerificationKey.from_zkey(self.zkey)

    def test_prove_verify_no_zk(self):
        """Deterministic proof (r=s=0) verifies correctly."""
        proof, public_signals = self.compiled.prove_circom(
            self.wtns,
            self.az_mont,
            self.bz_mont,
            no_zk=True,
        )
        self.assertTrue(verify(self.vk, proof, public_signals))

    def test_prove_verify_with_zk(self):
        """Randomized ZK proof verifies correctly."""
        proof, public_signals = self.compiled.prove_circom(
            self.wtns,
            self.az_mont,
            self.bz_mont,
            no_zk=False,
        )
        self.assertTrue(verify(self.vk, proof, public_signals))

    def test_invalid_signal_rejects(self):
        """Proof with wrong public signals should fail verification."""
        proof, _ = self.compiled.prove_circom(
            self.wtns,
            self.az_mont,
            self.bz_mont,
            no_zk=True,
        )
        # Wrong public signal (99 instead of 60)
        self.assertFalse(verify(self.vk, proof, ["99"]))


if __name__ == "__main__":
    absltest.main()
