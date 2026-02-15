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

"""Tests for Groth16 proof generation.

Uses the multiplier_3 circuit (3-input multiplier):
    z = [1, 60, 3, 4, 5, 12]
    z[0] = 1 (constant), z[1] = 60 (output: 3 * 4 * 5)
    z[2..4] = inputs (3, 4, 5), z[5] = 12 (intermediate: 3 * 4)
"""

from pathlib import Path

from absl.testing import absltest

from rabbitsnark.circom.wtns import parse_wtns
from rabbitsnark.circom.zkey import parse_zkey
from rabbitsnark.groth16 import Groth16Proof, prove, write_public_signals


class TestGroth16Prove(absltest.TestCase):
    """Tests for Groth16 proof generation."""

    def setUp(self):
        self.test_data_dir = Path(__file__).parent / "data"
        self.zkey = parse_zkey(self.test_data_dir / "multiplier_3.zkey")
        self.wtns = parse_wtns(self.test_data_dir / "multiplier_3.wtns")

    def test_prove_no_zk(self):
        """Deterministic proof (r=s=0) produces non-zero points."""
        proof, public_signals = prove(self.zkey, self.wtns, no_zk=True)

        self.assertIsInstance(proof, Groth16Proof)

        # Verify JSON serialization produces non-trivial coordinates
        proof_json = proof.to_json()
        self.assertNotEqual(proof_json["pi_a"][0], "0")
        self.assertNotEqual(proof_json["pi_a"][1], "0")
        self.assertNotEqual(proof_json["pi_c"][0], "0")
        self.assertNotEqual(proof_json["pi_c"][1], "0")
        self.assertEqual(proof_json["protocol"], "groth16")
        self.assertEqual(proof_json["curve"], "bn128")

    def test_public_signals(self):
        """Public signals are correctly extracted from witness."""
        _, public_signals = prove(self.zkey, self.wtns, no_zk=True)

        # multiplier_3 has 1 public input: the output 60
        self.assertEqual(public_signals, ["60"])

    def test_write_public_signals(self):
        """write_public_signals extracts z[1:l+1]."""
        witnesses = [1, 60, 3, 4, 5, 12]
        signals = write_public_signals(witnesses, num_public=1)
        self.assertEqual(signals, ["60"])

    def test_prove_with_zk(self):
        """ZK proof (random r, s) differs from no-ZK proof."""
        proof_no_zk, _ = prove(self.zkey, self.wtns, no_zk=True)
        proof_zk, _ = prove(self.zkey, self.wtns, no_zk=False)

        no_zk_json = proof_no_zk.to_json()
        zk_json = proof_zk.to_json()

        # Random blinding should change at least pi_a
        self.assertNotEqual(no_zk_json["pi_a"], zk_json["pi_a"])

    def test_proof_json_structure(self):
        """Proof JSON has snarkjs-compatible structure."""
        proof, _ = prove(self.zkey, self.wtns, no_zk=True)
        proof_json = proof.to_json()

        # pi_a: [x, y, "1"]
        self.assertEqual(len(proof_json["pi_a"]), 3)
        self.assertEqual(proof_json["pi_a"][2], "1")

        # pi_b: [[x0, x1], [y0, y1], ["1", "0"]]
        self.assertEqual(len(proof_json["pi_b"]), 3)
        self.assertEqual(len(proof_json["pi_b"][0]), 2)
        self.assertEqual(len(proof_json["pi_b"][1]), 2)
        self.assertEqual(proof_json["pi_b"][2], ["1", "0"])

        # pi_c: [x, y, "1"]
        self.assertEqual(len(proof_json["pi_c"]), 3)
        self.assertEqual(proof_json["pi_c"][2], "1")


if __name__ == "__main__":
    absltest.main()
