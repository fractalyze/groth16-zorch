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

"""Tests for Groth16 proof verification.

Uses the multiplier_3 circuit (3-input multiplier):
    z = [1, 60, 3, 4, 5, 12]
    z[0] = 1 (constant), z[1] = 60 (output: 3 * 4 * 5)
    z[2..4] = inputs (3, 4, 5), z[5] = 12 (intermediate: 3 * 4)
"""

import json
from pathlib import Path

from absl.testing import absltest

from rabbitsnark.circom.zkey import parse_zkey
from rabbitsnark.groth16 import VerificationKey, verify


def _build_vk_json(zkey):
    """Build a snarkjs-format verification_key.json dict from a zkey."""
    vk = zkey.verifying_key

    def _g1_to_json(pt):
        return [str(pt.x), str(pt.y), "1"]

    def _g2_to_json(pt):
        return [
            [str(pt.x[0]), str(pt.x[1])],
            [str(pt.y[0]), str(pt.y[1])],
            ["1", "0"],
        ]

    return {
        "protocol": "groth16",
        "curve": "bn128",
        "nPublic": zkey.header_groth.num_public_inputs,
        "vk_alpha_1": _g1_to_json(vk.alpha_g1),
        "vk_beta_2": _g2_to_json(vk.beta_g2),
        "vk_gamma_2": _g2_to_json(vk.gamma_g2),
        "vk_delta_2": _g2_to_json(vk.delta_g2),
        "IC": [_g1_to_json(pt) for pt in zkey.ic],
    }


class TestGroth16Verify(absltest.TestCase):
    """Tests for Groth16 proof verification using snarkjs-generated proofs."""

    def setUp(self):
        test_data_dir = Path(__file__).parent / "data"
        self.zkey = parse_zkey(test_data_dir / "multiplier_3.zkey")

        with open(test_data_dir / "multiplier_3_proof.json") as f:
            self.proof_json = json.load(f)
        with open(test_data_dir / "multiplier_3_public.json") as f:
            self.public_signals = json.load(f)

    def test_valid_proof(self):
        """A snarkjs-generated proof verifies successfully."""
        vk = VerificationKey.from_zkey(self.zkey)
        self.assertTrue(verify(vk, self.proof_json, self.public_signals))

    def test_tampered_public_signal(self):
        """Tampering with a public signal makes verification fail."""
        vk = VerificationKey.from_zkey(self.zkey)
        self.assertFalse(verify(vk, self.proof_json, ["61"]))

    def test_from_json(self):
        """VerificationKey.from_json builds a VK that verifies proofs."""
        vk_json = _build_vk_json(self.zkey)
        vk = VerificationKey.from_json(vk_json)
        self.assertTrue(verify(vk, self.proof_json, self.public_signals))


if __name__ == "__main__":
    absltest.main()
