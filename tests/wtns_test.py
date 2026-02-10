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

"""Tests for wtns parser."""

import tempfile
from pathlib import Path

from absl.testing import absltest

from rabbitsnark.circom.base import Modulus
from rabbitsnark.circom.wtns import WtnsV2, parse_wtns

# BN254 scalar field modulus (Fr)
BN254_FR_MODULUS = (
    21888242871839275222246405745257275088548364400416034343698204186575808495617
)


class TestWtnsParser(absltest.TestCase):
    """Tests for wtns file parsing."""

    def setUp(self):
        """Set up test data directory."""
        self.test_data_dir = Path(__file__).parent / "data"

    def test_parse_multiplier_3(self):
        """Test parsing multiplier_3.wtns file.

        The file was generated with input { "in": ["3", "4", "5"] }.
        """
        wtns_path = self.test_data_dir / "multiplier_3.wtns"
        wtns = parse_wtns(wtns_path)

        self.assertEqual(wtns.version, 2)
        self.assertIsInstance(wtns, WtnsV2)

        # Check header
        expected_modulus = Modulus.from_int(BN254_FR_MODULUS)
        self.assertEqual(wtns.header.modulus, expected_modulus)
        self.assertEqual(wtns.header.num_witness, 6)

        # Check witnesses
        # witness[0] = 1 (constant)
        # witness[1] = 60 (output: 3 * 4 * 5 = 60)
        # witness[2] = 3 (input[0])
        # witness[3] = 4 (input[1])
        # witness[4] = 5 (input[2])
        # witness[5] = 12 (intermediate: 3 * 4 = 12)
        expected_witnesses = [1, 60, 3, 4, 5, 12]
        self.assertEqual(wtns.witnesses, expected_witnesses)
        self.assertEqual(wtns.num_witness, 6)

    def test_parse_invalid_magic(self):
        """Test that invalid magic raises ValueError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            invalid_file = Path(tmp_dir) / "invalid.wtns"
            invalid_file.write_bytes(b"xxxx")

            with self.assertRaisesRegex(ValueError, "Invalid magic"):
                parse_wtns(invalid_file)

    def test_parse_unsupported_version(self):
        """Test that unsupported version raises ValueError."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            invalid_file = Path(tmp_dir) / "invalid.wtns"
            # Magic + version 99
            invalid_file.write_bytes(b"wtns" + (99).to_bytes(4, "little"))

            with self.assertRaisesRegex(ValueError, "Unsupported version"):
                parse_wtns(invalid_file)


if __name__ == "__main__":
    absltest.main()
