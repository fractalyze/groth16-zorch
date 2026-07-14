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

"""Tests for gnark binary export loader.

Uses the tiny_multiply fixture (x × x == y, x=3, y=9):
    4 wires, 2 constraints, domain_size=2
    num_public=2, num_secret=1, num_internal=1
"""

from pathlib import Path

import numpy as np
from absl.testing import absltest
from zk_dtypes import bn254_sf_mont

from groth16_zorch.gnark import load_gnark_export


class TestGnarkLoader(absltest.TestCase):
    """Tests for gnark export loading — CPU only, no require-zorch."""

    def setUp(self):
        self.data_dir = Path(__file__).parent / "data" / "tiny_multiply"
        self.data = load_gnark_export(self.data_dir)

    def test_load_metadata(self):
        """metadata.json is parsed correctly."""
        self.assertEqual(self.data.num_wires, 4)
        self.assertEqual(self.data.num_constraints, 2)
        self.assertEqual(self.data.domain_size, 2)
        self.assertEqual(self.data.num_public, 2)
        self.assertEqual(self.data.num_secret, 1)
        self.assertEqual(self.data.num_internal, 1)

    def test_witness_shape(self):
        """witness_full has correct shape and dtype."""
        self.assertEqual(self.data.witness_full.shape, (4,))
        self.assertEqual(self.data.witness_full.dtype, np.dtype(bn254_sf_mont))

    def test_pk_a_g1_count(self):
        """pk_a_g1 length matches num_wires."""
        self.assertLen(self.data.pk_a_g1, 4)

    def test_pk_b_g1_count(self):
        """pk_b_g1 length matches num_wires."""
        self.assertLen(self.data.pk_b_g1, 4)

    def test_pk_b_g2_count(self):
        """pk_b_g2 length matches num_wires."""
        self.assertLen(self.data.pk_b_g2, 4)

    def test_vk_ic_count(self):
        """vk_ic has correct number of points."""
        self.assertGreater(len(self.data.vk_ic), 0)

    def test_infinity_masks(self):
        """Infinity masks have correct shape."""
        self.assertEqual(self.data.infinity_a.shape, (4,))
        self.assertEqual(self.data.infinity_b.shape, (4,))


if __name__ == "__main__":
    absltest.main()
