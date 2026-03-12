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

from rabbitsnark.gnark import load_gnark_export


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

    def test_r1cs_coo_format(self):
        """r1cs_a COO tuple has consistent shapes."""
        rows, cols, vals = self.data.r1cs_a
        self.assertEqual(rows.shape, cols.shape)
        self.assertEqual(rows.shape[0], vals.shape[0])
        self.assertEqual(rows.dtype, np.uint32)
        self.assertEqual(cols.dtype, np.uint32)
        self.assertEqual(vals.dtype, np.dtype(bn254_sf_mont))

    def test_r1cs_b_coo_format(self):
        """r1cs_b COO tuple has consistent shapes."""
        rows, cols, vals = self.data.r1cs_b
        self.assertEqual(rows.shape, cols.shape)
        self.assertEqual(rows.shape[0], vals.shape[0])

    def test_r1cs_c_coo_format(self):
        """r1cs_c COO tuple has consistent shapes."""
        rows, cols, vals = self.data.r1cs_c
        self.assertEqual(rows.shape, cols.shape)
        self.assertEqual(rows.shape[0], vals.shape[0])

    def test_solution_vectors(self):
        """Solution vectors have correct size (num_constraints)."""
        self.assertLen(self.data.solution_a, 2)
        self.assertLen(self.data.solution_b, 2)
        self.assertLen(self.data.solution_c, 2)

    def test_infinity_masks(self):
        """Infinity masks have correct shape."""
        self.assertEqual(self.data.infinity_a.shape, (4,))
        self.assertEqual(self.data.infinity_b.shape, (4,))

    def test_level_sizes(self):
        """Level sizes are non-empty and sum to num_constraints."""
        self.assertGreater(len(self.data.level_sizes), 0)
        self.assertEqual(self.data.level_sizes.sum(), self.data.num_constraints)

    def test_level_order(self):
        """Level order contains all constraint indices."""
        self.assertEqual(len(self.data.level_order), self.data.num_constraints)


if __name__ == "__main__":
    absltest.main()
