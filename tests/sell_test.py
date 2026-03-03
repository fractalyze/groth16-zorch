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

"""Tests for SELL (Sliced ELL) sparse matrix format and SpMV."""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from zk_dtypes import bn254_sf_mont

from rabbitsnark.circom.wtns import parse_wtns
from rabbitsnark.circom.zkey import parse_zkey
from rabbitsnark.spmv import (
    CSRMatrix,
    SELLMatrix,
    build_r1cs_matrices,
    spmv_sell,
    witness_to_montgomery,
)


def _assert_eq(test_case, actual, expected):
    """Assert two ZK field arrays are element-wise equal."""
    test_case.assertTrue(bool(jnp.all(actual == expected)))


class TestOptimalPartition(absltest.TestCase):
    """Tests for DP-based optimal partition algorithm."""

    def test_uniform_distribution(self):
        """All rows have same NNZ -> single partition (K=1)."""
        lengths = np.array([5, 5, 5, 5])
        partitions = SELLMatrix._optimal_partition(lengths, max_partitions=32)
        self.assertEqual(len(partitions), 1)
        self.assertEqual(partitions[0], (0, 4))

    def test_bimodal_distribution(self):
        """Dense + sparse rows -> DP finds 2 partitions (lower cost)."""
        # P=1: 4 * 50 = 200 elements.  P=2: 2*50 + 2*5 = 110 elements.
        # 110 is well below 200 * 1.05 = 210, so DP picks P=2.
        lengths = np.array([50, 50, 5, 5])
        partitions = SELLMatrix._optimal_partition(lengths, max_partitions=32)
        self.assertEqual(len(partitions), 2)
        self.assertEqual(partitions[0], (0, 2))
        self.assertEqual(partitions[1], (2, 4))

    def test_bimodal_large(self):
        """Dense + sparse rows at scale -> two partitions."""
        lengths = np.sort(
            np.concatenate(
                [
                    np.full(500, 50),
                    np.full(500, 5),
                ]
            )
        )[::-1]
        partitions = SELLMatrix._optimal_partition(lengths, max_partitions=32)
        self.assertEqual(len(partitions), 2)
        self.assertEqual(partitions[0], (0, 500))
        self.assertEqual(partitions[1], (500, 1000))

    def test_max_partitions_cap(self):
        """Partition count capped at max_partitions."""
        lengths = np.array([100, 50, 20, 10, 5, 2, 1])
        partitions = SELLMatrix._optimal_partition(lengths, max_partitions=3)
        self.assertLessEqual(len(partitions), 3)

    def test_empty_rows_covered(self):
        """Empty rows (NNZ=0) included in final partition."""
        lengths = np.array([10, 10, 0, 0])
        partitions = SELLMatrix._optimal_partition(lengths, max_partitions=32)
        last_start, last_end = partitions[-1]
        self.assertEqual(last_end, 4)

    def test_empty_input(self):
        """No rows -> single empty partition."""
        lengths = np.array([])
        partitions = SELLMatrix._optimal_partition(lengths, max_partitions=32)
        self.assertEqual(len(partitions), 1)
        self.assertEqual(partitions[0], (0, 0))

    def test_preserves_coverage(self):
        """All rows accounted for in partitions."""
        lengths = np.sort(
            np.concatenate(
                [
                    np.full(300, 100),
                    np.full(5, 50),
                    np.full(400, 10),
                    np.full(3, 1),
                ]
            )
        )[::-1]
        partitions = SELLMatrix._optimal_partition(lengths, max_partitions=32)
        total = sum(e - s for s, e in partitions)
        self.assertEqual(total, 708)

    def test_dp_beats_naive(self):
        """DP cost is less than or equal to single-partition cost."""
        lengths = np.sort(
            np.concatenate(
                [
                    np.full(100, 500),
                    np.full(900, 2),
                ]
            )
        )[::-1]
        partitions = SELLMatrix._optimal_partition(lengths, max_partitions=32)
        dp_cost = sum((e - s) * max(int(lengths[s]), 1) for s, e in partitions)
        naive_cost = 1000 * 500  # single partition
        self.assertLess(dp_cost, naive_cost)
        # Optimal P=2: 100*500 + 900*2 = 51800 vs naive 500000
        self.assertLessEqual(dp_cost, 51800)

    def test_tolerance_selects_smallest_p(self):
        """Tolerance picks smallest P within threshold of optimum."""
        lengths = np.sort(
            np.concatenate(
                [
                    np.full(100, 100),
                    np.full(100, 50),
                    np.full(100, 10),
                ]
            )
        )[::-1]
        # With tolerance=0 we get maximum partitions; with tolerance=1.0
        # we get P=1.
        p_strict = SELLMatrix._optimal_partition(
            lengths, max_partitions=32, tolerance=0.0
        )
        p_loose = SELLMatrix._optimal_partition(
            lengths, max_partitions=32, tolerance=1.0
        )
        self.assertGreaterEqual(len(p_strict), len(p_loose))


class TestSELLConversion(absltest.TestCase):
    """Tests for CSR -> SELL conversion."""

    def test_basic_conversion(self):
        """Test SELL conversion from a 3x3 sparse matrix."""
        # [[1, 2, 0],
        #  [0, 0, 3],
        #  [4, 0, 5]]
        row_ptrs = np.array([0, 2, 3, 5], dtype=np.int32)
        col_indices = np.array([0, 1, 2, 0, 2], dtype=np.int32)
        values = jnp.array([1, 2, 3, 4, 5], dtype=bn254_sf_mont)
        csr = CSRMatrix.from_arrays(row_ptrs, col_indices, values, 3, 3)

        sell = SELLMatrix.from_csr(csr)

        self.assertEqual(sell.config.n_rows, 3)
        self.assertEqual(sell.config.n_cols, 3)
        self.assertGreater(sell.config.num_partitions, 0)
        # Total partition sizes should equal n_rows
        self.assertEqual(sum(sell.config.partition_sizes), 3)

    def test_inverse_perm_correctness(self):
        """Test that inverse_perm restores original row order."""
        # Create matrix with varying row densities
        # Row 0: 1 nnz, Row 1: 3 nnz, Row 2: 2 nnz
        row_ptrs = np.array([0, 1, 4, 6], dtype=np.int32)
        col_indices = np.array([0, 0, 1, 2, 1, 2], dtype=np.int32)
        values = jnp.array([1, 2, 3, 4, 5, 6], dtype=bn254_sf_mont)
        csr = CSRMatrix.from_arrays(row_ptrs, col_indices, values, 3, 4)

        sell = SELLMatrix.from_csr(csr)

        # After sorting by NNZ descending: row 1 (3), row 2 (2), row 0 (1)
        # inverse_perm should map sorted positions back to original rows
        perm = np.array(sell.inverse_perm)
        self.assertEqual(perm.shape[0], 3)

    def test_stats(self):
        """Test stats computation."""
        row_ptrs = np.array([0, 2, 3, 5], dtype=np.int32)
        col_indices = np.array([0, 1, 2, 0, 2], dtype=np.int32)
        values = jnp.array([1, 2, 3, 4, 5], dtype=bn254_sf_mont)
        csr = CSRMatrix.from_arrays(row_ptrs, col_indices, values, 3, 3)

        sell = SELLMatrix.from_csr(csr)
        stats = sell.stats()

        self.assertEqual(stats["n_rows"], 3)
        self.assertEqual(stats["n_cols"], 3)
        self.assertGreater(stats["num_partitions"], 0)
        self.assertGreaterEqual(stats["memory_savings"], 0.0)
        self.assertLessEqual(stats["memory_savings"], 1.0)

    def test_single_partition_uniform(self):
        """Uniform NNZ -> single partition (P=1)."""
        # 3x4, each row has exactly 2 nnz
        row_ptrs = np.array([0, 2, 4, 6], dtype=np.int32)
        col_indices = np.array([0, 1, 2, 3, 0, 3], dtype=np.int32)
        values = jnp.array([1, 2, 3, 4, 5, 6], dtype=bn254_sf_mont)
        csr = CSRMatrix.from_arrays(row_ptrs, col_indices, values, 3, 4)

        sell = SELLMatrix.from_csr(csr)
        self.assertEqual(sell.config.num_partitions, 1)
        self.assertEqual(sell.config.partition_sizes[0], 3)
        self.assertEqual(sell.config.partition_max_nnz[0], 2)


class TestSELLSpMV(absltest.TestCase):
    """Tests for SELL SpMV correctness."""

    def test_spmv_sell_identity(self):
        """Test SELL SpMV with identity-like sparse matrix."""
        row_ptrs = np.array([0, 1, 2, 3], dtype=np.int32)
        col_indices = np.array([0, 1, 2], dtype=np.int32)
        values = jnp.array([1, 1, 1], dtype=bn254_sf_mont)
        csr = CSRMatrix.from_arrays(row_ptrs, col_indices, values, 3, 3)

        sell = SELLMatrix.from_csr(csr)
        x = jnp.array([10, 20, 30], dtype=bn254_sf_mont)
        y = spmv_sell(sell, x)

        _assert_eq(self, y, x)

    def test_spmv_sell_hand_computed(self):
        """Test SELL SpMV matches hand-computed result.

        A = [[1, 2, 0],
             [0, 0, 3],
             [4, 0, 5]]

        x = [10, 20, 30]

        A * x = [1*10 + 2*20, 3*30, 4*10 + 5*30] = [50, 90, 190]
        """
        row_ptrs = np.array([0, 2, 3, 5], dtype=np.int32)
        col_indices = np.array([0, 1, 2, 0, 2], dtype=np.int32)
        values = jnp.array([1, 2, 3, 4, 5], dtype=bn254_sf_mont)
        csr = CSRMatrix.from_arrays(row_ptrs, col_indices, values, 3, 3)

        sell = SELLMatrix.from_csr(csr)
        x = jnp.array([10, 20, 30], dtype=bn254_sf_mont)
        y = spmv_sell(sell, x)

        expected = jnp.array([50, 90, 190], dtype=bn254_sf_mont)
        _assert_eq(self, y, expected)

    def test_spmv_sell_empty_rows(self):
        """Test SELL SpMV with matrix containing empty rows."""
        row_ptrs = np.array([0, 1, 1, 2, 2], dtype=np.int32)
        col_indices = np.array([0, 1], dtype=np.int32)
        values = jnp.array([3, 7], dtype=bn254_sf_mont)
        csr = CSRMatrix.from_arrays(row_ptrs, col_indices, values, 4, 3)

        sell = SELLMatrix.from_csr(csr)
        x = jnp.array([10, 20, 30], dtype=bn254_sf_mont)
        y = spmv_sell(sell, x)

        expected = jnp.array([30, 0, 140, 0], dtype=bn254_sf_mont)
        _assert_eq(self, y, expected)

    def test_spmv_sell_field_arithmetic(self):
        """Test SELL SpMV with field arithmetic (negative values mod p)."""
        row_ptrs = np.array([0, 1, 2], dtype=np.int32)
        col_indices = np.array([0, 1], dtype=np.int32)
        values = jnp.array([-1, 2], dtype=bn254_sf_mont)
        csr = CSRMatrix.from_arrays(row_ptrs, col_indices, values, 2, 2)

        sell = SELLMatrix.from_csr(csr)
        x = jnp.array([5, 7], dtype=bn254_sf_mont)
        y = spmv_sell(sell, x)

        expected = jnp.array([-5, 14], dtype=bn254_sf_mont)
        _assert_eq(self, y, expected)

    def test_spmv_sell_single_dense_row(self):
        """Test matrix with one very dense row and many sparse rows.

        A = [[1, 2, 3, 4],
             [5, 0, 0, 0],
             [0, 6, 0, 0],
             [0, 0, 7, 0]]

        x = [1, 2, 3, 4]

        A * x = [1+4+9+16, 5, 12, 21] = [30, 5, 12, 21]
        """
        row_ptrs = np.array([0, 4, 5, 6, 7], dtype=np.int32)
        col_indices = np.array([0, 1, 2, 3, 0, 1, 2], dtype=np.int32)
        values = jnp.array([1, 2, 3, 4, 5, 6, 7], dtype=bn254_sf_mont)
        csr = CSRMatrix.from_arrays(row_ptrs, col_indices, values, 4, 4)

        sell = SELLMatrix.from_csr(csr)
        x = jnp.array([1, 2, 3, 4], dtype=bn254_sf_mont)
        y = spmv_sell(sell, x)

        expected = jnp.array([30, 5, 12, 21], dtype=bn254_sf_mont)
        _assert_eq(self, y, expected)


class TestSELLR1CSIntegration(absltest.TestCase):
    """Integration tests with multiplier_3 test data."""

    def setUp(self):
        self.test_data_dir = Path(__file__).parent / "data"

    def test_sell_r1cs_satisfaction(self):
        """Test R1CS satisfaction using SELL SpMV: Az . Bz = expected."""
        zkey = parse_zkey(self.test_data_dir / "multiplier_3.zkey")
        wtns = parse_wtns(self.test_data_dir / "multiplier_3.wtns")

        A, B = build_r1cs_matrices(zkey, bn254_sf_mont)
        z_mont = witness_to_montgomery(wtns.witnesses, bn254_sf_mont)

        sell_A = SELLMatrix.from_csr(A)
        sell_B = SELLMatrix.from_csr(B)

        Az = spmv_sell(sell_A, z_mont)
        Bz = spmv_sell(sell_B, z_mont)
        AzBz = Az * Bz

        expected = jnp.array(
            [-12, -60, 0, 0],
            dtype=bn254_sf_mont,
        )
        _assert_eq(self, AzBz, expected)


if __name__ == "__main__":
    absltest.main()
