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

"""Tests for sparse matrix-vector multiplication (SpMV).

Note: bn254_sf_mont dtype auto-converts to Montgomery form on array creation,
so standard-form values are passed directly without explicit to_montgomery().
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
from absl.testing import absltest
from zk_dtypes import bn254_sf_mont

from rabbitsnark.circom.wtns import parse_wtns
from rabbitsnark.circom.zkey import parse_zkey
from rabbitsnark.spmv import (
    CSRMatrix,
    build_r1cs_matrices,
    spmv,
    spmv_backend,
    witness_to_montgomery,
)

BN254_FR_MODULUS = (
    21888242871839275222246405745257275088548364400416034343698204186575808495617
)


def _assert_eq(test_case, actual, expected):
    """Assert two ZK field arrays are element-wise equal."""
    test_case.assertTrue(bool(jnp.all(actual == expected)))


class TestCSRMatrix(absltest.TestCase):
    """Tests for CSR matrix construction."""

    def test_from_arrays_basic(self):
        """Test CSR construction from raw arrays."""
        # 3×3 matrix:
        #   [[1, 2, 0],
        #    [0, 0, 3],
        #    [4, 0, 5]]
        row_ptrs = np.array([0, 2, 3, 5], dtype=np.int32)
        col_indices = np.array([0, 1, 2, 0, 2], dtype=np.int32)
        values = jnp.array([1, 2, 3, 4, 5], dtype=bn254_sf_mont)

        csr = CSRMatrix.from_arrays(row_ptrs, col_indices, values, 3, 3)

        self.assertEqual(csr.n_rows, 3)
        self.assertEqual(csr.n_cols, 3)
        self.assertEqual(csr.nnz, 5)

    def test_from_arrays_empty_rows(self):
        """Test CSR with empty rows."""
        # 4×3 matrix with rows 1 and 3 empty:
        #   [[1, 0, 0],
        #    [0, 0, 0],
        #    [0, 2, 0],
        #    [0, 0, 0]]
        row_ptrs = np.array([0, 1, 1, 2, 2], dtype=np.int32)
        col_indices = np.array([0, 1], dtype=np.int32)
        values = jnp.array([1, 2], dtype=bn254_sf_mont)

        csr = CSRMatrix.from_arrays(row_ptrs, col_indices, values, 4, 3)

        self.assertEqual(csr.nnz, 2)

    def test_from_coefficients(self):
        """Test CSR construction from Coefficient objects."""
        from rabbitsnark.circom.zkey.coefficient import Coefficient

        coefficients = [
            Coefficient.from_ints(matrix=0, constraint=0, signal=1, value=3),
            Coefficient.from_ints(matrix=0, constraint=1, signal=0, value=7),
            Coefficient.from_ints(matrix=0, constraint=0, signal=0, value=5),
            # Matrix B entry (should be filtered out)
            Coefficient.from_ints(matrix=1, constraint=0, signal=2, value=9),
        ]

        csr = CSRMatrix.from_coefficients(
            coefficients,
            is_matrix_a=True,
            n_rows=2,
            n_cols=3,
            dtype=bn254_sf_mont,
            modulus=BN254_FR_MODULUS,
        )

        self.assertEqual(csr.n_rows, 2)
        self.assertEqual(csr.n_cols, 3)
        self.assertEqual(csr.nnz, 3)
        # Sorted by (constraint, signal): (0,0)=5, (0,1)=3, (1,0)=7
        self.assertTrue(bool(jnp.all(csr.col_indices == jnp.array([0, 1, 0]))))
        np.testing.assert_array_equal(csr.row_ptrs, [0, 2, 3])

        # Values should match standard-form inputs (dtype handles Montgomery)
        expected_vals = jnp.array([5, 3, 7], dtype=bn254_sf_mont)
        _assert_eq(self, csr.values, expected_vals)


class TestSpMV(absltest.TestCase):
    """Tests for sparse matrix-vector multiplication."""

    def test_spmv_identity(self):
        """Test SpMV with identity-like sparse matrix."""
        # 3×3 identity
        row_ptrs = np.array([0, 1, 2, 3], dtype=np.int32)
        col_indices = np.array([0, 1, 2], dtype=np.int32)
        values = jnp.array([1, 1, 1], dtype=bn254_sf_mont)
        csr = CSRMatrix.from_arrays(row_ptrs, col_indices, values, 3, 3)

        x = jnp.array([10, 20, 30], dtype=bn254_sf_mont)
        y = spmv(csr, x)

        _assert_eq(self, y, x)

    def test_spmv_hand_computed(self):
        """Test SpMV with hand-computed 3×3 matrix.

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

        x = jnp.array([10, 20, 30], dtype=bn254_sf_mont)
        y = spmv(csr, x)

        expected = jnp.array([50, 90, 190], dtype=bn254_sf_mont)
        _assert_eq(self, y, expected)

    def test_spmv_empty_rows(self):
        """Test SpMV with matrix containing empty rows."""
        # 4×3, rows 1 and 3 empty
        row_ptrs = np.array([0, 1, 1, 2, 2], dtype=np.int32)
        col_indices = np.array([0, 1], dtype=np.int32)
        values = jnp.array([3, 7], dtype=bn254_sf_mont)
        csr = CSRMatrix.from_arrays(row_ptrs, col_indices, values, 4, 3)

        x = jnp.array([10, 20, 30], dtype=bn254_sf_mont)
        y = spmv(csr, x)

        expected = jnp.array([30, 0, 140, 0], dtype=bn254_sf_mont)
        _assert_eq(self, y, expected)

    def test_spmv_field_arithmetic(self):
        """Test SpMV with field arithmetic (negative values mod p)."""
        p = BN254_FR_MODULUS

        # 2×2 matrix: [[-1, 0], [0, 2]]
        row_ptrs = np.array([0, 1, 2], dtype=np.int32)
        col_indices = np.array([0, 1], dtype=np.int32)
        # -1 mod p = p - 1; bn254_sf_mont auto-converts to Montgomery form
        values = jnp.array([p - 1, 2], dtype=bn254_sf_mont)
        csr = CSRMatrix.from_arrays(row_ptrs, col_indices, values, 2, 2)

        x = jnp.array([5, 7], dtype=bn254_sf_mont)
        y = spmv(csr, x)

        # Expected: [-5, 14] = [p-5, 14] in standard form
        expected = jnp.array([p - 5, 14], dtype=bn254_sf_mont)
        _assert_eq(self, y, expected)


class TestR1CSIntegration(absltest.TestCase):
    """Integration tests with multiplier_3 test data.

    Circuit: multiplier_3 (3-input multiplier)
    Witnesses: z = [1, 60, 3, 4, 5, 12]
        z[0] = 1 (constant)
        z[1] = 60 (output: 3 * 4 * 5)
        z[2] = 3 (input[0])
        z[3] = 4 (input[1])
        z[4] = 5 (input[2])
        z[5] = 12 (intermediate: 3 * 4)

    Coefficients from zkey:
        A: (0,2)=-1, (1,5)=-1, (2,0)=1, (3,1)=1
        B: (0,3)=1, (1,4)=1

    Expected:
        Az = [-3, -12, 1, 60] (mod p)
        Bz = [4, 5, 0, 0]
        Az . Bz = [-12, -60, 0, 0] (mod p)
    """

    def setUp(self):
        self.test_data_dir = Path(__file__).parent / "data"
        self.p = BN254_FR_MODULUS

    def test_build_r1cs_matrices(self):
        """Test building A and B matrices from zkey."""
        zkey = parse_zkey(self.test_data_dir / "multiplier_3.zkey")
        A, B = build_r1cs_matrices(zkey, bn254_sf_mont)

        # A has 4 nonzeros in a 4×6 matrix
        self.assertEqual(A.n_rows, 4)
        self.assertEqual(A.n_cols, 6)
        self.assertEqual(A.nnz, 4)

        # B has 2 nonzeros in a 4×6 matrix
        self.assertEqual(B.n_rows, 4)
        self.assertEqual(B.n_cols, 6)
        self.assertEqual(B.nnz, 2)

    def test_witness_to_montgomery(self):
        """Test witness conversion to Montgomery form."""
        wtns = parse_wtns(self.test_data_dir / "multiplier_3.wtns")
        z_mont = witness_to_montgomery(wtns.witnesses, bn254_sf_mont, self.p)

        self.assertEqual(z_mont.shape, (6,))

        # Values should match standard-form witnesses (dtype handles Montgomery)
        expected = jnp.array([1, 60, 3, 4, 5, 12], dtype=bn254_sf_mont)
        _assert_eq(self, z_mont, expected)

    def test_spmv_multiplier_3(self):
        """Test full SpMV pipeline with multiplier_3 circuit.

        Verifies Az and Bz produce correct results.
        """
        zkey = parse_zkey(self.test_data_dir / "multiplier_3.zkey")
        wtns = parse_wtns(self.test_data_dir / "multiplier_3.wtns")

        A, B = build_r1cs_matrices(zkey, bn254_sf_mont)
        z_mont = witness_to_montgomery(wtns.witnesses, bn254_sf_mont, self.p)

        Az = spmv(A, z_mont)
        Bz = spmv(B, z_mont)

        # Az = [-3, -12, 1, 60] = [p-3, p-12, 1, 60] in standard form
        expected_Az = jnp.array(
            [self.p - 3, self.p - 12, 1, 60],
            dtype=bn254_sf_mont,
        )
        _assert_eq(self, Az, expected_Az)

        # Bz = [4, 5, 0, 0]
        expected_Bz = jnp.array([4, 5, 0, 0], dtype=bn254_sf_mont)
        _assert_eq(self, Bz, expected_Bz)

    def test_r1cs_satisfaction(self):
        """Test that Az . Bz = Cz (R1CS satisfaction).

        For multiplier_3, the C matrix has:
        - No explicit C coefficients in the zkey (C is computed differently)

        Instead, verify the Hadamard product Az . Bz matches expected values:
            Az . Bz = [(-3)*4, (-12)*5, 1*0, 60*0]
                    = [-12, -60, 0, 0] (mod p)
        """
        zkey = parse_zkey(self.test_data_dir / "multiplier_3.zkey")
        wtns = parse_wtns(self.test_data_dir / "multiplier_3.wtns")

        A, B = build_r1cs_matrices(zkey, bn254_sf_mont)
        z_mont = witness_to_montgomery(wtns.witnesses, bn254_sf_mont, self.p)

        Az = spmv(A, z_mont)
        Bz = spmv(B, z_mont)

        # Hadamard product (pointwise multiplication)
        AzBz = Az * Bz

        expected = jnp.array(
            [self.p - 12, self.p - 60, 0, 0],
            dtype=bn254_sf_mont,
        )
        _assert_eq(self, AzBz, expected)


class TestSpMVBackend(absltest.TestCase):
    """Tests for ZKX backend native CSR SpMV."""

    def test_backend_hand_computed(self):
        """Test backend SpMV with hand-computed 3×3 matrix.

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

        x = jnp.array([10, 20, 30], dtype=bn254_sf_mont)
        y = spmv_backend(csr, x)

        expected = jnp.array([50, 90, 190], dtype=bn254_sf_mont)
        _assert_eq(self, y, expected)

    def test_backend_identity(self):
        """Test backend SpMV with identity matrix."""
        row_ptrs = np.array([0, 1, 2, 3], dtype=np.int32)
        col_indices = np.array([0, 1, 2], dtype=np.int32)
        values = jnp.array([1, 1, 1], dtype=bn254_sf_mont)
        csr = CSRMatrix.from_arrays(row_ptrs, col_indices, values, 3, 3)

        x = jnp.array([10, 20, 30], dtype=bn254_sf_mont)
        y = spmv_backend(csr, x)

        _assert_eq(self, y, x)

    def test_backend_field_arithmetic(self):
        """Test backend SpMV with field arithmetic (negative values mod p)."""
        p = BN254_FR_MODULUS

        # 2×2 matrix: [[-1, 0], [0, 2]]
        row_ptrs = np.array([0, 1, 2], dtype=np.int32)
        col_indices = np.array([0, 1], dtype=np.int32)
        values = jnp.array([p - 1, 2], dtype=bn254_sf_mont)
        csr = CSRMatrix.from_arrays(row_ptrs, col_indices, values, 2, 2)

        x = jnp.array([5, 7], dtype=bn254_sf_mont)
        y = spmv_backend(csr, x)

        expected = jnp.array([p - 5, 14], dtype=bn254_sf_mont)
        _assert_eq(self, y, expected)

    def test_backend_multiplier_3(self):
        """Test backend SpMV with multiplier_3 circuit data."""
        test_data_dir = Path(__file__).parent / "data"
        p = BN254_FR_MODULUS

        zkey = parse_zkey(test_data_dir / "multiplier_3.zkey")
        wtns = parse_wtns(test_data_dir / "multiplier_3.wtns")

        A, B = build_r1cs_matrices(zkey, bn254_sf_mont)
        z_mont = witness_to_montgomery(wtns.witnesses, bn254_sf_mont, p)

        Az = spmv_backend(A, z_mont)
        Bz = spmv_backend(B, z_mont)

        expected_Az = jnp.array(
            [p - 3, p - 12, 1, 60],
            dtype=bn254_sf_mont,
        )
        _assert_eq(self, Az, expected_Az)

        expected_Bz = jnp.array([4, 5, 0, 0], dtype=bn254_sf_mont)
        _assert_eq(self, Bz, expected_Bz)


if __name__ == "__main__":
    absltest.main()
