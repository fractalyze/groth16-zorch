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

"""Tests for NTT (Number Theoretic Transform) implementations."""

import math
import random

import jax.numpy as jnp
from absl.testing import absltest
from zk_dtypes import bn254_sf_mont, pfinfo

from rabbitsnark.ntt import (
    BN254_FR_ROOT_OF_UNITY,
    NTT,
    batch_ntt,
    coset_intt,
    coset_ntt,
)


def _assert_eq(test_case, actual, expected):
    """Assert two ZK field arrays are element-wise equal."""
    test_case.assertTrue(bool(jnp.all(actual == expected)))


def _random_field_elements(n, dtype, seed=42):
    """Generate deterministic random field elements."""
    rng = random.Random(seed)
    return jnp.array([rng.randint(1, 10**6) for _ in range(n)], dtype=dtype)


def _get_omega(ntt_instance, n):
    """Compute n-th root of unity from NTT instance."""
    log_n = int(math.log2(n))
    dtype = ntt_instance.DTYPE
    omega = dtype(ntt_instance.ROOT_OF_UNITY)
    for _ in range(ntt_instance.MAX_LOG_N - log_n):
        omega = omega * omega
    return omega


def _naive_ntt(coeffs, ntt_instance):
    """O(n²) reference NTT: result[k] = sum_j coeffs[j] * omega^(j * k)."""
    n = coeffs.shape[0]
    dtype = ntt_instance.DTYPE
    one = dtype(1)
    omega = _get_omega(ntt_instance, n)

    # Precompute omega powers: omega_powers[i] = omega^i for i in [0, n)
    omega_powers = [one]
    for _ in range(1, n):
        omega_powers.append(omega_powers[-1] * omega)

    result = []
    for k in range(n):
        val = dtype(0)
        for j in range(n):
            val = val + coeffs[j] * omega_powers[(j * k) % n]
        result.append(val)

    return jnp.array(result)


def _naive_intt(evals, ntt_instance):
    """
    O(n²) reference inverse NTT: coeffs[k] = (1/n) * sum_j evals[j] * omega^(-j * k).
    """
    n = evals.shape[0]
    dtype = ntt_instance.DTYPE
    one = dtype(1)
    omega_inv = one / _get_omega(ntt_instance, n)

    # Precompute inverse omega powers
    omega_inv_powers = [one]
    for _ in range(1, n):
        omega_inv_powers.append(omega_inv_powers[-1] * omega_inv)

    inv_n = one / dtype(n)
    result = []
    for k in range(n):
        val = dtype(0)
        for j in range(n):
            val = val + evals[j] * omega_inv_powers[(j * k) % n]
        result.append(val * inv_n)

    return jnp.array(result)


def _naive_coset_ntt(coeffs, ntt_instance, shift):
    """Naive coset NTT: NTT(f * [shift⁰, shift¹, ..., shift^(n - 1)])."""
    n = coeffs.shape[0]
    dtype = coeffs.dtype
    one = dtype.type(1)

    shift_powers = [one]
    for _ in range(1, n):
        shift_powers.append(shift_powers[-1] * shift)

    return _naive_ntt(coeffs * jnp.array(shift_powers), ntt_instance)


def _naive_coset_intt(evals, ntt_instance, shift):
    """Naive coset INTT: INTT(v) * [shift⁻⁰, shift⁻¹, ..., shift^(-(n - 1))]."""
    n = evals.shape[0]
    dtype = evals.dtype
    one = dtype.type(1)
    shift_inv = one / shift

    inv_shift_powers = [one]
    for _ in range(1, n):
        inv_shift_powers.append(inv_shift_powers[-1] * shift_inv)

    return _naive_intt(evals, ntt_instance) * jnp.array(inv_shift_powers)


class TestNTT(absltest.TestCase):
    """Tests for BN254 scalar field NTT implementation."""

    def setUp(self):
        """Create NTT instance."""
        self.ntt = NTT(bn254_sf_mont, BN254_FR_ROOT_OF_UNITY)

    def test_forward_ntt(self):
        """Test forward NTT matches naive O(n²) DFT."""
        coeffs = _random_field_elements(8, bn254_sf_mont)

        expected = _naive_ntt(coeffs, self.ntt)
        actual = self.ntt.forward(coeffs)

        _assert_eq(self, actual, expected)

    def test_inverse_ntt(self):
        """Test inverse NTT matches naive O(n²) inverse DFT."""
        evals = _random_field_elements(8, bn254_sf_mont, seed=99)

        expected = _naive_intt(evals, self.ntt)
        actual = self.ntt.inverse(evals)

        _assert_eq(self, actual, expected)

    def test_ntt_unified_interface(self):
        """Test the unified ntt() method."""
        coeffs = _random_field_elements(4, bn254_sf_mont)

        evals = self.ntt.ntt(coeffs, inverse=False)
        recovered = self.ntt.ntt(evals, inverse=True)

        _assert_eq(self, recovered, coeffs)

    def test_root_of_unity_property(self):
        """Test that root of unity has correct order."""
        p = pfinfo(bn254_sf_mont).modulus

        # root^(2²⁸) should equal 1 (mod p)
        result = pow(BN254_FR_ROOT_OF_UNITY, 1 << 28, p)
        self.assertEqual(result, 1)

        # root^(2²⁷) should not equal 1 (primitive root check)
        result_half = pow(BN254_FR_ROOT_OF_UNITY, 1 << 27, p)
        self.assertNotEqual(result_half, 1)

    def test_bit_reversal(self):
        """Test bit reversal permutation."""
        # For n=8, bit reversal should be: [0, 4, 2, 6, 1, 5, 3, 7]
        coeffs = jnp.array(list(range(8)), dtype=bn254_sf_mont)
        reversed_coeffs = self.ntt.bit_reverse(coeffs)
        expected = jnp.array([0, 4, 2, 6, 1, 5, 3, 7], dtype=bn254_sf_mont)
        _assert_eq(self, reversed_coeffs, expected)

    def test_pytree_flatten_unflatten(self):
        """Test JAX pytree registration."""
        ntt = NTT(bn254_sf_mont, BN254_FR_ROOT_OF_UNITY)
        children, aux_data = ntt.tree_flatten()
        recovered = NTT.tree_unflatten(aux_data, children)

        coeffs = _random_field_elements(4, bn254_sf_mont)
        _assert_eq(self, ntt.forward(coeffs), recovered.forward(coeffs))


class TestCosetNTT(absltest.TestCase):
    """Tests for coset NTT utilities."""

    def setUp(self):
        """Create NTT instance and coset shift."""
        self.ntt = NTT(bn254_sf_mont, BN254_FR_ROOT_OF_UNITY)
        pf = pfinfo(bn254_sf_mont)
        shift_int = pow(BN254_FR_ROOT_OF_UNITY, 1 << (pf.two_adicity - 4), pf.modulus)
        self.shift = bn254_sf_mont(shift_int)

    def test_coset_ntt(self):
        """Test coset NTT matches naive coset NTT."""
        coeffs = _random_field_elements(8, bn254_sf_mont)

        expected = _naive_coset_ntt(coeffs, self.ntt, self.shift)
        actual = coset_ntt(self.ntt, coeffs, self.shift)

        _assert_eq(self, actual, expected)

    def test_coset_intt(self):
        """Test coset INTT matches naive coset INTT."""
        evals = _random_field_elements(8, bn254_sf_mont, seed=99)

        expected = _naive_coset_intt(evals, self.ntt, self.shift)
        actual = coset_intt(self.ntt, evals, self.shift)

        _assert_eq(self, actual, expected)


class TestBatchNTT(absltest.TestCase):
    """Tests for batch NTT operations."""

    def setUp(self):
        """Create NTT instance."""
        self.ntt = NTT(bn254_sf_mont, BN254_FR_ROOT_OF_UNITY)

    def test_batch_forward_ntt(self):
        """Test batch forward NTT matches per-row naive NTT."""
        batch = jnp.array(
            [
                _random_field_elements(4, bn254_sf_mont, seed=1),
                _random_field_elements(4, bn254_sf_mont, seed=2),
                _random_field_elements(4, bn254_sf_mont, seed=3),
            ],
        )

        result = batch_ntt(self.ntt, batch, inverse=False)

        for i in range(batch.shape[0]):
            expected = _naive_ntt(batch[i], self.ntt)
            _assert_eq(self, result[i], expected)

    def test_batch_inverse_ntt(self):
        """Test batch inverse NTT matches per-row naive INTT."""
        batch = jnp.array(
            [
                _random_field_elements(4, bn254_sf_mont, seed=10),
                _random_field_elements(4, bn254_sf_mont, seed=20),
                _random_field_elements(4, bn254_sf_mont, seed=30),
            ],
        )

        result = batch_ntt(self.ntt, batch, inverse=True)

        for i in range(batch.shape[0]):
            expected = _naive_intt(batch[i], self.ntt)
            _assert_eq(self, result[i], expected)

    def test_batch_ntt_preserves_shape(self):
        """Test that batch NTT preserves array shape."""
        batch = jnp.zeros((5, 16), dtype=bn254_sf_mont)
        result = batch_ntt(self.ntt, batch, inverse=False)
        self.assertEqual(result.shape, batch.shape)


if __name__ == "__main__":
    absltest.main()
