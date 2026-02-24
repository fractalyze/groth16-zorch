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
    _forward_ntt,
    _inverse_ntt,
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
        val = jnp.zeros((), dtype=dtype)
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
        val = jnp.zeros((), dtype=dtype)
        for j in range(n):
            val = val + evals[j] * omega_inv_powers[(j * k) % n]
        result.append(val * inv_n)

    return jnp.array(result)


class TestNTT(absltest.TestCase):
    """Tests for BN254 scalar field NTT implementation."""

    def setUp(self):
        """Create NTT instance."""
        self.ntt = NTT(bn254_sf_mont, BN254_FR_ROOT_OF_UNITY)

    def test_forward_ntt(self):
        """Test forward NTT matches naive O(n²) DFT."""
        coeffs = _random_field_elements(8, bn254_sf_mont)
        log_n = 3
        fwd_tw, _, _ = self.ntt.get_stage_twiddles(log_n)

        expected = _naive_ntt(coeffs, self.ntt)
        actual = _forward_ntt(coeffs, log_n, *fwd_tw)

        _assert_eq(self, actual, expected)

    def test_inverse_ntt(self):
        """Test inverse NTT matches naive O(n²) inverse DFT."""
        evals = _random_field_elements(8, bn254_sf_mont, seed=99)
        log_n = 3
        _, inv_tw, inv_n = self.ntt.get_stage_twiddles(log_n)

        expected = _naive_intt(evals, self.ntt)
        actual = _inverse_ntt(evals, inv_n, log_n, *inv_tw)

        _assert_eq(self, actual, expected)

    def test_roundtrip(self):
        """Test forward then inverse NTT recovers original coefficients."""
        coeffs = _random_field_elements(4, bn254_sf_mont)
        log_n = 2
        fwd_tw, inv_tw, inv_n = self.ntt.get_stage_twiddles(log_n)

        evals = _forward_ntt(coeffs, log_n, *fwd_tw)
        recovered = _inverse_ntt(evals, inv_n, log_n, *inv_tw)

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


if __name__ == "__main__":
    absltest.main()
