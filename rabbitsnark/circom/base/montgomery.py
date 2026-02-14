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

"""Montgomery form conversion utilities."""

# BN254 base field modulus (Fq)
BN254_FQ_MODULUS = (
    21888242871839275222246405745257275088696311157297823662689037894645226208583
)

# BN254 scalar field modulus (Fr)
BN254_FR_MODULUS = (
    21888242871839275222246405745257275088548364400416034343698204186575808495617
)

# R = 2^256 for 256-bit fields
R = 2**256


def _mod_inverse(a: int, m: int) -> int:
    """Compute modular inverse using extended Euclidean algorithm."""
    if a < 0:
        a = a % m
    g, x, _ = _extended_gcd(a, m)
    if g != 1:
        raise ValueError("Modular inverse does not exist")
    return x % m


def _extended_gcd(a: int, b: int) -> tuple[int, int, int]:
    """Extended Euclidean algorithm."""
    if a == 0:
        return b, 0, 1
    gcd, x1, y1 = _extended_gcd(b % a, a)
    x = y1 - (b // a) * x1
    y = x1
    return gcd, x, y


# Precomputed R^(-1) mod p for common fields
_R_INV_FQ = _mod_inverse(R, BN254_FQ_MODULUS)
_R_INV_FR = _mod_inverse(R, BN254_FR_MODULUS)


def from_montgomery(value: int, modulus: int) -> int:
    """Convert from Montgomery form to standard form.

    Montgomery form: a_mont = a * R mod p
    Standard form: a = a_mont * R^(-1) mod p

    Args:
        value: The value in Montgomery form.
        modulus: The field modulus.

    Returns:
        The value in standard form.
    """
    if modulus == BN254_FQ_MODULUS:
        r_inv = _R_INV_FQ
    elif modulus == BN254_FR_MODULUS:
        r_inv = _R_INV_FR
    else:
        r_inv = _mod_inverse(R, modulus)

    return (value * r_inv) % modulus


def to_montgomery(value: int, modulus: int) -> int:
    """Convert from standard form to Montgomery form.

    Args:
        value: The value in standard form.
        modulus: The field modulus.

    Returns:
        The value in Montgomery form.
    """
    return (value * R) % modulus
