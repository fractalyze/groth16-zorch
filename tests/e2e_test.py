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

"""End-to-end test: compile -> prove -> verify round-trip.

Uses the multiplier_3 circuit (3-input multiplier):
    z = [1, 60, 3, 4, 5, 12]
    z[0] = 1 (constant), z[1] = 60 (output: 3 * 4 * 5)
    z[2..4] = inputs (3, 4, 5), z[5] = 12 (intermediate: 3 * 4)
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import jax.numpy as jnp
from absl.testing import absltest
from zk_dtypes import bn254_sf_mont

from rabbitsnark.circom.wtns import parse_wtns
from rabbitsnark.circom.zkey import parse_zkey
from rabbitsnark.groth16 import compile_circom
from rabbitsnark.groth16.verifier import VerificationKey, verify

if TYPE_CHECKING:
    from jax import Array

    from rabbitsnark.circom.wtns.wtns import WtnsV2
    from rabbitsnark.circom.zkey.zkey import ZKeyV1


def _compute_az_bz(
    zkey: ZKeyV1,
    wtns: WtnsV2,
) -> tuple[Array, Array]:
    """Compute A*z and B*z in Montgomery form from zkey coefficients.

    Simple dense implementation for testing (not suitable for large circuits).
    """
    n = zkey.domain_size
    m = zkey.header_groth.num_vars
    modulus = zkey.header_groth.r.to_int()

    a_dense = [[0] * m for _ in range(n)]
    b_dense = [[0] * m for _ in range(n)]
    for coeff in zkey.coefficients:
        row, col, val = coeff.constraint, coeff.signal, coeff.value
        if coeff.matrix == 0:
            a_dense[row][col] = (a_dense[row][col] + val) % modulus
        else:
            b_dense[row][col] = (b_dense[row][col] + val) % modulus

    z = [int(w) for w in wtns.witnesses]
    az_vals = [sum(a_dense[i][j] * z[j] for j in range(m)) % modulus for i in range(n)]
    bz_vals = [sum(b_dense[i][j] * z[j] for j in range(m)) % modulus for i in range(n)]

    return (
        jnp.array(az_vals, dtype=bn254_sf_mont),
        jnp.array(bz_vals, dtype=bn254_sf_mont),
    )


class TestE2EProveVerify(absltest.TestCase):
    """End-to-end: compile_circom -> prove_circom -> verify."""

    def setUp(self):
        test_data_dir = Path(__file__).parent / "data"
        self.zkey = parse_zkey(test_data_dir / "multiplier_3.zkey")
        self.wtns = parse_wtns(test_data_dir / "multiplier_3.wtns")
        self.compiled = compile_circom(self.zkey)
        self.az_mont, self.bz_mont = _compute_az_bz(self.zkey, self.wtns)
        self.vk = VerificationKey.from_zkey(self.zkey)

    def test_prove_verify_no_zk(self):
        """Deterministic proof (r=s=0) verifies correctly."""
        proof, public_signals = self.compiled.prove_circom(
            self.wtns,
            self.az_mont,
            self.bz_mont,
            no_zk=True,
        )
        self.assertTrue(verify(self.vk, proof, public_signals))

    def test_prove_verify_with_zk(self):
        """Randomized ZK proof verifies correctly."""
        proof, public_signals = self.compiled.prove_circom(
            self.wtns,
            self.az_mont,
            self.bz_mont,
            no_zk=False,
        )
        self.assertTrue(verify(self.vk, proof, public_signals))

    def test_invalid_signal_rejects(self):
        """Proof with wrong public signals should fail verification."""
        proof, _ = self.compiled.prove_circom(
            self.wtns,
            self.az_mont,
            self.bz_mont,
            no_zk=True,
        )
        # Wrong public signal (99 instead of 60)
        self.assertFalse(verify(self.vk, proof, ["99"]))


if __name__ == "__main__":
    absltest.main()
