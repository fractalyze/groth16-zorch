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

"""Tests for Groth16 proof generation.

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
from rabbitsnark.groth16 import (
    CompiledProver,
    Groth16Proof,
    compile_circom,
    write_public_signals,
)

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

    # Build dense A, B as Python ints
    a_dense = [[0] * m for _ in range(n)]
    b_dense = [[0] * m for _ in range(n)]
    for coeff in zkey.coefficients:
        row, col, val = coeff.constraint, coeff.signal, coeff.value
        if coeff.matrix == 0:
            a_dense[row][col] = (a_dense[row][col] + val) % modulus
        else:
            b_dense[row][col] = (b_dense[row][col] + val) % modulus

    # Dense matrix-vector multiply mod p
    z = [int(w) for w in wtns.witnesses]
    az_vals = [sum(a_dense[i][j] * z[j] for j in range(m)) % modulus for i in range(n)]
    bz_vals = [sum(b_dense[i][j] * z[j] for j in range(m)) % modulus for i in range(n)]

    return (
        jnp.array(az_vals, dtype=bn254_sf_mont),
        jnp.array(bz_vals, dtype=bn254_sf_mont),
    )


class TestGroth16Prove(absltest.TestCase):
    """Tests for Groth16 proof generation."""

    def setUp(self):
        test_data_dir = Path(__file__).parent / "data"
        self.zkey = parse_zkey(test_data_dir / "multiplier_3.zkey")
        self.wtns = parse_wtns(test_data_dir / "multiplier_3.wtns")
        self.compiled = compile_circom(self.zkey)
        self.az_mont, self.bz_mont = _compute_az_bz(self.zkey, self.wtns)

    def test_compile_returns_compiled_prover(self):
        """compile_circom() returns a CompiledProver instance."""
        self.assertIsInstance(self.compiled, CompiledProver)

    def test_prove_no_zk(self):
        """Deterministic proof (r=s=0) produces non-zero points."""
        proof, public_signals = self.compiled.prove_circom(
            self.wtns,
            self.az_mont,
            self.bz_mont,
            no_zk=True,
        )

        self.assertIsInstance(proof, Groth16Proof)

        # Verify JSON serialization produces non-trivial coordinates
        proof_json = proof.to_json()
        self.assertNotEqual(proof_json["pi_a"][0], "0")
        self.assertNotEqual(proof_json["pi_a"][1], "0")
        self.assertNotEqual(proof_json["pi_c"][0], "0")
        self.assertNotEqual(proof_json["pi_c"][1], "0")
        self.assertEqual(proof_json["protocol"], "groth16")
        self.assertEqual(proof_json["curve"], "bn128")

    def test_public_signals(self):
        """Public signals are correctly extracted from witness."""
        _, public_signals = self.compiled.prove_circom(
            self.wtns,
            self.az_mont,
            self.bz_mont,
            no_zk=True,
        )

        # multiplier_3 has 1 public input: the output 60
        self.assertEqual(public_signals, ["60"])

    def test_write_public_signals(self):
        """write_public_signals extracts z[1:l+1]."""
        witnesses = [1, 60, 3, 4, 5, 12]
        signals = write_public_signals(witnesses, num_public=1)
        self.assertEqual(signals, ["60"])

    def test_prove_with_zk(self):
        """ZK proof (random r, s) differs from no-ZK proof."""
        proof_no_zk, _ = self.compiled.prove_circom(
            self.wtns,
            self.az_mont,
            self.bz_mont,
            no_zk=True,
        )
        proof_zk, _ = self.compiled.prove_circom(
            self.wtns,
            self.az_mont,
            self.bz_mont,
            no_zk=False,
        )

        no_zk_json = proof_no_zk.to_json()
        zk_json = proof_zk.to_json()

        # Random blinding should change at least pi_a
        self.assertNotEqual(no_zk_json["pi_a"], zk_json["pi_a"])

    def test_proof_json_structure(self):
        """Proof JSON has snarkjs-compatible structure."""
        proof, _ = self.compiled.prove_circom(
            self.wtns,
            self.az_mont,
            self.bz_mont,
            no_zk=True,
        )
        proof_json = proof.to_json()

        # pi_a: [x, y, "1"]
        self.assertEqual(len(proof_json["pi_a"]), 3)
        self.assertEqual(proof_json["pi_a"][2], "1")

        # pi_b: [[x0, x1], [y0, y1], ["1", "0"]]
        self.assertEqual(len(proof_json["pi_b"]), 3)
        self.assertEqual(len(proof_json["pi_b"][0]), 2)
        self.assertEqual(len(proof_json["pi_b"][1]), 2)
        self.assertEqual(proof_json["pi_b"][2], ["1", "0"])

        # pi_c: [x, y, "1"]
        self.assertEqual(len(proof_json["pi_c"]), 3)
        self.assertEqual(proof_json["pi_c"][2], "1")

    def test_compile_prove_reuse(self):
        """Compiled prover can generate multiple proofs."""
        proof1, _ = self.compiled.prove_circom(
            self.wtns,
            self.az_mont,
            self.bz_mont,
            no_zk=True,
        )
        proof2, _ = self.compiled.prove_circom(
            self.wtns,
            self.az_mont,
            self.bz_mont,
            no_zk=True,
        )

        # Deterministic (no_zk) proofs from the same witness must match
        self.assertEqual(proof1.to_json(), proof2.to_json())


if __name__ == "__main__":
    absltest.main()
