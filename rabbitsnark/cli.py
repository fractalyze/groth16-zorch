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

"""Command-line interface for RabbitSNARK Groth16 prover.

Usage mirrors rapidsnark::

    rabbitsnark <circuit.zkey> <witness.wtns> <proof.json> <public.json>
"""

from __future__ import annotations

import argparse
import json
import time


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="rabbitsnark",
        description="Groth16 prover (circom/snarkjs compatible)",
    )
    parser.add_argument("zkey", help="proving key (.zkey)")
    parser.add_argument("wtns", help="witness (.wtns)")
    parser.add_argument("proof", help="output proof path (.json)")
    parser.add_argument("public", help="output public signals path (.json)")
    args = parser.parse_args(argv)

    from rabbitsnark.circom.wtns import parse_wtns
    from rabbitsnark.circom.zkey import parse_zkey
    from rabbitsnark.groth16 import compile

    print(f"Loading zkey: {args.zkey}")
    zkey = parse_zkey(args.zkey)
    print(f"Loading wtns: {args.wtns}")
    wtns = parse_wtns(args.wtns)

    print("Compiling proving key...")
    t0 = time.time()
    compiled = compile(zkey)
    elapsed = time.time() - t0
    print(f"Compiled in {elapsed:.2f}s")

    print("Generating proof...")
    t0 = time.time()
    proof, public_signals = compiled.prove(wtns)
    elapsed = time.time() - t0
    print(f"Proof generated in {elapsed:.2f}s")

    with open(args.proof, "w") as f:
        json.dump(proof.to_json(), f)
    print(f"Proof written to: {args.proof}")

    with open(args.public, "w") as f:
        json.dump(public_signals, f)
    print(f"Public signals written to: {args.public}")


if __name__ == "__main__":
    main()
