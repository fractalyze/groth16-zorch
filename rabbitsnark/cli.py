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

"""Command-line interface for RabbitSNARK Groth16 prover and verifier.

Usage::

    # Prove (backward-compatible positional args)
    rabbitsnark <circuit.zkey> <witness.wtns> <proof.json> <public.json>

    # Prove (explicit subcommand)
    rabbitsnark prove <circuit.zkey> <witness.wtns> <proof.json> <public.json>

    # Verify (mirrors snarkjs groth16 verify)
    rabbitsnark verify <vkey.json> <public.json> <proof.json>
"""

from __future__ import annotations

import argparse
import json
import sys
import time


def _cmd_prove(args: argparse.Namespace) -> None:
    from rabbitsnark.circom.wtns import parse_wtns
    from rabbitsnark.circom.zkey import parse_zkey
    from rabbitsnark.groth16 import compile_circom

    print(f"Loading zkey: {args.zkey}")
    zkey = parse_zkey(args.zkey)
    print(f"Loading wtns: {args.wtns}")
    wtns = parse_wtns(args.wtns)

    print("Compiling proving key...")
    t0 = time.time()
    compiled = compile_circom(zkey)
    elapsed = time.time() - t0
    print(f"Compiled in {elapsed:.2f}s")

    print("Generating proof...")
    t0 = time.time()
    proof, public_signals = compiled.prove_circom(wtns)
    elapsed = time.time() - t0
    print(f"Proof generated in {elapsed:.2f}s")

    with open(args.proof, "w") as f:
        json.dump(proof.to_json(), f)
    print(f"Proof written to: {args.proof}")

    with open(args.public, "w") as f:
        json.dump(public_signals, f)
    print(f"Public signals written to: {args.public}")


def _cmd_verify(args: argparse.Namespace) -> None:
    from rabbitsnark.groth16 import VerificationKey, verify

    print(f"Loading verification key: {args.vkey}")
    vk = VerificationKey.from_file(args.vkey)

    print(f"Loading public signals: {args.public}")
    with open(args.public) as f:
        public_signals = json.load(f)

    print(f"Loading proof: {args.proof}")
    with open(args.proof) as f:
        proof = json.load(f)

    print("Verifying proof...")
    t0 = time.time()
    valid = verify(vk, proof, public_signals)
    elapsed = time.time() - t0
    print(f"Verification completed in {elapsed:.2f}s")

    if valid:
        print("VALID")
    else:
        print("INVALID")
        sys.exit(1)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="rabbitsnark",
        description="Groth16 prover and verifier (circom/snarkjs compatible)",
    )
    subparsers = parser.add_subparsers(dest="command")

    # prove subcommand
    prove_parser = subparsers.add_parser("prove", help="Generate a Groth16 proof")
    prove_parser.add_argument("zkey", help="proving key (.zkey)")
    prove_parser.add_argument("wtns", help="witness (.wtns)")
    prove_parser.add_argument("proof", help="output proof path (.json)")
    prove_parser.add_argument("public", help="output public signals path (.json)")

    # verify subcommand
    verify_parser = subparsers.add_parser("verify", help="Verify a Groth16 proof")
    verify_parser.add_argument("vkey", help="verification key (.json)")
    verify_parser.add_argument("public", help="public signals (.json)")
    verify_parser.add_argument("proof", help="proof (.json)")

    args = parser.parse_args(argv)

    # Backward compat: 4 positional args without subcommand -> prove
    if args.command is None:
        if argv is not None:
            raw_args = argv
        else:
            raw_args = sys.argv[1:]
        if len(raw_args) == 4:
            args = prove_parser.parse_args(raw_args)
            args.command = "prove"
        else:
            parser.print_help()
            sys.exit(1)

    if args.command == "prove":
        _cmd_prove(args)
    elif args.command == "verify":
        _cmd_verify(args)


if __name__ == "__main__":
    main()
