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

Supports both circom (.zkey/.wtns) and gnark (binary export) formats::

    # Circom
    rabbitsnark circom prove <circuit.zkey> <witness.wtns> <proof.json> <public.json>
    rabbitsnark circom verify <vkey.json> <public.json> <proof.json>

    # Gnark
    rabbitsnark gnark prove <export_dir> <proof.json> <public.json>
    rabbitsnark gnark verify <export_dir> <public.json> <proof.json>
"""

from __future__ import annotations

import argparse
import json
import sys
import time


def _cmd_circom_prove(args: argparse.Namespace) -> None:
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


def _cmd_circom_verify(args: argparse.Namespace) -> None:
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


def _cmd_gnark_prove(args: argparse.Namespace) -> None:
    from pathlib import Path

    from rabbitsnark.gnark import load_gnark_export, load_solutions_mont
    from rabbitsnark.groth16 import compile_gnark

    export_dir = Path(args.export_dir)

    print(f"Loading gnark export from {export_dir}")
    data = load_gnark_export(export_dir)
    print(
        f"Loaded: wires={data.num_wires:,}, "
        f"constraints={data.num_constraints:,}, "
        f"domain={data.domain_size:,}"
    )

    print("Compiling proving key...")
    compiled = compile_gnark(data)

    print("Preparing solution vectors (Az, Bz)...")
    az_mont, bz_mont = load_solutions_mont(export_dir, data.domain_size)

    print("Generating proof...")
    t0 = time.time()
    proof, public_signals = compiled.prove_gnark(
        data.witness_full,
        az_mont,
        bz_mont,
        no_zk=args.no_zk,
        deterministic=args.deterministic,
    )
    elapsed = time.time() - t0
    print(f"Proof generated in {elapsed:.2f}s")

    with open(args.proof, "w") as f:
        json.dump(proof.to_json(), f)
    print(f"Proof written to: {args.proof}")

    with open(args.public, "w") as f:
        json.dump(public_signals, f)
    print(f"Public signals written to: {args.public}")


def _cmd_gnark_verify(args: argparse.Namespace) -> None:
    from pathlib import Path

    from rabbitsnark.gnark import load_gnark_export
    from rabbitsnark.groth16.verifier import VerificationKey, verify

    export_dir = Path(args.export_dir)
    data = load_gnark_export(export_dir)
    vk = VerificationKey.from_gnark(data)

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


def _add_circom_subcommands(subparsers: argparse._SubParsersAction) -> None:
    circom = subparsers.add_parser("circom", help="Circom format (.zkey/.wtns)")
    circom_sub = circom.add_subparsers(dest="circom_command")

    prove = circom_sub.add_parser("prove", help="Generate a Groth16 proof")
    prove.add_argument("zkey", help="proving key (.zkey)")
    prove.add_argument("wtns", help="witness (.wtns)")
    prove.add_argument("proof", help="output proof path (.json)")
    prove.add_argument("public", help="output public signals path (.json)")

    verify = circom_sub.add_parser("verify", help="Verify a Groth16 proof")
    verify.add_argument("vkey", help="verification key (.json)")
    verify.add_argument("public", help="public signals (.json)")
    verify.add_argument("proof", help="proof (.json)")


def _add_gnark_subcommands(subparsers: argparse._SubParsersAction) -> None:
    gnark = subparsers.add_parser("gnark", help="Gnark format (binary export)")
    gnark_sub = gnark.add_subparsers(dest="gnark_command")

    prove = gnark_sub.add_parser("prove", help="Generate a Groth16 proof")
    prove.add_argument("export_dir", help="gnark binary export directory")
    prove.add_argument("proof", help="output proof path (.json)")
    prove.add_argument("public", help="output public signals path (.json)")
    prove.add_argument("--no-zk", action="store_true", help="no ZK blinding (r=s=0)")
    prove.add_argument(
        "--deterministic",
        action="store_true",
        help="fixed non-zero r, s for reproducible proofs with full blinding",
    )

    verify = gnark_sub.add_parser("verify", help="Verify a Groth16 proof")
    verify.add_argument("export_dir", help="gnark binary export directory (for VK)")
    verify.add_argument("public", help="public signals (.json)")
    verify.add_argument("proof", help="proof (.json)")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="rabbitsnark",
        description="Groth16 prover and verifier for circom and gnark formats",
    )
    subparsers = parser.add_subparsers(dest="command")
    _add_circom_subcommands(subparsers)
    _add_gnark_subcommands(subparsers)

    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    if args.command == "circom":
        if args.circom_command == "prove":
            _cmd_circom_prove(args)
        elif args.circom_command == "verify":
            _cmd_circom_verify(args)
        else:
            parser.parse_args(["circom", "--help"])
    elif args.command == "gnark":
        if args.gnark_command == "prove":
            _cmd_gnark_prove(args)
        elif args.gnark_command == "verify":
            _cmd_gnark_verify(args)
        else:
            parser.parse_args(["gnark", "--help"])


if __name__ == "__main__":
    main()
