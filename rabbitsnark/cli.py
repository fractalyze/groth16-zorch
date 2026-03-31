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
    import numpy as np
    from zk_dtypes import bn254_sf, bn254_sf_mont

    from rabbitsnark.circom.zkey import parse_zkey
    from rabbitsnark.groth16 import compile_circom, write_public_signals
    from rabbitsnark.r1cs_solver import compute_abc

    print(f"Loading zkey: {args.zkey}")
    zkey = parse_zkey(args.zkey)

    print("Compiling proving key...")
    t0 = time.time()
    compiled = compile_circom(zkey)
    elapsed = time.time() - t0
    print(f"Compiled in {elapsed:.2f}s")

    if args.wtns:
        # Path 1: pre-computed witness file (.wtns)
        from rabbitsnark.circom.wtns import parse_wtns

        print(f"Loading wtns: {args.wtns}")
        wtns = parse_wtns(args.wtns)
        # Witness is standard form — bitcast for compute_abc.
        witness_mont = wtns.data._witnesses.view(np.dtype(bn254_sf_mont))
        z_std = wtns.data._witnesses
        public_signals = write_public_signals(
            wtns.witnesses, compiled.config.num_public
        )
    else:
        # Path 2: compute witness from circuit + inputs (production)
        from rabbitsnark.circom.witness_calculator import (
            CircomWitnessCalculator,
            load_w2s,
        )

        print(f"Loading circuit: {args.circuit}")
        calc = CircomWitnessCalculator(args.circuit)

        print(f"Loading inputs: {args.input}")
        with open(args.input) as f:
            inputs = json.load(f)

        print(f"Loading w2s: {args.w2s}")
        w2s = load_w2s(args.w2s)

        print("Computing witness...")
        t0 = time.time()
        witness = calc.compute_witness(inputs, w2s)
        elapsed = time.time() - t0
        print(f"Witness computed in {elapsed:.2f}s")

        # Witness bytes are standard form (from_mont applied in calculator),
        # tagged as bn254_sf_mont — use as-is for compute_abc, view as
        # bn254_sf for z_std.
        witness_mont = witness
        z_std = witness.view(np.dtype(bn254_sf))
        public_signals = write_public_signals(
            z_std[: compiled.config.num_public + 1],
            compiled.config.num_public,
        )

    print("Computing Az/Bz...")
    t0 = time.time()
    from rabbitsnark.circom.zkey_to_terms import zkey_to_terms

    _terms, coefficients = zkey_to_terms(zkey)
    az_mont, bz_mont = compute_abc(
        witness_mont,
        compiled.terms,
        coefficients,
        compiled.domain_size,
        compiled.domain_size,
    )
    elapsed = time.time() - t0
    print(f"Az/Bz computed in {elapsed:.2f}s")

    print("Generating proof...")
    t0 = time.time()
    proof, public_signals = compiled.prove(z_std, az_mont, bz_mont, public_signals)
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

    from rabbitsnark.gnark import load_gnark_export, load_solver_data
    from rabbitsnark.groth16 import compile_gnark
    from rabbitsnark.r1cs_solver import solve_and_compute

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

    print("Solving witness + computing Az/Bz via native solver...")
    solver = load_solver_data(export_dir)
    z_std, az_mont, bz_mont = solve_and_compute(data.witness_full, solver)

    public_signals = [str(int(z_std[i])) for i in range(compiled.config.num_public)]

    print("Generating proof...")
    t0 = time.time()
    proof, public_signals = compiled.prove(
        z_std,
        az_mont,
        bz_mont,
        public_signals,
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
    circom = subparsers.add_parser("circom", help="Circom format (.zkey)")
    circom_sub = circom.add_subparsers(dest="circom_command")

    prove = circom_sub.add_parser("prove", help="Generate a Groth16 proof")
    prove.add_argument("zkey", help="proving key (.zkey)")
    prove.add_argument("proof", help="output proof path (.json)")
    prove.add_argument("public", help="output public signals path (.json)")
    # Witness source: --wtns or --circuit + --input + --w2s
    witness_group = prove.add_mutually_exclusive_group(required=True)
    witness_group.add_argument("--wtns", help="pre-computed witness (.wtns)")
    witness_group.add_argument("--circuit", help="compiled circuit (.so)")
    prove.add_argument(
        "--input", help="circuit inputs (.json), required with --circuit"
    )
    prove.add_argument(
        "--w2s", help="witness-to-signal mapping (.json), required with --circuit"
    )

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
