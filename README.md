# RabbitSNARK-py

A Groth16 prover implementation in Python using [Zorch](https://github.com/fractalyze/jax).
This is the Python counterpart of [RabbitSNARK](https://github.com/fractalyze/rabbitsnark),
which implements the Groth16 proving scheme using HLO via [PrimeIR](https://github.com/fractalyze/prime-ir)
and [ZKX](https://github.com/fractalyze/zkx).

| Project                                                  | Language | Backend        |
| -------------------------------------------------------- | -------- | -------------- |
| [rapidsnark](https://github.com/iden3/rapidsnark)        | C++      | Native         |
| [RabbitSNARK](https://github.com/fractalyze/rabbitsnark) | C++      | HLO (ZKIR/ZKX) |
| **RabbitSNARK-py**                                       | Python   | Zorch          |

## Features

- Parse [circom](https://docs.circom.io/) proving key (`.zkey`) files
- Two witness paths for circom: pre-computed `.wtns` file or compiled circuit `.so` + inputs
- Load [gnark](https://github.com/Consensys/gnark) binary exports for circuit proving
- Native R1CS solver via [r1cs-solver](https://github.com/fractalyze/r1cs-solver) shared library
- Groth16 proof generation with snarkjs-compatible JSON output

## How to build

1. Clone the repository

   ```shell
   git clone https://github.com/fractalyze/rabbitsnark-py.git
   ```

1. Navigate to the project directory

   ```shell
   cd rabbitsnark-py
   ```

1. Install the package

   ```shell
   pip install -e .
   ```

   For development with test dependencies:

   ```shell
   pip install -e ".[dev]"
   ```

## How to run

### CLI

#### Circom

```shell
# Path 1: pre-computed witness (.wtns)
rabbitsnark circom prove <circuit.zkey> <proof.json> <public.json> --wtns <witness.wtns>

# Path 2: compiled circuit (.so) + inputs (production)
rabbitsnark circom prove <circuit.zkey> <proof.json> <public.json> \
    --circuit <circuit.so> --input <input.json> --w2s <w2s.json>

rabbitsnark circom verify <vkey.json> <public.json> <proof.json>
```

#### Gnark

```shell
rabbitsnark gnark prove <export_dir> <proof.json> <public.json> [--no-zk] [--deterministic]
rabbitsnark gnark verify <export_dir> <public.json> <proof.json>
```

### Python API

#### Circom (`.zkey` + `.wtns`)

```python
import numpy as np
from zk_dtypes import bn254_sf_mont

from rabbitsnark.circom.wtns import parse_wtns
from rabbitsnark.circom.zkey import parse_zkey
from rabbitsnark.groth16 import compile_circom, write_public_signals
from rabbitsnark.r1cs_solver import compute_abc

zkey = parse_zkey("path/to/circuit.zkey")
compiled = compile_circom(zkey)  # one-time: parse zkey, build CSR + arrays

wtns = parse_wtns("path/to/circuit.wtns")
witness_mont = wtns.data._witnesses.view(np.dtype(bn254_sf_mont))
az_mont, bz_mont = compute_abc(
    witness_mont, compiled.csr, compiled.domain_size, compiled.domain_size
)
z_std = wtns.data._witnesses
public_signals = write_public_signals(wtns.witnesses, compiled.config.num_public)
proof, public_signals = compiled.prove(z_std, az_mont, bz_mont, public_signals)
```

#### Circom (`.so` + `input.json`)

```python
import numpy as np
from zk_dtypes import bn254_sf

from rabbitsnark.circom.witness_calculator import CircomWitnessCalculator, load_w2s
from rabbitsnark.circom.zkey import parse_zkey
from rabbitsnark.groth16 import compile_circom, write_public_signals
from rabbitsnark.r1cs_solver import compute_abc

zkey = parse_zkey("path/to/circuit.zkey")
compiled = compile_circom(zkey)

calc = CircomWitnessCalculator("path/to/circuit.so")
w2s = load_w2s("path/to/w2s.json")
witness = calc.compute_witness({"a": "3", "b": "4", "c": "5"}, w2s)

az_mont, bz_mont = compute_abc(
    witness, compiled.csr, compiled.domain_size, compiled.domain_size
)
z_std = witness.view(np.dtype(bn254_sf))
public_signals = write_public_signals(
    z_std[: compiled.config.num_public + 1], compiled.config.num_public
)
proof, public_signals = compiled.prove(z_std, az_mont, bz_mont, public_signals)
```

#### Gnark binary export

```python
from rabbitsnark.gnark import load_gnark_export, load_solver_data
from rabbitsnark.groth16 import compile_gnark
from rabbitsnark.r1cs_solver import solve_and_compute

data = load_gnark_export("path/to/export/")
compiled = compile_gnark(data)

solver = load_solver_data("path/to/export/")
z_std, az_mont, bz_mont = solve_and_compute(data.witness_full, solver)
public_signals = [str(int(z_std[i])) for i in range(compiled.config.num_public)]
proof, public_signals = compiled.prove(z_std, az_mont, bz_mont, public_signals)
```

The output `proof.json` and `public.json` follow the same JSON schema as
snarkjs, so you can verify with:

```shell
snarkjs groth16 verify verification_key.json public.json proof.json
```

## Architecture

```
rabbitsnark/
  r1cs_solver/     — Native r1cs-solver wrapper (shared by circom + gnark)
                     CSRMatrices, compute_abc, solve_witness, solve_and_compute
  circom/          — Circom format parsers + witness calculator
    zkey/          — .zkey parser (proving key)
    wtns/          — .wtns parser (witness)
    zkey_to_csr.py — zkey coefficients → CSR (aR² double Montgomery)
    witness_calculator.py — circuit .so ctypes wrapper
  gnark/           — Gnark binary export loaders
    loader.py      — load_gnark_export (points, witness)
    compute_abc.py — load_solver_data (CSR, levels, hints)
  groth16/         — Groth16 prover + verifier
    prover.py      — compile_circom, compile_gnark, CompiledProver.prove
    verifier.py    — verify
```

## Compatibility

### Circom / snarkjs

RabbitSNARK-py is **input/output compatible** with the circom/snarkjs
ecosystem:

|                                      | snarkjs        | rapidsnark     | **RabbitSNARK-py** |
| ------------------------------------ | -------------- | -------------- | ------------------ |
| Input `.zkey`                        | yes            | yes            | yes                |
| Input `.wtns`                        | yes            | yes            | yes                |
| Input circuit `.so` + inputs         | —              | —              | yes                |
| Output `proof.json`                  | snarkjs format | snarkjs format | snarkjs format     |
| Verify with `snarkjs groth16 verify` | yes            | yes            | yes                |

### Gnark

RabbitSNARK-py loads binary exports produced by gnark's Go exporter
([sp1-groth16-bench](https://github.com/fractalyze/sp1-groth16-bench)
`cmd/export`):

|                                           | gnark (Go)         | **RabbitSNARK-py**           |
| ----------------------------------------- | ------------------ | ---------------------------- |
| Binary export (`metadata.json` + `*.bin`) | produces           | consumes                     |
| Proving key points                        | setup              | loaded from export           |
| Witness solving                           | Go solver          | native r1cs-solver           |
| Proof generation                          | `groth16.Prove()`  | `compiled.prove()`           |
| Verification                              | `groth16.Verify()` | `verify(vk, proof, signals)` |

## How to test

```shell
bazel test //...
```

Test organization:

- `//tests/circom:e2e_test` — Circom prove/verify via `.wtns`
- `//tests/circom:e2e_circuit_test` — Circom prove/verify via circuit `.so` + inputs
- `//tests/gnark:e2e_test` — Gnark export solve/prove/verify
- `//tests/circom:zkey_test` — `.zkey` parser unit tests
- `//tests/circom:wtns_test` — `.wtns` parser unit tests

## License

Apache License 2.0
