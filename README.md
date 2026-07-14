# groth16-zorch

A Groth16 prover implementation in Python using [Zorch](https://github.com/fractalyze/jax).
This is the Python counterpart of [RabbitSNARK](https://github.com/fractalyze/rabbitsnark),
which implements the Groth16 proving scheme using HLO via [PrimeIR](https://github.com/fractalyze/prime-ir)
and [ZKX](https://github.com/fractalyze/zkx).

| Project                                                  | Language | Backend        |
| -------------------------------------------------------- | -------- | -------------- |
| [rapidsnark](https://github.com/iden3/rapidsnark)        | C++      | Native         |
| [RabbitSNARK](https://github.com/fractalyze/rabbitsnark) | C++      | HLO (ZKIR/ZKX) |
| **groth16-zorch**                                        | Python   | Zorch          |

## Features

- Parse [circom](https://docs.circom.io/) proving key (`.zkey`) files
- Circom witness from a pre-computed `.wtns` file (e.g. produced by snarkjs)
- Load [gnark](https://github.com/Consensys/gnark) binary exports for circuit proving
- Az/Bz evaluated in pure JAX (`jax.ops.segment_sum` over the BN254 field dtype),
  so they run on the GPU alongside the prover — no native library needed
- Groth16 proof generation with snarkjs-compatible JSON output

## How to build

1. Clone the repository

   ```shell
   git clone https://github.com/fractalyze/groth16-zorch.git
   ```

1. Navigate to the project directory

   ```shell
   cd groth16-zorch
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
# Witness is a pre-computed .wtns (e.g. `snarkjs wtns calculate`)
groth16-zorch circom prove <circuit.zkey> <witness.wtns> <proof.json> <public.json>

groth16-zorch circom verify <vkey.json> <public.json> <proof.json>
```

#### Gnark

```shell
groth16-zorch gnark prove <export_dir> <proof.json> <public.json> [--no-zk] [--deterministic]
groth16-zorch gnark verify <export_dir> <public.json> <proof.json>
```

### Python API

#### Circom (`.zkey` + `.wtns`)

```python
import numpy as np
from zk_dtypes import bn254_sf_mont

from groth16_zorch.circom.wtns import parse_wtns
from groth16_zorch.circom.zkey import parse_zkey
from groth16_zorch.circom.zkey_to_terms import zkey_to_terms
from groth16_zorch.groth16 import compile_circom, write_public_signals
from groth16_zorch.r1cs import compute_abc

zkey = parse_zkey("path/to/circuit.zkey")
compiled = compile_circom(zkey)  # one-time: parse zkey, build term matrices + arrays

wtns = parse_wtns("path/to/circuit.wtns")
witness_mont = wtns.data._witnesses.view(np.dtype(bn254_sf_mont))
_terms, coefficients = zkey_to_terms(zkey)
az_mont, bz_mont = compute_abc(
    witness_mont, compiled.terms, coefficients, compiled.domain_size
)
z_std = wtns.data._witnesses
public_signals = write_public_signals(wtns.witnesses, compiled.config.num_public)
proof, public_signals = compiled.prove(z_std, az_mont, bz_mont, public_signals)
```

#### Gnark binary export

```python
import numpy as np
from jax import lax
from zk_dtypes import bn254_sf, bn254_sf_mont

from groth16_zorch.gnark import load_gnark_export
from groth16_zorch.groth16 import compile_gnark

# The export carries the solved witness and Az/Bz (solution_a/b), both
# produced by gnark's Go solver — nothing is recomputed here.
data = load_gnark_export("path/to/export/")
compiled = compile_gnark(data)

z_std = np.asarray(
    lax.convert_element_type(data.witness_full.view(bn254_sf_mont), bn254_sf)
)
public_signals = [str(int(z_std[i])) for i in range(compiled.config.num_public)]
proof, public_signals = compiled.prove(
    z_std, data.az_mont, data.bz_mont, public_signals
)
```

The output `proof.json` and `public.json` follow the same JSON schema as
snarkjs, so you can verify with:

```shell
snarkjs groth16 verify verification_key.json public.json proof.json
```

## Compatibility

### Circom / snarkjs

groth16-zorch is **input/output compatible** with the circom/snarkjs
ecosystem:

|                                      | snarkjs        | rapidsnark     | **groth16-zorch** |
| ------------------------------------ | -------------- | -------------- | ----------------- |
| Input `.zkey`                        | yes            | yes            | yes               |
| Input `.wtns`                        | yes            | yes            | yes               |
| Output `proof.json`                  | snarkjs format | snarkjs format | snarkjs format    |
| Verify with `snarkjs groth16 verify` | yes            | yes            | yes               |

### Gnark

groth16-zorch loads binary exports produced by a gnark Go program (see
`tests/gnark/gen_fixture` for a minimal example). The export must include
`witness_full.bin` and `solution_a/b.bin` — the witness and Az/Bz that gnark's
`r1csTyped.Solve` computes natively:

|                                           | gnark (Go)         | **groth16-zorch**            |
| ----------------------------------------- | ------------------ | ---------------------------- |
| Binary export (`metadata.json` + `*.bin`) | produces           | consumes                     |
| Proving key points                        | setup              | loaded from export           |
| Witness + Az/Bz                           | Go solver          | loaded from export           |
| Proof generation                          | `groth16.Prove()`  | `compiled.prove()`           |
| Verification                              | `groth16.Verify()` | `verify(vk, proof, signals)` |

## Benchmark

Prove time for SP1's final Groth16 verifier circuit — BN254, **15,965,950
constraints**, domain 2²⁴ — on an **RTX 5090**. The reference is gnark's
[ICICLE](https://github.com/ingonyama-zk/icicle) GPU Groth16 prover on the same
circuit and GPU. groth16-zorch's proof `verify`s and is deterministic (fixed
output across runs).

Both provers consume the same gnark export, which already carries the solved
witness and Az/Bz (gnark's Go solver produces them at export time). So this is a
**prove-only** comparison — it excludes witness solving on both sides:

| prover                       | prove (median) | speedup |
| ---------------------------- | -------------- | ------- |
| **groth16-zorch** (JAX, GPU) | **1573 ms**    | 1.50×   |
| gnark ICICLE (GPU)           | 2355 ms        | 1.00×   |

gnark's end-to-end run additionally re-solves the witness on every proof
(~2.2 s), for ~4.5 s total; groth16-zorch loads that pre-solved witness/Az/Bz
straight from the export. One-time setup for groth16-zorch (not counted in the
prove time): ~4.4 s to load the 19 GB export and ~5.0 s to compile the 2²⁴
executable.

Reproduce with `//benchmark:sp1_groth16` (see `.github/workflows/benchmark.yml`):

```shell
JAX_PLATFORMS=cuda,cpu bazel run //benchmark:sp1_groth16 -- \
    --export_dir=<sp1-groth16-export> --deterministic --circuit=sp1
```

## How to test

```shell
bazel test //...
```

Test organization:

- `//tests/circom:e2e_test` — Circom prove/verify via `.wtns`
- `//tests/gnark:e2e_test` — Gnark export prove/verify
- `//tests/circom:zkey_test` — `.zkey` parser unit tests
- `//tests/circom:wtns_test` — `.wtns` parser unit tests

## License

Apache License 2.0
