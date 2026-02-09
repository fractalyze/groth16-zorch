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

- Parse [circom](https://docs.circom.io/) witness (`.wtns`) files
- Parse [circom](https://docs.circom.io/) proving key (`.zkey`) files
- Groth16 proof generation (WIP)

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

### Parsing witness files

```python
from rabbitsnark.circom.wtns import parse_wtns

wtns = parse_wtns("path/to/circuit.wtns")
print(f"Version: {wtns.version}")
print(f"Number of witnesses: {wtns.num_witness}")
print(f"Witnesses: {wtns.witnesses}")
```

### Parsing zkey files

```python
from rabbitsnark.circom.zkey import parse_zkey

zkey = parse_zkey("path/to/circuit.zkey")
print(f"Version: {zkey.version}")
print(f"Domain size: {zkey.domain_size}")
print(f"Verifying key: {zkey.verifying_key}")
```

## How to test

```shell
pytest
```

With verbose output:

```shell
pytest -v
```

## License

Apache License 2.0
