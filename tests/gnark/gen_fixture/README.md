# Gnark Test Fixture Generator

Generates a tiny gnark Groth16 fixture (x × x == y, 2 constraints, 4 wires)
for testing the Python gnark loader.

## Usage

```shell
cd tests/gnark/gen_fixture
go run . -output_dir=../data/tiny_multiply
```

The output files are committed to `tests/gnark/data/tiny_multiply/` so this
only needs to be re-run if the fixture format changes.
