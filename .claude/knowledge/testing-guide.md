# rabbitsnark-py Testing Guide

## Running Tests
```
bazel test //...
# or
pytest tests/
```

## Test Data
- Test circuits: circom .zkey files in testdata/
- Reference proofs: verified against snarkjs output

## What to Test
- Proof generation matches snarkjs output exactly
- Round-trip: prove → verify must succeed
- Edge cases: minimal circuits, large circuits
