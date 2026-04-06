# rabbitsnark-py Architecture

## Groth16 Proving Pipeline

```
Input: .zkey (proving key) + witness (.wtns or .so + inputs)
  ↓
Witness Loading → R1CS Solver (optional, via r1cs-solver .so)
  ↓
Groth16 Prove:
  - Phase 1: NTT/FFT for H polynomial (Cz = Az ⊙ Bz, IFFT x 3 → Coset NTT x 3)
  - Phase 2: 5x MSM via lax.msm (GPU memory managed by MsmChunkSplit)
  - Phase 3: EC assembly + ZK blinding (CPU-only, scalar multiply)
  ↓
Output: snarkjs-compatible JSON proof
```

## Related Projects
| Project | Language | Backend |
|---------|----------|---------|
| rapidsnark | C++ | Native |
| RabbitSNARK | C++ | HLO (ZKIR/ZKX) |
| **rabbitsnark-py** | Python | Zorch |

## Dependencies
- whir-zorch: proving backend (Zorch framework)
- r1cs-solver: native R1CS solver (shared library)
- zk_dtypes: field types (BN254)
