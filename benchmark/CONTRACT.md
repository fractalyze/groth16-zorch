# Cross-impl benchmark contract

Authority for the multi-impl, multi-device benchmark suite tracked in
[`#77`](https://github.com/fractalyze/rabbitsnark-py/issues/77). Reviewers of
[`fractalyze/sp1#25`](https://github.com/fractalyze/sp1/issues/25) (sp1-ref
control) and [`fractalyze/whir-zorch#156`](https://github.com/fractalyze/whir-zorch/issues/156)
(logup-GKR + SMCS commit) implement against this doc.

The contract is the rules harnesses follow so a single orchestrator can
compare them on the same fixture with `output_hash` equality as the
correctness gate.

## Output schema

The shared schema is [`zkbench.schema.BenchmarkReport`](https://github.com/fractalyze/zkbench-py/blob/main/zkbench/schema.py)
(Python). The Go counterpart [`fractalyze/zkbench-go`](https://github.com/fractalyze/zkbench-go)
emits the same shape. Rust harnesses MUST emit JSON deserializable into
the Python type.

```json
{
  "metadata": {
    "implementation": "rabbitsnark",
    "version": "0.1.0",
    "commit_sha": "<short sha>",
    "timestamp": "<ISO-8601 UTC>",
    "platform": {
      "os": "linux", "arch": "x86_64", "cpu_count": 32,
      "cpu_vendor": "AMD", "gpu_vendor": "nvidia"
    }
  },
  "benchmarks": {
    "<primitive-name>": {
      "latency":    {"value": 1.23, "unit": "s",
                     "lower_value": 1.20, "upper_value": 1.27},
      "memory":     {"value": 8.5,  "unit": "GiB"},
      "throughput": {"value": 1e9,  "unit": "ops/s"},
      "iterations": 10,
      "test_vectors": {
        "input_hash":  "<hex sha256>",
        "output_hash": "<hex sha256>",
        "verified":    true
      },
      "metadata": {"field": "bn254", "degree": "20"}
    }
  }
}
```

Field semantics:

- `latency.lower_value` / `upper_value` are bootstrap 95% bounds when
  `iterations >= 5`; omit otherwise.
- `verified` is the harness's own validity check (proof verifies, hash
  matches expected, …) — NOT a cross-impl equality assertion. That check
  belongs to the orchestrator (`run_local.sh`).
- `metadata` per-benchmark is free-form for impl-specific notes; reserved
  keys are `field`, `degree`, `circuit`.

## Implementation-name registry

`metadata.implementation` MUST be one of:

| String | Repo / source | Harness language |
|---|---|---|
| `rabbitsnark` | `fractalyze/rabbitsnark-py` (this repo) | Python (JAX, jax_fork) |
| `gnark` | `fractalyze/sp1` (`ref` branch), `sp1-groth16-bench/cmd/*` | Go (`fractalyze/zkbench-go`) |
| `sp1-ref` | `fractalyze/sp1` (`ref` branch), `sp1-gpu/crates/*/bin/*_bench.rs` | Rust |
| `whir-zorch` | `fractalyze/whir-zorch` | Rust (TBD per `#156`) |

New impls extend the table via PR to this file.

## Primitive names

`benchmarks.<key>` uses these keys. Cross-impl `output_hash` equality is
asserted by the orchestrator for keys marked **gate**; the rest are
single-impl-only.

| Key | Description | Cross-impl gate | Reference impls |
|---|---|---|---|
| `groth16_sp1_verifier` | Full prove pass (compute_abc + NTT + MSM + EC) | gate (with `--deterministic`) | `rabbitsnark`, `gnark` |
| `groth16_sp1_verifier_solver` | R1CS solver only | gate | `rabbitsnark`, `gnark` |
| `fft` | Forward NTT, gnark domain generator | gate | `rabbitsnark`, `gnark` |
| `ifft` | Inverse NTT | gate | `rabbitsnark`, `gnark` |
| `msm_g1` | BN254 G1 MSM | gate | `rabbitsnark`, `gnark` |
| `logup_gkr` | Logup-GKR sumcheck pass | gate | `sp1-ref`, `whir-zorch` |
| `zerocheck` | Zerocheck pass (SMCS commit analog on the SP1 side) | gate | `sp1-ref`, `whir-zorch` |
| `merkle_commit` | Merkle commitment over a fixed-leaf-count tensor | single-impl | `sp1-ref` |
| `poseidon2` | Poseidon2 permutation (fixed input) | single-impl | `sp1-ref` |

The `logup_gkr` ↔ `zerocheck` ↔ "SMCS commit" mapping between SP1-ref
and whir-zorch is finalized in `#156` and `#25` during their spec phase.
Update this table when those land.

## Required CLI args

Every harness binary MUST accept:

| Flag | Type | Default | Meaning |
|---|---|---|---|
| `--output <path>` | str | (stdout) | Write the `BenchmarkReport` JSON to this path. |
| `--iterations <N>` | int | 5 | Measured iterations per op. |
| `--warmup <N>` | int | 1 | Warmup iterations (excluded from stats). |
| `--deterministic` | flag | off | For randomized ops (groth16 prove): fix internal randomness so `output_hash` is reproducible. |

Primitive-specific args (extend per impl):

| Primitive | Flag | Example |
|---|---|---|
| `groth16_*` | `--export_dir <path>` | `--export_dir=/data/sp1-groth16` |
| `fft` / `ifft` / `msm_g1` | `--sizes <csv>` | `--sizes=16,18,20,22,24` (log2) |
| `fft` / `ifft` / `msm_g1` | `--seed <u32>` | `--seed=42` (default; deterministic inputs) |

Subcommand dispatching is per-impl. Rabbit uses one Bazel target per
harness (`//benchmark:primitives_benchmark`, `//benchmark:sp1_groth16`).
SP1 ref's `sp1-groth16-bench` dispatches via `cmd/<subcommand>/`
(`bench`, `primitives`, `solve`, …). Image ENTRYPOINTs SHOULD expose a
subcommand wrapper so `docker run <image> <primitive> --flags...` works
uniformly across impls.

## Fixture mounts

Bench inputs live outside the harness binaries to keep images small.
Convention:

- `bench_config.<target>.yaml` declares `fixture.local_path` (dev box) and
  `fixture.container_path` (in-image). The orchestrator mounts the local
  path at the container path.
- Default container paths: `/data/<fixture-name>` (e.g. `/data/sp1-groth16`).
- Fixtures themselves are repo-external (tracked via test-data scripts);
  this contract assumes the fixture is reachable at the declared paths.

For deterministic-input primitives (FFT/IFFT/MSM), the fixture is the
`(seed, log_size)` pair — no on-disk fixture; harnesses generate inputs
in-process.

## Canonical `output_hash` rules

`output_hash` is computed in impl-independent canonical form so equality
across implementations is meaningful.

**Authoritative reference**: SP1 ref's `sp1-groth16-bench/sp1/primitives_common.go`
(`HashElements`, `HashG1Affine`). The canonical form is gnark-crypto's
`Marshal()`:

| Element type | Canonical byte representation |
|---|---|
| Scalar (BN254 `fr.Element`) | 32 bytes, big-endian, **standard form** (NOT Montgomery) |
| G1 affine point | 64 bytes, **uncompressed**: x big-endian (32) ‖ y big-endian (32), both in standard form. `gnark-crypto.G1Affine.Marshal()` returns this shape for BN254. |
| G1 projective | normalize to affine first, then encode as above |
| G2 affine point | 128 bytes, uncompressed: x ‖ y where each is a 2-element Fp2 in big-endian standard form |
| Vector of any of the above | concatenation of element canonical bytes, in array order |

Hash: SHA-256 over the canonical byte stream. Hex-encoded, lowercase, no
prefix.

Rabbit's `benchmark/primitives_benchmark.py` aligns to this form in
Task 1.1: `_hash_scalars` for fr arrays, `_hash_g1_affine` for the MSM
result point. The Montgomery → standard conversion for G1 happens in
Python (the `lax.convert_element_type(g1_mont → g1_std)` lowering in
zkx is currently broken — see open issue).

**Cross-impl gate** (orchestrator-enforced):

```
output_hash(impl_a, primitive, fixture) == output_hash(impl_b, primitive, fixture)
```

for every `(impl_a, impl_b)` pair listed under "Cross-impl gate" in the
Primitive names table, on the same `(seed, log_size)` or same on-disk
fixture sha256.

**Carve-out — groth16**: prove is randomized (`r`, `s` ∈ Fr). Equality
only under `--deterministic` AND identical `r`, `s` between impls.
Rabbit uses a fixed `r`, `s` derived from a seed; SP1 ref does the same.
The harness's `metadata.circuit` field SHOULD carry the seed when used.

**Carve-out — FFT/IFFT** (pending [`fractalyze/sp1#26`](https://github.com/fractalyze/sp1/pull/26) merge):
rabbit's `lax.fft(s, "FFT", n, generator=5)` produces the mathematically
correct natural-in/natural-out NTT (verified via delta-input test —
`lax.fft(delta[1])` returns `[1, ω, ω², …]` with ω = `5^((p-1)/n)`).
SP1 ref's `sp1-groth16-bench` standalone bench was misusing gnark's
API (`fft.DIT` expects bit-reversed input; the bench fed natural-order
scalars), producing `fft`/`ifft` `output_hash` values that did not
represent the NTT of the input. The fix in sp1#26 routes the bench
through `fft.DIF` + `fft.BitReverse` and now matches rabbit's hashes
byte-for-byte at log_size ∈ {4, 16, 18, 20}. Once that PR merges and
the `sp1-ref-cuda` image (sp1#25) is rebuilt against the fixed bench,
the cross-impl gate for `fft`/`ifft` works without further changes
here.

## Versioning

The contract is versioned by this file's git history. Breaking changes
(renaming a primitive key, changing a CLI flag default, altering the
canonical form) require:

1. A PR to this file with `BREAKING:` in the title.
2. Coordinated PRs to every impl in the registry.
3. A note on `#77` flagging the bump.

Additive changes (a new primitive key, a new optional CLI flag, a new
impl row) need only the PR to this file.
