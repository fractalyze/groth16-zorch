# CLAUDE.md

## Project Overview
rabbitsnark-py is a Python Groth16 prover using Zorch (JAX-based GPU proving).
Parses circom .zkey files and gnark binary exports, generates snarkjs-compatible
proofs. Python counterpart of RabbitSNARK (C++/HLO).

## Current Focus
Q2: E2E proving p99 ≤ 7s on 16 GPUs (excluding verification).
Sprint: E2E correctness — 5 test blocks Phase 1-3 bug-free.
Out of scope: Multi-zkVM 2nd backend, community building, internal tooling, external talks.

## Commands
- Build: `bazel build //...`
- Test: `bazel test //...`
- Alt install: `pip install -e ".[dev]"`

## Why Decisions
- Python Groth16 alongside C++ RabbitSNARK: Zorch framework enables GPU proving via JAX without C++ build complexity.
- Dual build (Bazel + pip): Bazel for CI reproducibility, pip for quick local dev.
- NTT in Montgomery form: ring isomorphism preserves NTT structure, eliminating conversion overhead across entire pipeline.

## Rules
- Do NOT modify cryptographic pairing or elliptic curve arithmetic without expert review.
- Do NOT change Groth16 phase order: Phase 1 (NTT/FFT) → Phase 2 (MSM) → Phase 3 (EC assembly).
- Do NOT use `jnp.arange` with field types — use explicit index arrays.
- Do NOT use `vmap` when it fails with custom ZK dtypes — use `lax.map` for manual batching.
- Proof output MUST be snarkjs-compatible JSON.
- Always run `bazel test //...` before committing.

## Invisible Traps
- NTT operates directly in Montgomery form — no conversions needed if pipeline uses Montgomery consistently. Roots auto-convert via `dtype(root)`.
- Auto-detect `z` witness format only. `az_mont`/`bz_mont` MUST stay Montgomery for EC ops — auto-detecting these breaks proofs silently.
- Phase 3 (EC assembly) always runs on CPU — EC scalar multiply fusions create ~68KB stack frames causing CUDA_ERROR_OUT_OF_MEMORY.

## Knowledge Files
Read ONLY when relevant to your current task:
@.claude/knowledge/architecture.md — Groth16 pipeline (NTT → MSM → EC assembly)
@.claude/knowledge/testing-guide.md — Test conventions and reference data
@.claude/knowledge/solutions.md — Past bug resolution patterns
