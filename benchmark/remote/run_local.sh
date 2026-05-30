#!/usr/bin/env bash
#
# Local sweep driver for the cross-impl benchmark suite (#77).
#
# Discovers benchmark/remote/bench_config.*.yaml, runs every target on
# the local box via docker (rabbit's CUDA / ROCm images, sp1-ref-cuda
# from sp1#25 when published), and merges per-target JSONs into one
# consolidated BenchmarkReport at
# benchmark/results/local-<sha>-<ts>.json.
#
# Usage:
#   benchmark/remote/run_local.sh                     # all configs
#   benchmark/remote/run_local.sh <config> [<config>] # specific configs
#
# Vendor-specific docker flags (--gpus all for nvidia, --device flags
# for amd) are picked from each config's hardware.vendor field.
#
# See benchmark/CONTRACT.md ("Local sweep orchestrator") for the
# orchestrator's contract.

set -euo pipefail

REPO_ROOT=$(git -C "$(dirname "$0")" rev-parse --show-toplevel)
cd "$REPO_ROOT"

SHA=$(git rev-parse --short HEAD)
TS=$(date -u +%Y%m%dT%H%M%SZ)
OUT_DIR="benchmark/results"
OUT_PATH="${OUT_DIR}/local-${SHA}-${TS}.json"
mkdir -p "$OUT_DIR"

if [ "$#" -gt 0 ]; then
  CONFIGS=("$@")
else
  # Default: all bench_config.*.yaml.
  shopt -s nullglob
  CONFIGS=(benchmark/remote/bench_config.*.yaml)
  shopt -u nullglob
fi

if [ "${#CONFIGS[@]}" -eq 0 ]; then
  echo "No bench_config.*.yaml found." >&2
  exit 1
fi

TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

failed=0
for cfg in "${CONFIGS[@]}"; do
  echo "=== $cfg ==="
  if ! python3 benchmark/remote/_run_one.py "$cfg" "$TMP"; then
    echo "FAILED: $cfg (continuing)" >&2
    failed=$((failed + 1))
  fi
done

# Merge whatever per-target JSONs landed (skip silently if none — the
# merger reports an empty-benchmarks report rather than crashing).
shopt -s nullglob
JSONS=("$TMP"/*.json)
shopt -u nullglob
python3 benchmark/remote/merge_results.py "${JSONS[@]}" > "$OUT_PATH"
echo
echo "Wrote $OUT_PATH ($(jq -r '.benchmarks | length' "$OUT_PATH" 2>/dev/null || echo '?') benchmarks)"

if [ "$failed" -gt 0 ]; then
  echo "$failed config(s) failed — partial sweep saved." >&2
  exit 2
fi
