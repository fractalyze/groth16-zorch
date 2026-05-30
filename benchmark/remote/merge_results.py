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
"""Merge per-target BenchmarkReport JSONs into one consolidated report.

Each input JSON is one binary invocation's output (a zkbench
BenchmarkReport). The output is a single BenchmarkReport whose
benchmarks dict is keyed by `${impl}.${primitive}` so a cross-impl run
of the same primitive produces distinct entries (e.g. `rabbit.msm_g1`
vs `gnark.msm_g1`).

Run via: python3 merge_results.py <json>... > merged.json

The orchestrator (run_local.sh) calls this with every per-target JSON
collected from a local sweep.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


def merge(paths: list[str]) -> dict[str, Any]:
    out: dict[str, Any] = {"metadata": None, "benchmarks": {}}
    for path in paths:
        data = json.loads(Path(path).read_text())
        meta = data.get("metadata", {})
        impl = meta.get("implementation", "unknown")
        # Pick the first report's metadata as the run's metadata; per-impl
        # metadata lives inside each benchmark entry.
        if out["metadata"] is None:
            out["metadata"] = meta
        for prim_key, result in data.get("benchmarks", {}).items():
            merged_key = f"{impl}.{prim_key}"
            if merged_key in out["benchmarks"]:
                print(
                    f"warning: duplicate key {merged_key} from {path}; " "later wins",
                    file=sys.stderr,
                )
            # Stash the source impl + path in the merged entry's metadata
            # so the consolidated JSON is self-describing.
            entry_meta = dict(result.get("metadata", {}))
            entry_meta.setdefault("implementation", impl)
            entry_meta.setdefault("source_json", Path(path).name)
            merged_result = dict(result)
            merged_result["metadata"] = entry_meta
            out["benchmarks"][merged_key] = merged_result
    return out


def main() -> int:
    paths = sys.argv[1:]
    if not paths:
        # Empty sweep — emit a shell report so consumers don't crash.
        print(json.dumps({"metadata": None, "benchmarks": {}}, indent=2))
        return 0
    print(json.dumps(merge(paths), indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
