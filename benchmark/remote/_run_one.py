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
"""Per-config runner used by run_local.sh.

Reads one bench_config.<target>.yaml, pulls the image, and runs every
entry in targets[]. Per-target JSONs land in <out_dir> (mounted at
/output inside the container) — run_local.sh's merger picks them up.

Vendor-specific docker flags are picked from hardware.vendor:
  nvidia -> --gpus all          (requires nvidia-container-toolkit)
  amd    -> --device /dev/kfd --device /dev/dri --group-add video
                                (requires ROCm host setup)
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def _docker_gpu_args(vendor: str) -> list[str]:
    if vendor == "nvidia":
        return ["--gpus", "all"]
    if vendor == "amd":
        return [
            "--device",
            "/dev/kfd",
            "--device",
            "/dev/dri",
            "--group-add",
            "video",
        ]
    return []


def run_one(config_path: Path, out_dir: Path) -> int:
    cfg = yaml.safe_load(config_path.read_text())
    image = cfg["image"]
    vendor = cfg.get("hardware", {}).get("vendor", "")
    fixture = cfg.get("fixture")
    targets = cfg["targets"]

    print(f"[{cfg['name']}] pulling {image}", file=sys.stderr)
    pull = subprocess.run(["docker", "pull", image], check=False)
    if pull.returncode != 0:
        print(
            f"[{cfg['name']}] pull failed (image may be unpublished)", file=sys.stderr
        )
        return pull.returncode

    failures = 0
    for target in targets:
        name = target["name"]
        mounts = ["-v", f"{out_dir.absolute()}:/output"]
        if fixture:
            mounts += [
                "-v",
                f"{fixture['local_path']}:{fixture['container_path']}:ro",
            ]
        cmd = (
            ["docker", "run", "--rm"]
            + _docker_gpu_args(vendor)
            + mounts
            + [image]
            + list(target["container_args"])
        )
        print(f"[{cfg['name']}/{name}] {' '.join(cmd)}", file=sys.stderr)
        rc = subprocess.run(cmd, check=False).returncode
        if rc != 0:
            print(f"[{cfg['name']}/{name}] FAILED rc={rc}", file=sys.stderr)
            failures += 1
    return failures


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("config", type=Path)
    parser.add_argument("out_dir", type=Path)
    args = parser.parse_args()
    return run_one(args.config, args.out_dir)


if __name__ == "__main__":
    sys.exit(main())
