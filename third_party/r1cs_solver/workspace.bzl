# Copyright 2026 The RabbitSNARK Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Workspace macro for the r1cs-solver native dependency.

Defines the r1cs_solver http_archive. Transitive deps are loaded from
@r1cs_solver//bazel:r1cs_solver_deps.bzl in WORKSPACE.bazel.
"""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

R1CS_SOLVER_COMMIT = "21a18581e1d0a651f2a7743f58b319afb90159a4"
R1CS_SOLVER_SHA256 = "5f2f5eeba1c9e80ac518b7c0f27896f6cc53dff9d9ca67296bb1fa3abbfc47b2"

def repo():
    """Define the r1cs_solver repository."""
    if not native.existing_rule("r1cs_solver"):
        http_archive(
            name = "r1cs_solver",
            sha256 = R1CS_SOLVER_SHA256,
            strip_prefix = "r1cs-solver-{commit}".format(commit = R1CS_SOLVER_COMMIT),
            urls = ["https://github.com/fractalyze/r1cs-solver/archive/{commit}.tar.gz".format(commit = R1CS_SOLVER_COMMIT)],
        )
