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

"""Workspace macro for the zk_dtypes dependency.

zk_dtypes supplies the BN254 field/curve numpy dtypes used throughout the
prover, and its ``//third_party/py`` macros bootstrap the hermetic Python
toolchain + pip (``@pypi``) for this WORKSPACE. It is fetched directly here
(rather than transitively) so the build no longer pulls prime-ir / LLVM.
"""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

ZK_DTYPES_COMMIT = "99a8ffa6fb4a82687c828e2b71532a8bd29e2759"
ZK_DTYPES_SHA256 = "eea14c863a609d35cd8a2936da3d6f704c2ca4454586634bdfd47107966aa711"

def repo():
    """Define the zk_dtypes repository."""
    if not native.existing_rule("zk_dtypes"):
        http_archive(
            name = "zk_dtypes",
            sha256 = ZK_DTYPES_SHA256,
            strip_prefix = "zk_dtypes-{commit}".format(commit = ZK_DTYPES_COMMIT),
            urls = ["https://github.com/fractalyze/zk_dtypes/archive/{commit}/zk_dtypes-{commit}.tar.gz".format(commit = ZK_DTYPES_COMMIT)],
        )
