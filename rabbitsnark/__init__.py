# Copyright 2025 The RabbitSNARK Authors.
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

"""RabbitSNARK Python library for circom file parsing and ZK primitives."""

__version__ = "0.1.0"

# groth16.verifier imports jax._src.lax.zk_ops (pairing_check) which is
# not yet available; defer import to avoid ImportError at package level.
import importlib as _importlib

from rabbitsnark import msm, ntt, spmv


def __getattr__(name: str):
    if name == "groth16":
        return _importlib.import_module("rabbitsnark.groth16")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "__version__",
    "groth16",
    "msm",
    "ntt",
    "spmv",
]
