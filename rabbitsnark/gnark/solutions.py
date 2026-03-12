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

"""Load pre-computed Az/Bz solution vectors from gnark binary export."""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from zk_dtypes import bn254_sf_mont

from .loader import FIELD_ELEM_SIZE


def load_solutions_mont(
    export_dir: Path, domain_size: int
) -> tuple[jax.Array, jax.Array]:
    """Load pre-computed Az/Bz solution vectors from gnark binary export.

    Args:
        export_dir: Directory containing ``solution_a.bin`` and ``solution_b.bin``.
        domain_size: NTT domain size (power of 2, >= num_constraints).

    Returns:
        Tuple of (az, bz) JAX arrays in Montgomery form.
    """
    az = _load_solution_mont(export_dir / "solution_a.bin", domain_size)
    bz = _load_solution_mont(export_dir / "solution_b.bin", domain_size)
    return az, bz


def _load_solution_mont(path: Path, domain_size: int) -> jax.Array:
    """Load pre-computed solution vector as padded bn254_sf_mont JAX array.

    The Go exporter writes raw Montgomery form bytes (gnark fr.Element is
    [4]uint64 Montgomery).  We reinterpret the bytes directly as bn254_sf_mont
    to avoid the double-conversion that would occur if we parsed to Python ints
    and then passed them through the bn254_sf_mont constructor (which auto-
    converts standard -> Montgomery).

    Args:
        path: Binary file of 32-byte LE Montgomery field elements.
        domain_size: NTT domain size (power of 2, >= num_constraints).

    Returns:
        (domain_size,) JAX array in Montgomery form.
    """
    raw = np.fromfile(str(path), dtype=np.uint8)
    num_constraints = raw.size // FIELD_ELEM_SIZE
    # Pad raw bytes to domain_size with zero bytes (Montgomery(0) = 0).
    if num_constraints < domain_size:
        padding = np.zeros(
            (domain_size - num_constraints) * FIELD_ELEM_SIZE, dtype=np.uint8
        )
        raw = np.concatenate([raw, padding])
    mont_np = raw.view(np.dtype(bn254_sf_mont))
    return jnp.array(mont_np.tolist(), dtype=bn254_sf_mont)
