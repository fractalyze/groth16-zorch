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

"""SELL (Sliced ELL) sparse matrix format for workload-balanced SpMV.

Plain ELL pads every row to max_nnz_per_row, wasting 80-95% of memory on
R1CS matrices where a few rows are dense but most are sparse. SELL sorts
rows by NNZ count (descending), auto-partitions into groups of similar NNZ,
and pads each partition independently — reducing memory and compute waste.

SELL layout (P partitions):
    For each partition p:
        col_indices_p: flat (partition_sizes[p] * partition_max_nnz[p],)
        values_p:      flat (partition_sizes[p] * partition_max_nnz[p],)
    inverse_perm: (n_rows,) restores original row order via gather
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import jax.numpy as jnp
import numpy as np

if TYPE_CHECKING:
    from jax import Array

    from .csr_matrix import CSRMatrix


class SELLConfig(NamedTuple):
    """Static SELL configuration (hashable for JIT static_argnums).

    Attributes:
        n_rows: Total number of rows.
        n_cols: Number of columns.
        num_partitions: Number of SELL partitions (P).
        partition_sizes: Per-partition row counts.
        partition_max_nnz: Per-partition max nonzeros per row.
    """

    n_rows: int
    n_cols: int
    num_partitions: int
    partition_sizes: tuple[int, ...]
    partition_max_nnz: tuple[int, ...]


@dataclass
class SELLMatrix:
    """SELL sparse matrix with Montgomery-form field element values.

    Attributes:
        config: Static SELL configuration.
        partition_col_indices: Per-partition flat int32 column index arrays.
        partition_values: Per-partition flat field-dtype value arrays.
        inverse_perm: Gather indices to restore original row order.
    """

    config: SELLConfig
    partition_col_indices: tuple[Array, ...]
    partition_values: tuple[Array, ...]
    inverse_perm: Array

    @staticmethod
    def _optimal_partition(
        sorted_row_lengths: np.ndarray,
        max_partitions: int,
        tolerance: float = 0.05,
    ) -> list[tuple[int, int]]:
        """Find partition boundaries minimizing total padded elements via DP.

        Compresses the sorted NNZ distribution into K unique-value groups,
        then runs DP over all P from 1..max_P to find the partition boundaries
        that minimize total_elements = sum(partition_rows * partition_max_nnz).
        Selects the smallest P whose cost is within ``tolerance`` of the best.

        Time: O(n + max_P * K²) where K = unique NNZ count (typically < 500).

        Args:
            sorted_row_lengths: Row NNZ counts sorted descending.
            max_partitions: Maximum number of partitions to consider.
            tolerance: Accept smallest P within this fraction of optimal cost.

        Returns:
            List of (start, end) index pairs for each partition.
        """
        n = len(sorted_row_lengths)
        if n == 0:
            return [(0, 0)]

        # Compress consecutive rows with same NNZ into groups (vectorized).
        changes = np.where(np.diff(sorted_row_lengths) != 0)[0] + 1
        group_starts = np.concatenate([[0], changes, [n]]).tolist()
        group_nnz = [max(int(sorted_row_lengths[s]), 1) for s in group_starts[:-1]]

        K = len(group_nnz)
        P = min(max_partitions, K)

        if P <= 1 or K <= 1:
            return [(0, n)]

        # dp[p][k] = min total padded elements for first k groups using p
        # partitions.  parent[p][k] = split point j for backtracking.
        INF = float("inf")
        dp = [[INF] * (K + 1) for _ in range(P + 1)]
        parent = [[0] * (K + 1) for _ in range(P + 1)]
        dp[0][0] = 0

        for p in range(1, P + 1):
            for k in range(p, K + 1):
                for j in range(p - 1, k):
                    rows = group_starts[k] - group_starts[j]
                    cost = dp[p - 1][j] + rows * group_nnz[j]
                    if cost < dp[p][k]:
                        dp[p][k] = cost
                        parent[p][k] = j

        # Pick smallest P within tolerance of the best achievable cost.
        costs = [dp[p][K] for p in range(1, P + 1)]
        min_cost = min(costs)
        threshold = min_cost * (1.0 + tolerance)

        best_p = P  # fallback
        for p in range(1, P + 1):
            if costs[p - 1] <= threshold:
                best_p = p
                break

        # Backtrack to recover partition boundaries.
        partitions: list[tuple[int, int]] = []
        k = K
        for p in range(best_p, 0, -1):
            j = parent[p][k]
            partitions.append((group_starts[j], group_starts[k]))
            k = j
        partitions.reverse()

        return partitions

    @classmethod
    def from_csr(
        cls,
        csr: CSRMatrix,
        max_partitions: int = 32,
    ) -> SELLMatrix:
        """Convert a CSR matrix to SELL format.

        1. Compute row lengths from CSR row_ptrs
        2. Sort rows by NNZ descending
        3. Find optimal partition boundaries via DP (minimizes total elements)
        4. Build per-partition ELL arrays
        5. Build inverse permutation for result reordering

        Args:
            csr: Source CSR matrix.
            max_partitions: Maximum number of partitions.

        Returns:
            SELLMatrix with P partitions.
        """
        row_lengths = np.diff(csr.row_ptrs)
        sorted_indices = np.argsort(-row_lengths)
        sorted_lengths = row_lengths[sorted_indices]

        # inverse_perm[sorted_pos] = original_row  →  used as gather index
        inverse_perm = np.argsort(sorted_indices)

        partition_bounds = cls._optimal_partition(sorted_lengths, max_partitions)

        # Extract CSR data as numpy arrays to avoid JAX iteration issues.
        # Using np.asarray avoids the ZKX JAX 'sign' primitive limitation
        # that triggers when iterating over large JAX arrays with int().
        col_indices_np = np.asarray(csr.col_indices)
        values_np = np.asarray(csr.values)

        partition_col_indices = []
        partition_values = []
        partition_sizes = []
        partition_max_nnz = []

        for start, end in partition_bounds:
            p_size = end - start
            if p_size == 0:
                continue

            p_max_nnz = max(int(sorted_lengths[start]), 1)

            ell_cols = np.zeros(p_size * p_max_nnz, dtype=np.int32)
            ell_vals = np.zeros(p_size * p_max_nnz, dtype=values_np.dtype)

            for local_i, sorted_i in enumerate(range(start, end)):
                orig_row = int(sorted_indices[sorted_i])
                row_start = int(csr.row_ptrs[orig_row])
                row_end = int(csr.row_ptrs[orig_row + 1])
                row_len = row_end - row_start

                base = local_i * p_max_nnz
                ell_cols[base : base + row_len] = col_indices_np[row_start:row_end]
                ell_vals[base : base + row_len] = values_np[row_start:row_end]

            partition_col_indices.append(jnp.array(ell_cols.tolist(), dtype=jnp.int32))
            partition_values.append(
                jnp.array(ell_vals.tolist(), dtype=csr.values.dtype)
            )
            partition_sizes.append(p_size)
            partition_max_nnz.append(p_max_nnz)

        config = SELLConfig(
            n_rows=csr.n_rows,
            n_cols=csr.n_cols,
            num_partitions=len(partition_sizes),
            partition_sizes=tuple(partition_sizes),
            partition_max_nnz=tuple(partition_max_nnz),
        )

        return cls(
            config=config,
            partition_col_indices=tuple(partition_col_indices),
            partition_values=tuple(partition_values),
            inverse_perm=jnp.array(inverse_perm.tolist(), dtype=jnp.int32),
        )

    def partition_arrays(self) -> tuple[Array, ...]:
        """Build flat alternating (col_indices, values) tuple for all partitions.

        Returns:
            Tuple of (col_0, val_0, col_1, val_1, ..., col_{P-1}, val_{P-1}).
        """
        arrays: list[Array] = []
        for p in range(self.config.num_partitions):
            arrays.append(self.partition_col_indices[p])
            arrays.append(self.partition_values[p])
        return tuple(arrays)

    def stats(self) -> dict:
        """Compute partition statistics and memory savings vs plain ELL.

        Returns:
            Dictionary with partition details and savings metrics.
        """
        cfg = self.config
        sell_total = sum(
            cfg.partition_sizes[p] * cfg.partition_max_nnz[p]
            for p in range(cfg.num_partitions)
        )
        plain_ell_max_nnz = max(cfg.partition_max_nnz) if cfg.num_partitions > 0 else 1
        plain_ell_total = cfg.n_rows * plain_ell_max_nnz
        savings = 1.0 - sell_total / plain_ell_total if plain_ell_total > 0 else 0.0

        return {
            "n_rows": cfg.n_rows,
            "n_cols": cfg.n_cols,
            "num_partitions": cfg.num_partitions,
            "partition_sizes": cfg.partition_sizes,
            "partition_max_nnz": cfg.partition_max_nnz,
            "sell_total_elements": sell_total,
            "plain_ell_total_elements": plain_ell_total,
            "memory_savings": savings,
        }
