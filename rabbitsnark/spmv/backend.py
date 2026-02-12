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

"""ZKX backend native CSR SpMV via Python MLIR construction.

Builds an MLIR module with a stablehlo.dot_general over a sparse-encoded
tensor, compiles it through the ZKX backend, and executes with packed CSR
data. This bypasses the ELL workaround and uses the backend's native
EmitMatrixVectorMultiplicationOp.

The key challenge is that sparse_tensor.EncodingAttr doesn't carry NNZ.
We solve this by setting custom function attributes (sparse_nnz_0, ...)
which the C++ backend reads in MlirToZkxComputation to fix NNZ on the
HloModule's parameter shapes before codegen.
"""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING

import numpy as np
from jax._src.interpreters.mlir import dtype_to_ir_type, make_ir_context
from jax._src.zkx_bridge import get_backend
from jaxlib import _jax
from jaxlib.mlir import ir
from jaxlib.mlir.dialects import func as func_dialect
from jaxlib.mlir.dialects import sparse_tensor, stablehlo

if TYPE_CHECKING:
    from jax import Array

    from .csr_matrix import CSRMatrix


def _build_sparse_dot_module(
    n_rows: int,
    n_cols: int,
    nnz: int,
    dtype: np.dtype,
) -> tuple[ir.Module, ir.Context]:
    """Build MLIR module: func(sparse_matrix, dense_vec) -> dense_vec.

    Sets sparse_nnz_0 attribute on the function so MlirToZkxComputation
    can fix NNZ on the HloModule parameter shape.

    Returns:
        (module, context) — context must be kept alive while module is in use.
    """
    ctx = make_ir_context()

    with ctx, ir.Location.unknown(ctx):
        element_type = dtype_to_ir_type(np.dtype(dtype))

        # CSR encoding: [dense, compressed]
        dense_lvl = sparse_tensor.EncodingAttr.build_level_type(
            sparse_tensor.LevelFormat.dense
        )
        compressed_lvl = sparse_tensor.EncodingAttr.build_level_type(
            sparse_tensor.LevelFormat.compressed
        )
        dim_to_lvl = ir.AffineMap.get(
            2, 0, [ir.AffineDimExpr.get(0), ir.AffineDimExpr.get(1)]
        )
        encoding = sparse_tensor.EncodingAttr.get(
            lvl_types=[dense_lvl, compressed_lvl],
            dim_to_lvl=dim_to_lvl,
            lvl_to_dim=dim_to_lvl,
            pos_width=32,
            crd_width=32,
        )

        # Types
        sparse_type = ir.RankedTensorType.get([n_rows, n_cols], element_type, encoding)
        dense_type = ir.RankedTensorType.get([n_cols], element_type)
        result_type = ir.RankedTensorType.get([n_rows], element_type)

        # Module
        module = ir.Module.create(loc=ir.Location.unknown())

        with ir.InsertionPoint(module.body):
            func_type = ir.FunctionType.get([sparse_type, dense_type], [result_type])
            func_op = func_dialect.FuncOp("main", func_type)

            # Set NNZ as function attribute (read by mlir_to_hlo.cc)
            func_op.operation.attributes["sparse_nnz_0"] = ir.IntegerAttr.get(
                ir.IntegerType.get_signless(64), nnz
            )

            entry = ir.Block.create_at_start(func_op.body, [sparse_type, dense_type])

            with ir.InsertionPoint(entry):
                dot_dims = stablehlo.DotDimensionNumbers.get(
                    lhs_batching_dimensions=[],
                    rhs_batching_dimensions=[],
                    lhs_contracting_dimensions=[1],
                    rhs_contracting_dimensions=[0],
                )
                result = stablehlo.dot_general(
                    result_type, entry.arguments[0], entry.arguments[1], dot_dims
                )
                func_dialect.ReturnOp([result])

        return module, ctx


@functools.lru_cache(maxsize=32)
def _compile_sparse_dot(
    n_rows: int,
    n_cols: int,
    nnz: int,
    dtype: np.dtype,
) -> _jax.LoadedExecutable:
    """Compile and cache sparse dot executable by shape key."""
    module, _ctx = _build_sparse_dot_module(n_rows, n_cols, nnz, dtype)

    backend = get_backend()
    devices = _jax.DeviceList(tuple(backend.local_devices()[:1]))
    options = _jax.CompileOptions()

    return backend.compile_and_load(module, devices, options)


def _pack_csr_buffer(
    row_ptrs: np.ndarray,
    col_indices: np.ndarray,
    values_bytes: bytes,
) -> bytes:
    """Pack CSR into flat bytes: [row_ptrs | col_indices | values].

    Format matches ZKX's SparseMatrix::ToCSRBuffer() with s_ignore_size=true:
    row_ptrs as uint32[], col_indices as uint32[], values as raw field bytes.
    """
    return b"".join(
        [
            np.asarray(row_ptrs, dtype=np.uint32).tobytes(),
            np.asarray(col_indices, dtype=np.uint32).tobytes(),
            values_bytes,
        ]
    )


def spmv_backend(matrix: CSRMatrix, x: Array) -> Array:
    """y = A @ x using ZKX native CSR SpMV.

    Packs CSR data as flat uint8 bytes in the format expected by the
    ZKX backend's EmitMatrixVectorMultiplicationOp.

    Args:
        matrix: CSR sparse matrix with Montgomery-form values.
        x: Dense input vector in Montgomery form, shape (n_cols,).

    Returns:
        Dense result vector y = A @ x, shape (n_rows,).
    """
    exe = _compile_sparse_dot(matrix.n_rows, matrix.n_cols, matrix.nnz, x.dtype)

    # Pack CSR data into flat bytes matching SparseMatrix::ToCSRBuffer()
    values_bytes = np.asarray(matrix.values).tobytes()
    csr_bytes = _pack_csr_buffer(
        matrix.row_ptrs, np.asarray(matrix.col_indices, dtype=np.int32), values_bytes
    )

    # Create device buffer from packed CSR bytes
    backend = get_backend()
    device = backend.local_devices()[0]
    csr_buf = backend.buffer_from_pyval(
        np.frombuffer(csr_bytes, dtype=np.uint8), device
    )

    # Execute: sparse_matrix (uint8 buffer) × dense_vec → result_vec
    results = exe.execute_sharded([csr_buf, x])
    out_arrays = results.disassemble_into_single_device_arrays()

    return out_arrays[0][0]
