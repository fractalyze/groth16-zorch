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

"""Circom witness calculator via compiled circuit shared library.

Wraps the circom MLIR witness calculator (compiled to .so) to compute
witness vectors from circuit inputs.  The .so is produced by compiling
circom circuit MLIR with the r1cs-solver build pipeline.

Usage:
    calc = CircomWitnessCalculator("/path/to/circuit.so")
    witness = calc.compute_witness(inputs, w2s)
    # witness is a (witness_size,) bn254_sf_mont numpy array
"""

from __future__ import annotations

import ctypes
import json
from pathlib import Path

import numpy as np
from zk_dtypes import bn254_sf_mont

FIELD_ELEM_SIZE = 32  # 256-bit field element = 32 bytes


class _StridedMemRef1D(ctypes.Structure):
    """Mirrors MLIR's StridedMemRefType<T, 1>."""

    _fields_ = [
        ("basePtr", ctypes.c_void_p),
        ("data", ctypes.c_void_p),
        ("offset", ctypes.c_int64),
        ("sizes", ctypes.c_int64),
        ("strides", ctypes.c_int64),
    ]


def _make_memref(arr: np.ndarray) -> _StridedMemRef1D:
    ref = _StridedMemRef1D()
    ptr = arr.ctypes.data
    ref.basePtr = ptr
    ref.data = ptr
    ref.offset = 0
    ref.sizes = arr.shape[0]
    ref.strides = 1
    return ref


_REF_PTR = ctypes.POINTER(_StridedMemRef1D)
_MONT_DT = np.dtype(bn254_sf_mont)


class CircomWitnessCalculator:
    """Compute witnesses from circom circuit inputs via compiled .so.

    The .so must export the following C interface functions:
    - ``_mlir_ciface_circuit_main(signals, subcmps)``
    - ``_mlir_ciface_circom_get_total_signals()``
    - ``_mlir_ciface_circom_get_num_components()``
    - ``_mlir_ciface_circom_get_num_outputs()``
    - ``_mlir_ciface_circom_get_witness_size()``
    - ``_mlir_ciface_circom_get_num_inputs()``
    - ``_mlir_ciface_to_mont_inplace(arr, start, count)``
    - ``_mlir_ciface_from_mont_inplace(arr, start, count)``
    """

    def __init__(self, circuit_so_path: str | Path):
        self._lib = ctypes.CDLL(str(circuit_so_path))

        # Set up function signatures
        self._lib._mlir_ciface_circuit_main.restype = None
        self._lib._mlir_ciface_circuit_main.argtypes = [_REF_PTR, _REF_PTR]

        self._lib._mlir_ciface_to_mont_inplace.restype = None
        self._lib._mlir_ciface_to_mont_inplace.argtypes = [
            _REF_PTR,
            ctypes.c_int64,
            ctypes.c_int64,
        ]
        self._lib._mlir_ciface_from_mont_inplace.restype = None
        self._lib._mlir_ciface_from_mont_inplace.argtypes = [
            _REF_PTR,
            ctypes.c_int64,
            ctypes.c_int64,
        ]

        # Query circuit metadata
        for fn_name in (
            "_mlir_ciface_circom_get_total_signals",
            "_mlir_ciface_circom_get_num_components",
            "_mlir_ciface_circom_get_num_outputs",
            "_mlir_ciface_circom_get_witness_size",
            "_mlir_ciface_circom_get_num_inputs",
        ):
            getattr(self._lib, fn_name).restype = ctypes.c_int64
            getattr(self._lib, fn_name).argtypes = []

        self.total_signals = self._lib._mlir_ciface_circom_get_total_signals()
        self.num_components = self._lib._mlir_ciface_circom_get_num_components()
        self.num_outputs = self._lib._mlir_ciface_circom_get_num_outputs()
        self.witness_size = self._lib._mlir_ciface_circom_get_witness_size()
        self.num_inputs = self._lib._mlir_ciface_circom_get_num_inputs()

    def compute_witness(
        self,
        inputs: dict[str, str | list[str]],
        w2s: list[int],
    ) -> np.ndarray:
        """Compute witness from circuit inputs.

        Args:
            inputs: Circuit inputs as {signal_name: value_or_list}.
                Values are decimal strings.
            w2s: Witness-to-signal mapping (w2s[i] = signal index for
                witness index i).

        Returns:
            (witness_size,) bn254_sf_mont numpy array.
        """
        # Allocate signal and subcmp buffers
        signals_buf = np.zeros(self.total_signals * FIELD_ELEM_SIZE, dtype=np.uint8)
        subcmps_buf = np.zeros(self.num_components, dtype=np.int64)

        # Signal layout: [constant_1, outputs..., inputs..., intermediates...]
        # Set constant wire (signal 0) = 1
        one = int.to_bytes(1, FIELD_ELEM_SIZE, "little")
        signals_buf[:FIELD_ELEM_SIZE] = np.frombuffer(one, dtype=np.uint8)

        # Map inputs: use w2s to find signal indices for witness indices.
        # Witness indices for inputs start at (1 + num_outputs).
        input_start = 1 + self.num_outputs
        input_idx = 0
        for _name, values in inputs.items():
            if isinstance(values, str):
                values = [values]
            for val_str in values:
                witness_idx = input_start + input_idx
                if witness_idx >= len(w2s):
                    raise ValueError(
                        f"Input index {witness_idx} out of w2s range " f"({len(w2s)})"
                    )
                signal_idx = w2s[witness_idx]
                val_bytes = int(val_str).to_bytes(FIELD_ELEM_SIZE, "little")
                offset = signal_idx * FIELD_ELEM_SIZE
                signals_buf[offset : offset + FIELD_ELEM_SIZE] = np.frombuffer(
                    val_bytes, dtype=np.uint8
                )
                input_idx += 1

        # Convert to Montgomery form
        mr_signals = _make_memref(signals_buf)
        self._lib._mlir_ciface_to_mont_inplace(
            ctypes.byref(mr_signals),
            ctypes.c_int64(0),
            ctypes.c_int64(self.total_signals),
        )

        # Run circuit
        mr_subcmps = _make_memref(subcmps_buf)
        self._lib._mlir_ciface_circuit_main(
            ctypes.byref(mr_signals), ctypes.byref(mr_subcmps)
        )

        # Convert back from Montgomery form
        self._lib._mlir_ciface_from_mont_inplace(
            ctypes.byref(mr_signals),
            ctypes.c_int64(0),
            ctypes.c_int64(self.total_signals),
        )

        # Extract witness using w2s mapping
        witness = np.zeros(self.witness_size * FIELD_ELEM_SIZE, dtype=np.uint8)
        for w_idx in range(self.witness_size):
            s_idx = w2s[w_idx]
            if s_idx < 0 or s_idx >= self.total_signals:
                raise ValueError(
                    f"Witness index {w_idx}: signal index {s_idx} out of "
                    f"range [0, {self.total_signals})"
                )
            src = s_idx * FIELD_ELEM_SIZE
            dst = w_idx * FIELD_ELEM_SIZE
            witness[dst : dst + FIELD_ELEM_SIZE] = signals_buf[
                src : src + FIELD_ELEM_SIZE
            ]

        return witness.view(_MONT_DT)


def load_w2s(path: str | Path) -> list[int]:
    """Load witness-to-signal mapping from JSON file."""
    with open(path) as f:
        return json.load(f)
