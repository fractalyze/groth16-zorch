# Copyright 2026 The Groth16Zorch Authors.
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
"""Smoke tests for benchmark scripts.

Runs each benchmark with minimal input sizes to verify:
  1. The benchmark executes without errors.
  2. All entries in the output contain test_vectors.verified == True.
"""
from __future__ import annotations

from absl.testing import absltest

from benchmark.primitives_benchmark import PrimitivesBenchmark


class BenchmarkSmokeTest(absltest.TestCase):
    def _assert_all_verified(self, report):
        """Assert all benchmarks in the report have verified test vectors."""
        self.assertGreater(len(report.benchmarks), 0)
        for name, result in report.benchmarks.items():
            self.assertIsNotNone(
                result.test_vectors,
                f"{name}: missing test_vectors",
            )
            self.assertTrue(
                result.test_vectors.verified,
                f"{name}: test_vectors.verified is False",
            )

    def test_primitives_benchmark(self):
        report = PrimitivesBenchmark().run_to_report(
            ["--sizes=4", "--iterations=1", "--warmup=1"]
        )
        self._assert_all_verified(report)


if __name__ == "__main__":
    absltest.main()
