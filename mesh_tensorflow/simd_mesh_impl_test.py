# coding=utf-8
# Copyright 2020 The Mesh TensorFlow Authors.
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

"""Tests for Mesh TensorFlow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

import mesh_tensorflow as mtf
import tensorflow.compat.v1 as tf


class SimdMeshImplTest(parameterized.TestCase):

  @parameterized.parameters(
      ([8, 8, 2], [("dp", None)]),
      ([8, 8, 2], [("dp", None), ("mp", [1, 1, 2])]),
      ([8, 8, 2], [("dp", [8, 8, 1]), ("mp", [1, 1, 2])]),
      ([8, 8, 2], [("dp", None), ("mp", [2, 8, 1])]),
      ([8, 8, 2], [("dp", None), ("mp1", [1, 8, 1]), ("mp2", [8, 1, 2])]),
      ([8, 8, 2], [("dp", None), ("mp1", [2, 2, 1]), ("mp2", [2, 2, 1])]),
      ([9, 15, 7], [("d1", [3, 5, 1]), ("d2", [3, 3, 7])]),
  )
  def testHierarchicalTiling(self, physical_shape, spec):
    hierarchical_tiling = mtf.simd_mesh_impl.HierarchicalTiling(
        spec, physical_shape)
    mesh_shape = hierarchical_tiling.mesh_shape
    logical_to_physical = hierarchical_tiling.logical_to_physical
    num_cores = physical_shape[0] * physical_shape[1] * physical_shape[2]
    expected_mesh_shape = (
        mtf.simd_mesh_impl.HierarchicalTiling.spec_to_mesh_shape(
            spec, num_cores))
    self.assertEqual(mesh_shape, expected_mesh_shape)
    self.assertCountEqual(logical_to_physical, list(range(num_cores)))

  @parameterized.parameters(
      ([128], [8, 8, 2]),
      ([8, 16], [8, 8, 2]),
      ([32, 4], [8, 8, 2]),
      ([2, 32, 4], [256]),
      ([4, 4, 8], [8, 8, 2]),
  )
  def testLogicalToPhysical(self, physical_shape, logical_shape):
    logical_to_physical = mtf.simd_mesh_impl.auto_logical_to_physical_tpu(
        physical_shape, logical_shape)
    self.assertCountEqual(
        logical_to_physical, list(range(mtf.list_product(physical_shape))))


if __name__ == "__main__":
  tf.disable_v2_behavior()
  tf.test.main()
