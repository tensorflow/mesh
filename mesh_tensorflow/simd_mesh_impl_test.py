# coding=utf-8
# Copyright 2019 The Mesh TensorFlow Authors.
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
import tensorflow as tf


class SimdMeshImplTest(parameterized.TestCase):

  @parameterized.parameters(
      ([8, 8, 2], [2, 2]),
      ([2, 2, 2], [1, 1]),
      ([8, 8, 2], [1, 8]),
      ([9, 15, 7], [3, 5]),
  )
  def testTile2d(self, physical_shape, tile_shape):
    mesh_shape, logical_to_physical = mtf.simd_mesh_impl.tile_2d(
        physical_shape, tile_shape)
    num_cores = physical_shape[0] * physical_shape[1] * physical_shape[2]
    expected_inner_dim = mtf.Dimension(
        "inner", tile_shape[0] * tile_shape[1] * physical_shape[2])
    expected_outer_dim = mtf.Dimension(
        "outer", num_cores // expected_inner_dim.size)
    self.assertEqual(mesh_shape,
                     mtf.Shape([expected_outer_dim, expected_inner_dim]))
    self.assertCountEqual(logical_to_physical, list(range(num_cores)))


if __name__ == "__main__":
  tf.test.main()
