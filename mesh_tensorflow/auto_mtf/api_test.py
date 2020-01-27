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

"""Tests for mesh_tensorflow.auto_mtf.layout."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mesh_tensorflow as mtf
import mesh_tensorflow.auto_mtf  # pylint: disable=unused-import
import mesh_tensorflow.auto_mtf.api
import tensorflow.compat.v1 as tf


class LayoutTest(tf.test.TestCase):

  def testLayout(self):
    # Construct a Mesh TensorFlow graph and mesh.
    mtf_graph = mtf.Graph()
    mesh = mtf.Mesh(mtf_graph, "my_mesh")
    x = mtf.zeros(mesh, "a:10,b:5")
    y = mtf.zeros(mesh, "b:5,c:20")
    z = mtf.einsum([x, y], "a:10,c:20")

    # Decide on a mesh shape.
    mesh_shape = mtf.convert_to_shape("m1:4,m2:2")

    # Compute a layout based on the graph and mesh.
    # Note that knowing the identity of the outputs is important to the
    # optimization since they cannot be freed.
    layout = mtf.auto_mtf.layout(mtf_graph, mesh_shape, [z])

    a_dim = mtf.convert_to_dimension(("a", 10))
    b_dim = mtf.convert_to_dimension(("b", 5))
    c_dim = mtf.convert_to_dimension(("c", 20))

    self.assertEqual(layout.tensor_dimension_to_mesh_axis(a_dim, mesh_shape), 1)
    self.assertIsNone(layout.tensor_dimension_to_mesh_axis(b_dim, mesh_shape))
    self.assertEqual(layout.tensor_dimension_to_mesh_axis(c_dim, mesh_shape), 0)

  def testLayoutAndMeshShape(self):
    # Same as previous test, but don't specify a 4x2 mesh.
    mtf_graph = mtf.Graph()
    mesh = mtf.Mesh(mtf_graph, "my_mesh")
    x = mtf.zeros(mesh, "a:10,b:5")
    y = mtf.zeros(mesh, "b:5,c:20")
    z = mtf.einsum([x, y], "a:10,c:20")

    layout, mesh_shape = mtf.auto_mtf.layout_and_mesh_shape(mtf_graph, 8, [z])

    a_dim = mtf.convert_to_dimension(("a", 10))
    b_dim = mtf.convert_to_dimension(("b", 5))
    c_dim = mtf.convert_to_dimension(("c", 20))

    self.assertEqual(layout.tensor_dimension_to_mesh_axis(a_dim, mesh_shape), 1)
    self.assertIsNone(layout.tensor_dimension_to_mesh_axis(b_dim, mesh_shape))
    self.assertEqual(layout.tensor_dimension_to_mesh_axis(c_dim, mesh_shape), 0)

    self.assertCountEqual(mesh_shape.dims,
                          [mtf.Dimension("mesh_0", 4),
                           mtf.Dimension("mesh_1", 2)])

    layout, mesh_shape = mtf.auto_mtf.layout_and_mesh_shape(
        mtf_graph, 8, [z], 1)

    self.assertIsNone(layout.tensor_dimension_to_mesh_axis(a_dim, mesh_shape))
    self.assertIsNone(layout.tensor_dimension_to_mesh_axis(b_dim, mesh_shape))
    self.assertIsNone(layout.tensor_dimension_to_mesh_axis(c_dim, mesh_shape))

    self.assertCountEqual(mesh_shape.dims, [mtf.Dimension("mesh_0", 8)])

  def testMeshShapeIterator(self):
    self.assertCountEqual(
        list(mesh_tensorflow.auto_mtf.api._mesh_shape_iterator(1)), [[1]])
    self.assertCountEqual(
        list(mesh_tensorflow.auto_mtf.api._mesh_shape_iterator(2)), [[2]])
    self.assertCountEqual(
        list(mesh_tensorflow.auto_mtf.api._mesh_shape_iterator(4)),
        [[4], [2, 2]])
    self.assertCountEqual(
        list(mesh_tensorflow.auto_mtf.api._mesh_shape_iterator(8)),
        [[8], [4, 2], [2, 2, 2]])
    self.assertCountEqual(
        list(mesh_tensorflow.auto_mtf.api._mesh_shape_iterator(512)),
        [[512],
         [256, 2],
         [128, 4],
         [128, 2, 2],
         [64, 8],
         [64, 4, 2],
         [64, 2, 2, 2],
         [32, 16],
         [32, 8, 2],
         [32, 4, 4],
         [32, 4, 2, 2],
         [32, 2, 2, 2, 2],
         [16, 16, 2],
         [16, 8, 4],
         [16, 8, 2, 2],
         [16, 4, 4, 2],
         [16, 4, 2, 2, 2],
         [16, 2, 2, 2, 2, 2],
         [8, 8, 8],
         [8, 8, 4, 2],
         [8, 8, 2, 2, 2],
         [8, 4, 4, 4],
         [8, 4, 4, 2, 2],
         [8, 4, 2, 2, 2, 2],
         [8, 2, 2, 2, 2, 2, 2],
         [4, 4, 4, 4, 2],
         [4, 4, 4, 2, 2, 2],
         [4, 4, 2, 2, 2, 2, 2],
         [4, 2, 2, 2, 2, 2, 2, 2],
         [2, 2, 2, 2, 2, 2, 2, 2, 2]])

    self.assertCountEqual(
        list(mesh_tensorflow.auto_mtf.api._mesh_shape_iterator(512, 1)),
        [[512]])
    self.assertCountEqual(
        list(mesh_tensorflow.auto_mtf.api._mesh_shape_iterator(512, 2)),
        [[512], [256, 2], [128, 4], [64, 8], [32, 16]])


if __name__ == "__main__":
  tf.disable_v2_behavior()
  tf.test.main()
