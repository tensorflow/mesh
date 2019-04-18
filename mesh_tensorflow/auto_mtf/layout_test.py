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

"""Tests for mesh_tensorflow.auto_mtf.layout."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mesh_tensorflow as mtf
import mesh_tensorflow.auto_mtf  # pylint: disable=unused-import
import tensorflow as tf


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
    print(layout)

    a_dim = mtf.convert_to_dimension(("a", 10))
    b_dim = mtf.convert_to_dimension(("b", 5))
    c_dim = mtf.convert_to_dimension(("c", 20))

    self.assertEqual(layout.tensor_dimension_to_mesh_axis(a_dim, mesh_shape), 1)
    self.assertIsNone(layout.tensor_dimension_to_mesh_axis(b_dim, mesh_shape))
    self.assertEqual(layout.tensor_dimension_to_mesh_axis(c_dim, mesh_shape), 0)


if __name__ == "__main__":
  tf.test.main()
