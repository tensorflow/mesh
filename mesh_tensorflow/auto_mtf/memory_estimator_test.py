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

"""Tests for mesh_tensorflow.cost_estimator.memory_estimator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mesh_tensorflow as mtf
from mesh_tensorflow.auto_mtf import memory_estimator
import tensorflow.compat.v1 as tf


class MemoryEstimatorTest(tf.test.TestCase):

  def setUp(self):
    super(MemoryEstimatorTest, self).setUp()
    mtf_graph = mtf.Graph()
    mesh = mtf.Mesh(mtf_graph, 'lowering_context_mesh')

    a_dim = mtf.Dimension('a', 3)
    b_dim = mtf.Dimension('b', 4)
    c_dim = mtf.Dimension('c', 5)

    x = (mtf.Constant(mesh, 0, mtf.Shape([a_dim, b_dim]), tf.int32, 'X')
         .outputs[0])
    y = (mtf.Constant(mesh, 0, mtf.Shape([b_dim, c_dim]), tf.int32, 'Y')
         .outputs[0])
    z = (mtf.EinsumOperation([x, y], mtf.Shape([a_dim, c_dim]), name='Z')
         .outputs[0])

    mesh_shape = mtf.Shape([('m1', 4), ('m2', 3)])

    self.estimator = memory_estimator.MemoryEstimator(
        mtf_graph, mesh_shape, [z])

  def test_LayoutValidator(self):
    validator = self.estimator.get_layout_validator()
    self.assertCountEqual(validator.splittable_mtf_dimension_names,
                          ['a', 'b', 'c'])
    self.assertFalse(validator.is_valid_assignment('a', 'm1'))
    self.assertTrue(validator.is_valid_assignment('a', 'm2'))

  def test_GraphInterface(self):
    graph = self.estimator.get_graph_interface()
    self.assertCountEqual(list(graph.get_all_operation_names()),
                          ['X', 'Y', 'Z'])

    self.assertEqual(graph.get_tensor_shape('X:0'), tf.TensorShape([3, 4]))
    self.assertEqual(graph.get_tensor_shape('Z:0'), tf.TensorShape([3, 5]))
    self.assertFalse(graph.is_tensor_final('Y:0'))
    self.assertTrue(graph.is_tensor_final('Z:0'))


if __name__ == '__main__':
  tf.disable_v2_behavior()
  tf.test.main()
