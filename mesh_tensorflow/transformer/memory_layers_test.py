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

# Lint as: python3
"""Tests for mesh_tensorflow.transformer.memory_layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mesh_tensorflow as mtf
from mesh_tensorflow.transformer import memory_layers

import mock
import numpy as np
import tensorflow.compat.v1 as tf


class FlatKeyValueMemoryTest(tf.test.TestCase):

  def setUp(self):
    super(FlatKeyValueMemoryTest, self).setUp()
    self.graph = mtf.Graph()
    self.mesh = mtf.Mesh(self.graph, "mtf_mesh")
    self.variable_dtype = mtf.VariableDType(activation_dtype=tf.float32)

    self.addCleanup(mock.patch.stopall)
    self.initializer_mock = mock.MagicMock()
    random_normal_initializer_mock = mock.patch.object(
        tf, "random_normal_initializer").start()
    random_normal_initializer_mock.return_value = self.initializer_mock

  def _export_to_tf_tensor(self, mtf_tensor):
    mesh_impl = mtf.placement_mesh_impl.PlacementMeshImpl(
        shape=[], layout={}, devices=[""])
    lowering = mtf.Lowering(self.graph, {self.mesh: mesh_impl})
    return lowering, lowering.export_to_tf_tensor(mtf_tensor)

  def test_call_shape(self):
    key_size = 5
    value_size = 10
    n_keys = 6
    n_heads = 2
    knn = 3

    seq_len = 4
    batch = 5

    model_dim = mtf.Dimension("model", value_size)
    seq_dim = mtf.Dimension("length", seq_len)
    batch_dim = mtf.Dimension("batch", batch)

    def initialize(shape, dtype):
      return tf.reshape(1 + tf.range(np.prod(shape), dtype=dtype), shape)

    self.initializer_mock.side_effect = initialize

    kv_memory = memory_layers.ProductKeyValueMemory(key_size, n_keys,
                                                    n_heads, knn)

    mtf_x = mtf.ones(self.mesh, mtf.Shape([batch_dim, seq_dim, model_dim]))
    context = mock.MagicMock()
    context.mesh = self.mesh
    context.variable_dtype = tf.float32

    out_tensor = kv_memory.call(context, mtf_x)

    # Dimensions should be untouched
    self.assertEqual(mtf_x.shape, out_tensor.shape)

  def test_get_indices(self):
    key_size = 2
    n_keys = 3
    product_size = 2
    head_size = 2
    batch = 2
    seq_len = 2
    knn = 2

    n_key_dim = mtf.Dimension("n_keys", n_keys)
    key_dim = mtf.Dimension("key", key_size // 2)
    seq_dim = mtf.Dimension("length", seq_len)
    batch_dim = mtf.Dimension("batch", batch)
    head_dim = mtf.Dimension("n_heads", head_size)
    product_dim = mtf.Dimension("product_key", product_size)
    knn_dim = mtf.Dimension("knn", knn)

    query_shape = mtf.Shape([batch_dim, seq_dim, head_dim,
                             product_dim, key_dim])
    keys_shape = mtf.Shape([head_dim, product_dim, n_key_dim, key_dim])
    query = mtf.ones(self.mesh, query_shape)

    keys_vals = [
        [
            [[4], [1], [2]],
            [[2], [-1], [2]],
        ],
        [
            [[1], [2], [5]],
            [[6], [1], [4]],
        ],
    ]
    # h1:
    #   First scores:
    #   [4, 2]
    #   [2, 2]
    #   Cartesian added scores:
    #   [6, 6]
    #   Indices:
    #   [0, 2]    [0*n_k + 0, 0*n_k + 2]
    # h2:
    #   First scores:
    #   [5, 2]
    #   [6, 4]
    #   Cartesian added scores:
    #   [11, 9]
    #   Indices:
    #   [6, 8]   [2*n_k+0, 2*n_k+2]
    expected_scores = np.broadcast_to(np.array([[6, 6], [11, 9]]),
                                      [batch, seq_len, head_size, knn])
    expected_indices = np.broadcast_to(np.array([[0, 2], [6, 8]]),
                                       [batch, seq_len, head_size, knn])

    keys = mtf.constant(self.mesh, keys_vals, keys_shape)

    pkm = memory_layers.ProductKeyValueMemory(key_size, n_keys, head_size, knn)
    mtf_scores, mtf_indices = pkm.get_indices(keys, query)

    # Shapes.
    expected_shape = mtf.Shape([batch_dim, seq_dim, head_dim, knn_dim])
    self.assertEqual(expected_shape, mtf_scores.shape)
    self.assertEqual(expected_shape, mtf_indices.shape)

    # Values
    lowering_s, scores = self._export_to_tf_tensor(mtf_scores)
    lowering_i, indices = self._export_to_tf_tensor(mtf_indices)
    self.evaluate(tf.global_variables_initializer())
    self.evaluate(lowering_s.copy_masters_to_slices())
    self.evaluate(lowering_i.copy_masters_to_slices())
    scores, indices = self.evaluate([scores, indices])

    self.assertAllEqual(expected_scores, scores)
    self.assertAllEqual(expected_indices, indices)


if __name__ == "__main__":
  tf.test.main()
