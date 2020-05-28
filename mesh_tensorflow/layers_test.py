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

"""Tests for Mesh TensorFlow layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

import mesh_tensorflow as mtf
import mock
import numpy as np

import tensorflow.compat.v1 as tf
from tensorflow.python.framework import test_util  # pylint:disable=g-direct-tensorflow-import


def initialize_by_shape(shape_to_value):
  """Create an initializer with values specified by tensor shape."""

  def initialize(shape, dtype, **unused_kwargs):
    shape = tuple(shape)
    if shape not in shape_to_value:
      raise ValueError(
          "Shape {} not found in shape to value map.".format(shape))
    return tf.reshape(
        tf.constant(shape_to_value[tuple(shape)], dtype=dtype), shape)

  return initialize


class LayersTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      (4, True, "not_channels"),
      (8, False, "channels"),
  )
  def testDense(self, units, use_bias, new_dim_name):
    batch = 2
    channels = 3
    inputs = tf.random_normal([batch, channels])

    graph = mtf.Graph()
    mesh = mtf.Mesh(graph, "my_mesh")
    batch_dim = mtf.Dimension("batch", batch)
    channels_dim = mtf.Dimension("channels", channels)
    new_dim = mtf.Dimension(new_dim_name, units)

    mtf_inputs = mtf.import_tf_tensor(
        mesh, inputs, shape=mtf.Shape([batch_dim, channels_dim]))
    mtf_outputs = mtf.layers.dense(
        mtf_inputs,
        new_dims=new_dim,
        reduced_dims=[channels_dim],
        activation=mtf.relu,
        use_bias=use_bias)
    mesh_impl = mtf.placement_mesh_impl.PlacementMeshImpl(
        shape=[], layout={}, devices=[""])
    lowering = mtf.Lowering(graph, {mesh: mesh_impl})
    actual_outputs = lowering.export_to_tf_tensor(mtf_outputs)

    expected_outputs = tf.keras.layers.Dense(units=units,
                                             activation=tf.nn.relu,
                                             use_bias=use_bias)(inputs)
    tf_group = lowering.copy_masters_to_slices()
    init = tf.global_variables_initializer()
    self.evaluate(init)
    self.evaluate(tf_group)
    actual, expected = self.evaluate([actual_outputs, expected_outputs])

    self.assertEqual(actual.shape, expected.shape)

  @test_util.run_in_graph_and_eager_modes()
  def testLayerNorm(self):
    batch = 2
    channels = 3
    inputs = tf.random_normal([batch, channels])

    graph = mtf.Graph()
    mesh = mtf.Mesh(graph, "my_mesh")
    batch_dim = mtf.Dimension("batch", batch)
    channels_dim = mtf.Dimension("channels", channels)

    mtf_inputs = mtf.import_tf_tensor(
        mesh, inputs, shape=mtf.Shape([batch_dim, channels_dim]))
    mtf_outputs = mtf.layers.layer_norm(mtf_inputs,
                                        dim=channels_dim)
    mesh_impl = mtf.placement_mesh_impl.PlacementMeshImpl(
        shape=[], layout={}, devices=[""])
    lowering = mtf.Lowering(graph, {mesh: mesh_impl})
    actual_outputs = lowering.export_to_tf_tensor(mtf_outputs)

    expected_outputs = tf.keras.layers.LayerNormalization()(inputs)
    tf_group = lowering.copy_masters_to_slices()
    init = tf.global_variables_initializer()
    self.evaluate(init)
    self.evaluate(tf_group)
    actual, expected = self.evaluate([actual_outputs, expected_outputs])

    self.assertEqual(actual.shape, expected.shape)

  @test_util.run_in_graph_and_eager_modes()
  def testBatchNorm(self):
    batch = 2
    channels = 3
    inputs = tf.constant([[0, 1, 2], [4, 5, 6]], dtype=np.float32)

    graph = mtf.Graph()
    mesh = mtf.Mesh(graph, "my_mesh")
    batch_dim = mtf.Dimension("batch", batch)
    channels_dim = mtf.Dimension("channels", channels)

    mtf_inputs = mtf.import_tf_tensor(
        mesh, inputs, shape=mtf.Shape([batch_dim, channels_dim]))

    mtf_outputs_0, _ = mtf.layers.batch_norm(
        mtf_inputs,
        is_training=True, momentum=0.95, epsilon=1e-6,
        dims_idx_start=0, dims_idx_end=1, name="bn0")
    mtf_outputs_1, _ = mtf.layers.batch_norm(
        mtf_outputs_0 * 2 + 1,
        is_training=True, momentum=0.95, epsilon=1e-6,
        dims_idx_start=0, dims_idx_end=1, name="bn1")

    mesh_impl = mtf.placement_mesh_impl.PlacementMeshImpl(
        shape=[], layout={}, devices=[""])
    lowering = mtf.Lowering(graph, {mesh: mesh_impl})

    actual_outputs_0 = lowering.export_to_tf_tensor(mtf_outputs_0)
    actual_outputs_1 = lowering.export_to_tf_tensor(mtf_outputs_1)

    tf_group = lowering.copy_masters_to_slices()
    init = tf.global_variables_initializer()
    self.evaluate(init)
    self.evaluate(tf_group)
    [actual_0, actual_1] = self.evaluate([actual_outputs_0, actual_outputs_1])

    expected = np.array([[-1, -1, -1], [1, 1, 1]])
    self.assertAllClose(actual_0, expected)
    self.assertAllClose(actual_1, expected)

  @test_util.run_in_graph_and_eager_modes()
  def testWeightsNonzero(self):
    inputs = tf.constant([[3, 1, 0], [1, 0, 0]])

    graph = mtf.Graph()
    mesh = mtf.Mesh(graph, "my_mesh")
    batch_dim = mtf.Dimension("batch", inputs.shape.as_list()[0])
    channels_dim = mtf.Dimension("channels", inputs.shape.as_list()[1])

    mtf_inputs = mtf.import_tf_tensor(
        mesh, inputs, shape=mtf.Shape([batch_dim, channels_dim]))
    mtf_outputs = mtf.layers.weights_nonzero(mtf_inputs)
    mesh_impl = mtf.placement_mesh_impl.PlacementMeshImpl(
        shape=[], layout={}, devices=[""])
    lowering = mtf.Lowering(graph, {mesh: mesh_impl})
    actual_outputs = lowering.export_to_tf_tensor(mtf_outputs)

    expected_outputs = tf.cast(tf.not_equal(inputs, 0), tf.float32)
    tf_group = lowering.copy_masters_to_slices()
    self.evaluate(tf_group)
    actual, expected = self.evaluate([actual_outputs, expected_outputs])

    self.assertAllEqual(actual, expected)

  @test_util.run_in_graph_and_eager_modes()
  def testDenseReluDense(self):
    batch = 2
    channels = 3
    hidden = 5
    inputs = tf.random_normal([batch, channels])

    graph = mtf.Graph()
    mesh = mtf.Mesh(graph, "my_mesh")
    batch_dim = mtf.Dimension("batch", batch)
    channels_dim = mtf.Dimension("channels", channels)
    hidden_dim = mtf.Dimension("hidden", hidden)

    mtf_inputs = mtf.import_tf_tensor(
        mesh, inputs, shape=mtf.Shape([batch_dim, channels_dim]))
    mtf_outputs = mtf.layers.dense_relu_dense(mtf_inputs,
                                              hidden_channels=hidden_dim)
    mesh_impl = mtf.placement_mesh_impl.PlacementMeshImpl(
        shape=[], layout={}, devices=[""])
    lowering = mtf.Lowering(graph, {mesh: mesh_impl})
    actual_outputs = lowering.export_to_tf_tensor(mtf_outputs)

    tf_group = lowering.copy_masters_to_slices()
    init = tf.global_variables_initializer()
    self.evaluate(init)
    self.evaluate(tf_group)
    actual = self.evaluate(actual_outputs)

    self.assertEqual(actual.shape, inputs.shape)

  @parameterized.parameters(
      (2, 16, 3, 4, 2, 2),
      (1, 8, 5, 3, 1, 4),
  )
  def testMaskedLocalAttention1D(self, batch, length, io_channels, kv_channels,
                                 heads, window_size):
    length_q = length
    query = tf.random_normal([batch, length_q, io_channels])

    graph = mtf.Graph()
    mesh = mtf.Mesh(graph, "my_mesh")
    batch_dim = mtf.Dimension("batch", batch)
    length_q_dim = mtf.Dimension("length_q", length_q)
    io_channels_dim = mtf.Dimension("io_channels", io_channels)
    kv_channels_dim = mtf.Dimension("kv_channels", kv_channels)
    heads_dim = mtf.Dimension("heads", heads)

    mtf_query = mtf.import_tf_tensor(
        mesh, query,
        shape=mtf.Shape([batch_dim, length_q_dim, io_channels_dim]))
    mtf_outputs = mtf.layers.masked_local_attention_1d(
        mtf_query,
        kv_channels=kv_channels_dim,
        heads=heads_dim,
        window_size=window_size)
    mesh_impl = mtf.placement_mesh_impl.PlacementMeshImpl(
        shape=[], layout={}, devices=[""])
    lowering = mtf.Lowering(graph, {mesh: mesh_impl})
    actual_outputs = lowering.export_to_tf_tensor(mtf_outputs)

    tf_group = lowering.copy_masters_to_slices()
    init = tf.global_variables_initializer()
    self.evaluate(init)
    self.evaluate(tf_group)
    actual = self.evaluate(actual_outputs)

    self.assertEqual(actual.shape, (batch, length_q, io_channels))

  @parameterized.parameters(
      (2, 4, 5, 7, 3, 1),
  )
  def testDotProductAttention(
      self, batch, heads, length_q, length_kv, depth_k, depth_v):
    query = tf.random_normal([batch, heads, length_q, depth_k])
    key = tf.random_normal([batch, heads, length_kv, depth_k])
    value = tf.random_normal([batch, heads, length_kv, depth_v])

    graph = mtf.Graph()
    mesh = mtf.Mesh(graph, "my_mesh")
    batch_dim = mtf.Dimension("batch", batch)
    heads_dim = mtf.Dimension("heads", heads)
    length_q_dim = mtf.Dimension("length_q", length_q)
    length_kv_dim = mtf.Dimension("length_kv", length_kv)
    depth_k_dim = mtf.Dimension("depth_k", depth_k)
    depth_v_dim = mtf.Dimension("depth_v", depth_v)

    mtf_query = mtf.import_tf_tensor(
        mesh, query,
        shape=mtf.Shape(
            [batch_dim, heads_dim, length_q_dim, depth_k_dim]))
    mtf_key = mtf.import_tf_tensor(
        mesh, key,
        shape=mtf.Shape(
            [batch_dim, heads_dim, length_kv_dim, depth_k_dim]))
    mtf_value = mtf.import_tf_tensor(
        mesh, value,
        shape=mtf.Shape(
            [batch_dim, heads_dim, length_kv_dim, depth_v_dim]))
    mtf_outputs = mtf.layers.dot_product_attention(
        mtf_query,
        mtf_key,
        mtf_value,
        mask=None)
    mesh_impl = mtf.placement_mesh_impl.PlacementMeshImpl(
        shape=[], layout={}, devices=[""])
    lowering = mtf.Lowering(graph, {mesh: mesh_impl})
    actual_outputs = lowering.export_to_tf_tensor(mtf_outputs)

    tf_group = lowering.copy_masters_to_slices()
    init = tf.global_variables_initializer()
    self.evaluate(init)
    self.evaluate(tf_group)
    actual = self.evaluate(actual_outputs)

    self.assertEqual(actual.shape, (batch, heads, length_q, depth_v))

  @parameterized.parameters(
      (16, 4),
      (32, 8),
  )
  def testMultiheadAttention(self, kv_channels, heads):
    batch = 2
    length = 8
    channels = 3
    query = tf.random_normal([batch, length, channels])

    graph = mtf.Graph()
    mesh = mtf.Mesh(graph, "my_mesh")
    batch_dim = mtf.Dimension("batch", batch)
    length_dim = mtf.Dimension("length", length)
    channels_dim = mtf.Dimension("channels", channels)
    kv_channels_dim = mtf.Dimension("kv_channels", kv_channels)
    heads_dim = mtf.Dimension("heads", heads)

    mtf_query = mtf.import_tf_tensor(
        mesh, query,
        shape=mtf.Shape([batch_dim, length_dim, channels_dim]))
    mtf_outputs = mtf.layers.multihead_attention(
        mtf_query,
        memory_antecedent=None,
        mask=None,
        kv_channels=kv_channels_dim,
        heads=heads_dim)
    mesh_impl = mtf.placement_mesh_impl.PlacementMeshImpl(
        shape=[], layout={}, devices=[""])
    lowering = mtf.Lowering(graph, {mesh: mesh_impl})
    actual_outputs = lowering.export_to_tf_tensor(mtf_outputs)

    tf_group = lowering.copy_masters_to_slices()
    init = tf.global_variables_initializer()
    self.evaluate(init)
    self.evaluate(tf_group)
    actual = self.evaluate(actual_outputs)

    self.assertEqual(actual.shape, query.shape)

  @parameterized.parameters(
      ("MAX_2D",), ("AVG_2D",), ("MAX_3D",), ("AVG_3D",),
  )
  def testPool(self, pooling_method):
    batch = 2
    depth = 3
    height = 4
    width = 6
    channels = 3
    tf.random.set_random_seed(1234)
    inputs = tf.random_normal([batch, depth, height, width, channels])

    stride_d = 3
    stride_h = 2
    stride_w = 3

    graph = mtf.Graph()
    mesh = mtf.Mesh(graph, "my_mesh")
    batch_dim = mtf.Dimension("batch", batch)
    depth_dim = mtf.Dimension("depth", depth)
    height_dim = mtf.Dimension("height", height)
    width_dim = mtf.Dimension("width", width)
    channels_dim = mtf.Dimension("channels", channels)

    mtf_inputs = mtf.import_tf_tensor(
        mesh, inputs, shape=mtf.Shape(
            [batch_dim, depth_dim, height_dim, width_dim, channels_dim]))

    if pooling_method == "MAX_2D":
      mtf_outputs = mtf.layers.max_pool2d(
          mtf_inputs, ksize=(stride_h, stride_w))
      inputs = tf.reshape(inputs, [batch * depth, height, width, channels])
      expected_outputs = tf.keras.layers.MaxPooling2D(
          (stride_h, stride_w))(inputs)
      expected_outputs = tf.reshape(
          expected_outputs,
          [batch, depth, int(height / stride_h),
           int(width / stride_w), channels])

    elif pooling_method == "AVG_2D":
      mtf_outputs = mtf.layers.avg_pool2d(
          mtf_inputs, ksize=(stride_h, stride_w))
      inputs = tf.reshape(inputs, [batch * depth, height, width, channels])
      expected_outputs = tf.keras.layers.AveragePooling2D(
          (stride_h, stride_w))(inputs)
      expected_outputs = tf.reshape(
          expected_outputs,
          [batch, depth, int(height / stride_h),
           int(width / stride_w), channels])

    elif pooling_method == "MAX_3D":
      mtf_outputs = mtf.layers.max_pool3d(
          mtf_inputs, ksize=[stride_d, stride_h, stride_w])
      expected_outputs = tf.keras.layers.MaxPooling3D(
          [stride_d, stride_h, stride_w])(inputs)

    elif pooling_method == "AVG_3D":
      mtf_outputs = mtf.layers.avg_pool3d(
          mtf_inputs, ksize=[stride_d, stride_h, stride_w])
      expected_outputs = tf.keras.layers.AveragePooling3D(
          [stride_d, stride_h, stride_w])(inputs)

    mtf_gradient = mtf.gradients([mtf_outputs], [mtf_inputs])[0]

    mesh_impl = mtf.placement_mesh_impl.PlacementMeshImpl(
        shape=[], layout={}, devices=[""])
    lowering = mtf.Lowering(graph, {mesh: mesh_impl})
    actual_outputs = lowering.export_to_tf_tensor(mtf_outputs)
    actual_gradient = lowering.export_to_tf_tensor(mtf_gradient)

    tf_group = lowering.copy_masters_to_slices()
    init = tf.global_variables_initializer()
    self.evaluate(init)
    self.evaluate(tf_group)
    actual, expected = self.evaluate([actual_outputs, expected_outputs])
    self.assertAllClose(actual, expected)

    actual = self.evaluate(actual_gradient)
    if pooling_method == "MAX_2D":
      expected_non_zeros = batch * depth * height * width * channels / (
          stride_h * stride_w)
      self.assertEqual(np.count_nonzero(actual), expected_non_zeros)

    elif pooling_method == "AVG_2D":
      expected = np.ones((batch, depth, height, width, channels),
                         dtype=np.float32) / stride_h / stride_w
      self.assertAllClose(actual, expected)

    elif pooling_method == "MAX_3D":
      expected_non_zeros = batch * depth * height * width * channels / (
          stride_d * stride_h * stride_w)
      self.assertEqual(np.count_nonzero(actual), expected_non_zeros)

    elif pooling_method == "AVG_3D":
      expected = np.ones((batch, depth, height, width, channels),
                         dtype=np.float32) / stride_d / stride_h / stride_w
      self.assertAllClose(actual, expected)

  @test_util.run_in_graph_and_eager_modes()
  def testConv1d(self):
    graph = mtf.Graph()
    mesh = mtf.Mesh(graph, "my_mesh")

    filter_size = 3
    depth_dim = mtf.Dimension("depth", 2)
    length_dim = mtf.Dimension("length", 4)
    output_dim = mtf.Dimension("output", 2)

    x = tf.constant([[1, 0], [0, 1], [1, 1], [2, 1]], dtype=tf.float32)
    mtf_x = mtf.import_tf_tensor(
        mesh, x, shape=mtf.Shape([length_dim, depth_dim]))

    initializer_mock = mock.MagicMock()
    initializer_mock.side_effect = initialize_by_shape({
        (1, 3, 2, 2): [[[[1, -1], [0, 0]], [[2, -2], [-1, 1]], [[3, -3],
                                                                [-2, 2]]]],
    })

    mtf_output = mtf.layers.conv1d(
        mtf_x,
        output_dim=output_dim,
        filter_size=filter_size,
        filter_initializer=initializer_mock)

    mesh_impl = mtf.placement_mesh_impl.PlacementMeshImpl(
        shape=[], layout={}, devices=[""])
    lowering = mtf.Lowering(graph, {mesh: mesh_impl})
    actual_output = lowering.export_to_tf_tensor(mtf_output)

    self.evaluate(tf.global_variables_initializer())
    self.evaluate(lowering.copy_masters_to_slices())
    actual = self.evaluate(actual_output)

    self.assertAllClose(actual, [[0, 0], [1, -1], [5, -5], [4, -4]])

  @mock.patch.object(tf, "truncated_normal_initializer", autospec=True)
  @test_util.run_in_graph_and_eager_modes()
  def testSeparableConv1d(self, random_normal_initializer_mock):
    graph = mtf.Graph()
    mesh = mtf.Mesh(graph, "my_mesh")

    depth_dim = mtf.Dimension("depth", 2)
    length_dim = mtf.Dimension("length", 4)
    output_dim = mtf.Dimension("output", 2)

    x = tf.constant([[1, 0], [0, 1], [1, 1], [2, 1]], dtype=tf.float32)
    mtf_x = mtf.import_tf_tensor(
        mesh, x, shape=mtf.Shape([length_dim, depth_dim]))

    initializer_mock = mock.MagicMock()
    random_normal_initializer_mock.return_value = initializer_mock
    initializer_mock.side_effect = initialize_by_shape({
        (2,): [1, 2],
        (2, 2): [[1, 0], [1, -1]],
    })

    mtf_output = mtf.layers.separable_conv1d(
        mtf_x,
        output_dim,
        min_relative_pos=-1,
        max_relative_pos=1,
        use_bias=True)

    mesh_impl = mtf.placement_mesh_impl.PlacementMeshImpl(
        shape=[], layout={}, devices=[""])
    lowering = mtf.Lowering(graph, {mesh: mesh_impl})
    actual_output = lowering.export_to_tf_tensor(mtf_output)

    self.evaluate(tf.global_variables_initializer())
    self.evaluate(lowering.copy_masters_to_slices())
    actual = self.evaluate(actual_output)

    self.assertAllClose(actual, [[3, -2], [6, -4], [9, -6], [7, -4]])


if __name__ == "__main__":
  tf.disable_v2_behavior()
  tf.enable_eager_execution()
  tf.test.main()
