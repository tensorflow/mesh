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

"""Tests for mesh_tensorflow.transformer.funnel_transformer."""

from absl.testing import parameterized
import mesh_tensorflow as mtf
from mesh_tensorflow import test_utils
from mesh_tensorflow.transformer import funnel_transformer
from mesh_tensorflow.transformer import transformer
from mesh_tensorflow.transformer import transformer_layers
import numpy as np
import tensorflow.compat.v1 as tf


def create_dummy_model(mesh,
                       shapes,
                       n_blocks=2,
                       block_param_size_str="2_2",
                       block_repeat_size_str="1_1"):
  """Creates a dummy model and layer stack with 4-dimensional input."""

  assert len(shapes) == 4
  outer_batch_size, batch_size, length, d_model = shapes
  batch_dim = mtf.Dimension("batch", batch_size)
  outer_batch_dim = mtf.Dimension("outer_batch", outer_batch_size)
  length_dim = mtf.Dimension("length", length)
  block_param_size = list(map(int, block_param_size_str.split("_")))
  block_repeat_size = list(map(int, block_repeat_size_str.split("_")))

  sublayers_initial = [
      transformer.sublayer_dropout,
  ]
  sublayers_per_layer = [
      transformer.sublayer_rms_norm,
      transformer.sublayer_call_layer,
      transformer.sublayer_dropout,
      transformer.sublayer_residual,
  ]
  sublayers_final = [
      transformer.sublayer_rms_norm,
      transformer.sublayer_dropout,
  ]
  submodules = [
      transformer_layers.SelfAttention(),
      transformer_layers.DenseReluDense()
  ]

  n_sublayers = np.array(block_param_size).prod()
  layers = submodules * n_sublayers
  layer_stack = funnel_transformer.FunnelTransformerLayerStack(
      layers=layers,
      n_blocks=n_blocks,
      block_param_size=block_param_size,
      block_repeat_size=block_repeat_size,
      sublayers_initial=sublayers_initial,
      sublayers_per_layer=sublayers_per_layer,
      sublayers_final=sublayers_final)

  model = transformer.Unitransformer(
      input_vocab_size=10,
      output_vocab_size=10,
      autoregressive=False,
      max_length=8,
      d_model=d_model,
      layer_stack=layer_stack)

  context = transformer.Context(
      model=model,
      mesh=mesh,
      batch_dims=[batch_dim, outer_batch_dim],
      length_dim=length_dim,
      variable_dtype=mtf.VariableDType(tf.float32),
      sequence_id=mtf.ones(mesh, mtf.Shape([length_dim])),
      position=mtf.range(mesh, length_dim, dtype=tf.int32)
  )
  return layer_stack, context


class FunnelTransformerTest(tf.test.TestCase, parameterized.TestCase):

  def setUp(self):
    super(FunnelTransformerTest, self).setUp()
    self.converter = test_utils.NumpyConverter()
    self.default_dim_names = ["outer_batch", "batch", "length", "d_model"]

  def test_layer_stack_call_padding_handling(self):
    self.converter = test_utils.NumpyConverter()
    x = np.random.randn(2, 3, 4, 5)
    layer_stack, context = create_dummy_model(
        self.converter.mesh, shapes=x.shape)

    # The last two sequence positions are padding.
    x[:, :, -2:, :] *= 0
    sequence_id = np.ones_like(x, dtype=np.int32)
    sequence_id[:, :, -2:, :] *= 0
    context.sequence_id = self.converter.convert_np_array_to_mtf_tensor(
        sequence_id, dim_names=self.default_dim_names)

    x_mtf = self.converter.convert_np_array_to_mtf_tensor(
        x, dim_names=self.default_dim_names, dtype=tf.float32)
    output_mtf = layer_stack.call(context, x_mtf)
    # [2, 3, 4, 5] -> [2, 3, 2, 5]
    actual = self.converter.convert_mtf_tensor_to_np_array(output_mtf)

    # After pooling, the last sequence position should be padding, i.e., zeros.
    last_position = actual[:, :, -1, :]
    self.assertAllClose(last_position, np.zeros_like(last_position))

  def test_layer_stack_call_pooled_length(self):
    converter = test_utils.NumpyConverter()
    x = np.random.randn(2, 3, 4, 5)
    layer_stack, context = create_dummy_model(
        converter.mesh, shapes=x.shape)
    x_mtf = converter.convert_np_array_to_mtf_tensor(
        x, dim_names=self.default_dim_names, dtype=tf.float32)
    output_mtf = layer_stack.call(context, x_mtf)
    actual = converter.convert_mtf_tensor_to_np_array(output_mtf)
    self.assertAllEqual(actual.shape, (2, 3, 2, 5))

  def test_layer_stack_call_num_output_layers(self):
    x = np.random.randn(2, 3, 4, 5)
    layer_stack, context = create_dummy_model(
        self.converter.mesh, shapes=x.shape)
    x_mtf = self.converter.convert_np_array_to_mtf_tensor(
        x, dim_names=self.default_dim_names, dtype=tf.float32)
    _ = layer_stack.call(context, x_mtf)
    # +1 accounts for the sublayers_initial. sublayer_final is merged with the
    # last layer of sublayers_per_layer.
    self.assertLen(context.layer_outputs, len(layer_stack.layers) + 1)

  def test_layer_stack_call_num_unique_layers(self):
    x = np.random.randn(2, 3, 4, 5)
    layer_stack, context = create_dummy_model(
        self.converter.mesh, shapes=x.shape)
    x_mtf = self.converter.convert_np_array_to_mtf_tensor(
        x, dim_names=self.default_dim_names, dtype=tf.float32)
    output_mtf = layer_stack.call(context, x_mtf)
    lowering, _ = self.converter.convert_mtf_tensor_to_tf_tensor(output_mtf)

    # Test the number of unique layers.
    all_vars = lowering.graph.all_variables
    self_attn_vars = [
        var.name for var in all_vars if "SelfAttention" in var.name
    ]

    # We expect total of `n_layers` of SelfAttention and DenseReluDense layers.
    n_layers = len(layer_stack.layers)

    # We expect n_sublayers` SelfAttention.
    n_sublayers = n_layers // 2

    # Each self attn layer has 4 variables: wq, wk, wv, wo.
    self.assertEqual(len(self_attn_vars) // 4, n_sublayers)

  def test_layer_stack_update_context_sequence_id(self):
    x = np.random.randn(2, 3, 4, 5)
    layer_stack, context = create_dummy_model(
        self.converter.mesh, shapes=x.shape)
    x_mtf = self.converter.convert_np_array_to_mtf_tensor(
        x, dim_names=self.default_dim_names, dtype=tf.float32)
    _ = layer_stack.call(context, x_mtf)
    self.assertEqual(2, context.length_dim.size)

  def test_layer_stack_update_context_position(self):
    x = np.random.randn(2, 3, 4, 5)
    layer_stack, context = create_dummy_model(
        self.converter.mesh, shapes=x.shape)
    x_mtf = self.converter.convert_np_array_to_mtf_tensor(
        x, dim_names=self.default_dim_names, dtype=tf.float32)
    _ = layer_stack.call(context, x_mtf)
    actual = self.converter.convert_mtf_tensor_to_np_array(context.position)
    self.assertAllEqual(np.arange(2), actual)


if __name__ == "__main__":
  tf.test.main()
