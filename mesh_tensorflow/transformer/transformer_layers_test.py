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

"""Tests for mesh_tensorflow.transformer.transformer_layers."""

import collections
import mesh_tensorflow as mtf
from mesh_tensorflow import test_utils
from mesh_tensorflow.transformer import transformer
from mesh_tensorflow.transformer import transformer_layers
import mock
import numpy as np
import tensorflow.compat.v1 as tf


def get_dummy_decoder_context(converter,
                              batch=2,
                              d_model=6,
                              length=4,
                              mode="incremental",
                              initial_position=None,
                              state=None,
                              inputs=None):

  batch_dim = mtf.Dimension("batch", batch)
  length_dim = mtf.Dimension("length", length)

  # Set up a dummy model
  layer_stack = transformer.LayerStack(layers=[])
  model = transformer.Unitransformer(
      d_model=d_model,
      input_vocab_size=10,  # dummy values
      output_vocab_size=10,  # dummy values
      autoregressive=True,
      max_length=length,
      layer_stack=layer_stack)

  if state is not None:
    state_mtf = converter.convert_np_array_to_mtf_tensor(
        state, dtype=tf.float32, dim_names=["batch", "length", "d_model"])
    states = [state_mtf]
  else:
    states = None

  if initial_position:
    initial_position = mtf.constant(
        converter.mesh,
        initial_position,
        shape=mtf.Shape([batch_dim]),
        dtype=tf.int32)

  if inputs is not None:
    inputs = converter.convert_np_array_to_mtf_tensor(
        inputs, dim_names=["batch", "length"])

  context = transformer.Context(
      model=model,
      mode=mode,
      states=states,
      new_states=[],
      mesh=converter.mesh,
      batch_dims=[batch_dim],
      length_dim=length_dim,
      variable_dtype=mtf.VariableDType(tf.float32),
      sequence_id=1,
      inputs=inputs,
      initial_position=initial_position)
  return context


class TransformerLayersTest(tf.test.TestCase):

  def setUp(self):
    super(TransformerLayersTest, self).setUp()
    self.converter = test_utils.NumpyConverter()

  def test_conv1d_call_same_input_output_dims(self):
    batch = 2
    d_model = 6
    length = 3
    inputs = np.random.randint(0, 10, size=[batch, length])
    inputs_mtf = self.converter.convert_np_array_to_mtf_tensor(
        inputs, dim_names=["batch", "length"])
    # Dummy context with necessary information for Conv1DLayer.call
    Context = collections.namedtuple("Context",
                                     ["inputs", "activation_dtype", "mode"])
    context = Context(
        inputs=inputs_mtf, activation_dtype=tf.float32, mode="train")
    x = np.random.randn(batch, length, d_model)
    x_mtf = self.converter.convert_np_array_to_mtf_tensor(
        x, dtype=tf.float32, dim_names=["batch", "length", "d_model"])
    conv_layer = transformer_layers.Conv1DLayer(
        filter_size=3, output_size=d_model)
    output_mtf = conv_layer.call(context, x_mtf)
    self.assertAllEqual([batch, length, d_model],
                        output_mtf.shape.to_integer_list)

  def test_conv1d_record_states_first_part_mode(self):
    batch = 2
    d_model = 6
    length = 6
    filter_size = 3

    inputs = np.random.randint(1, 10, size=[batch, length])
    context = get_dummy_decoder_context(
        self.converter,
        batch=batch,
        d_model=d_model,
        initial_position=2,  # indices 0 and 1 correspond to partial sequences.
        inputs=inputs,
        mode="first_part")

    x = np.zeros(shape=(batch, length, d_model))
    x_mtf = self.converter.convert_np_array_to_mtf_tensor(
        x, dtype=tf.float32, dim_names=["batch", "length", "d_model"])

    conv_layer = transformer_layers.Conv1D()
    conv_layer.record_states_first_part_mode(context, x_mtf, filter_size)
    actual = self.converter.convert_mtf_tensor_to_np_array(
        context.new_states[0])
    expected = np.zeros(shape=[batch, filter_size, d_model])

    self.assertAllClose(actual, expected)

  def test_conv1d_record_states_first_part_mode_with_partial_sequence(self):
    batch = 2
    d_model = 6
    length = 6
    filter_size = 3

    inputs = np.random.randint(1, 10, size=[batch, length])
    context = get_dummy_decoder_context(
        self.converter,
        batch=batch,
        d_model=d_model,
        initial_position=2,  # indices 0 and 1 correspond to partial sequences.
        inputs=inputs,
        mode="first_part")

    x = np.random.randn(batch, length, d_model)
    x_mtf = self.converter.convert_np_array_to_mtf_tensor(
        x, dtype=tf.float32, dim_names=["batch", "length", "d_model"])

    conv_layer = transformer_layers.Conv1D()
    conv_layer.record_states_first_part_mode(context, x_mtf, filter_size)
    actual = self.converter.convert_mtf_tensor_to_np_array(
        context.new_states[0])
    expected = np.zeros(shape=[batch, filter_size, d_model])
    expected[:, -2, :] = x[:, 0, :]
    expected[:, -1, :] = x[:, 1, :]

    self.assertAllClose(actual, expected)

  def test_conv1d_record_states_incremental_mode(self):
    batch = 2
    d_model = 6
    filter_size = 3

    state = np.random.randn(batch, filter_size, d_model)
    context = get_dummy_decoder_context(
        self.converter,
        batch=batch,
        d_model=d_model,
        state=state)

    x = np.random.randn(batch, d_model)
    x_mtf = self.converter.convert_np_array_to_mtf_tensor(
        x, dtype=tf.float32, dim_names=["batch", "d_model"])
    conv_layer = transformer_layers.Conv1D()
    _ = conv_layer.record_states_incremental_mode(context, x_mtf,
                                                  filter_size)
    actual = self.converter.convert_mtf_tensor_to_np_array(
        context.new_states[0])

    # [batch, 2, d_model], [batch, 1, d_model] -> [batch, 3, d_model]
    expected = np.concatenate([state[:, 1:, :], x[:, np.newaxis, :]], axis=1)
    self.assertAllClose(actual, expected)

  def test_conv1d_update_state(self):
    batch = 2
    d_model = 6
    filter_size = 3
    batch_dim = mtf.Dimension("batch", batch)
    filter_dim = mtf.Dimension("filter", filter_size)

    x = np.random.randn(batch, d_model)
    x_mtf = self.converter.convert_np_array_to_mtf_tensor(
        x, dtype=tf.float32, dim_names=["batch", "d_model"])

    old_state = np.random.randn(batch, filter_size, d_model)
    old_state_mtf = self.converter.convert_np_array_to_mtf_tensor(
        old_state, dtype=tf.float32, dim_names=["batch", "filter", "d_model"])

    position_mtf = mtf.constant(
        self.converter.mesh,
        filter_size - 1,
        shape=mtf.Shape([batch_dim]),
        dtype=tf.int32)
    conv_layer = transformer_layers.Conv1D()
    output_mtf = conv_layer.update_state(
        old_state_mtf, x_mtf, position_mtf, filter_dim, dtype=tf.float32)
    actual = self.converter.convert_mtf_tensor_to_np_array(output_mtf)

    expected = np.empty(shape=old_state.shape)
    expected[:, :filter_size - 1, :] = old_state[:, 1:, :]
    expected[:, -1, :] = x
    self.assertAllClose(actual, expected)

  def test_separable_conv1d_call_same_input_output_dims(self):
    batch = 2
    d_model = 6
    length = 3
    inputs = np.random.randint(0, 10, size=[batch, length])
    inputs_mtf = self.converter.convert_np_array_to_mtf_tensor(
        inputs, dim_names=["batch", "length"])
    # Dummy context with necessary information for Conv1DLayer.call
    Context = collections.namedtuple("Context",
                                     ["inputs", "activation_dtype", "mode"])
    context = Context(
        inputs=inputs_mtf, activation_dtype=tf.float32, mode="train")
    x = np.random.randn(batch, length, d_model)
    x_mtf = self.converter.convert_np_array_to_mtf_tensor(
        x, dtype=tf.float32, dim_names=["batch", "length", "d_model"])

    min_relative_pos = -1
    max_relative_pos = 2
    conv_layer = transformer_layers.SeparableConv1DLayer(
        min_relative_pos=min_relative_pos,
        max_relative_pos=max_relative_pos,
        output_size=d_model)

    output_mtf = conv_layer.call(context, x_mtf)
    self.assertAllEqual([batch, length, d_model],
                        output_mtf.shape.to_integer_list)

  def test_conv1d_call_incremental_mode(self):
    batch = 2
    d_model = 6
    length = 4
    filter_size = 3
    output_size = 2

    state = np.random.randn(batch, filter_size, d_model)
    context = get_dummy_decoder_context(
        self.converter,
        batch=batch,
        d_model=d_model,
        length=length,
        state=state)

    x = np.random.randn(batch, d_model)
    x_mtf = self.converter.convert_np_array_to_mtf_tensor(
        x, dtype=tf.float32, dim_names=["batch", "d_model"])

    conv_filter = np.random.randn(1, filter_size, d_model, output_size)

    def mock_initializer():
      # pylint: disable=unused-argument
      def conv_init(shape, dtype, **unused_kwargs):
        return conv_filter
      return conv_init

    with mock.patch.object(tf, "glorot_uniform_initializer", mock_initializer):
      conv_layer = transformer_layers.Conv1DLayer(
          filter_size=filter_size, output_size=output_size)
      output_mtf = conv_layer.call(context, x_mtf)
    actual = self.converter.convert_mtf_tensor_to_np_array(output_mtf)

    # [batch, 2, d_model], [batch, 1, d_model] -> [batch, 3, d_model]
    padded_x = np.concatenate([state[:, 1:, :], x[:, np.newaxis, :]], axis=1)
    # b: batch h: fake height, l: length (or filter), d: d_model, o: output_size
    expected = np.einsum("bld,hldo->bo", padded_x, conv_filter)
    self.assertAllClose(actual, expected)

  def test_separable_conv1d_layer_incremental_mode(self):
    batch = 2
    d_model = 6
    length = 4
    filter_size = 3
    output_size = 2

    state = np.random.randn(batch, filter_size, d_model)
    context = get_dummy_decoder_context(
        self.converter,
        batch=batch,
        d_model=d_model,
        length=length,
        state=state)

    x = np.random.randn(batch, d_model)
    x_mtf = self.converter.convert_np_array_to_mtf_tensor(
        x, dtype=tf.float32, dim_names=["batch", "d_model"])

    max_relative_pos = 0
    min_relative_pos = max_relative_pos - filter_size + 1
    conv_layer = transformer_layers.SeparableConv1DLayer(
        min_relative_pos=min_relative_pos,
        max_relative_pos=max_relative_pos,
        output_size=output_size)

    # Non-standard implementation of depthwise convolution in the
    #   transformer_layers.py requires somewhat complicated testing.
    # A list of weights (length filter_size) each of shape [model_dim], which is
    #   the depth dimension. So the total number of parameters is filter_size *
    #   model_dim as expected for depthwise convolution.
    all_kernel_wts = [np.random.randn(d_model) for _ in range(filter_size)]
    all_kernel_wts_mtf = [
        self.converter.convert_np_array_to_mtf_tensor(
            w, dtype=tf.float32, dim_names=["d_model"]) for w in all_kernel_wts
    ]
    pointwise_weight = np.random.randn(d_model, output_size)
    pointwise_weight_mtf = self.converter.convert_np_array_to_mtf_tensor(
        pointwise_weight, dtype=tf.float32, dim_names=["d_model", "d_model"])
    with mock.patch.object(mtf.layers,
                           "get_dense_kernel_weights") as mock_weights:
      mock_weights.return_value = pointwise_weight_mtf
      output_mtf = conv_layer.call(
          context, x_mtf, all_kernel_wts=all_kernel_wts_mtf)
    actual = self.converter.convert_mtf_tensor_to_np_array(output_mtf)

    # [filter_size, d_model]
    conv_filter = np.array(all_kernel_wts)
    # [batch, filter_size, d_model]
    padded_x = np.concatenate([state[:, 1:, :], x[:, np.newaxis, :]], axis=1)
    # b: batch, l: length (or filter), d: d_model
    depthwise_convolved = np.einsum("bld,ld->bd", padded_x, conv_filter)
    # The pointwise convolution can be implemented with matrix multiplication.
    # [batch, d_model] * [d_model, output_size] -> [batch, output_size]
    expected = np.dot(depthwise_convolved, pointwise_weight)
    self.assertAllClose(actual, expected)


if __name__ == "__main__":
  tf.test.main()
