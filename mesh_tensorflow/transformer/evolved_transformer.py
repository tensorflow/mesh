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
"""Custom layers for the evolved transformer.

See https://arxiv.org/abs/1901.11117 for more details.
"""

import gin
import mesh_tensorflow as mtf
from mesh_tensorflow.transformer import transformer
from mesh_tensorflow.transformer import transformer_layers

import tensorflow.compat.v1 as tf


@gin.configurable
class GatedLinearUnitLayer(transformer.TransformerLayer):
  """Layer performing a Gated Linear Unit transformation on its input.

  See https://arxiv.org/pdf/1612.08083.pdf.
  """

  def call(self, context, x, losses=None):
    """Call the layer."""
    return mtf.layers.dense_product(
        x,
        reduced_dims=x.shape.dims[-1:],
        new_dims=x.shape.dims[-1:],
        activation_functions=["linear", "sigmoid"],
        variable_dtype=context.variable_dtype,
        name="glu",
        expert_dims=context.model.ensemble_dims)


@gin.configurable
class EncoderConvolutionalLayer(transformer.TransformerLayer):
  """The convolutional layers custom to the evolved transformer encoder.

  The input is projected to 4 times the model dimension size followed by a ReLU
  on the left branch while it goes through a 3x1 convolution on the right
  branch. The outputs of the branches are summed and then passed through a layer
  norm. The output of the layer norm then goes through separable 9x1
  convolution.
  """

  def __init__(self,
               d_model,
               dropout_rate,
               initializer_scale=1.0,
               norm_epsilon=1e-6):
    """Create an EncoderConvolutionalLayer.

    Args:
      d_model: a positive integer, the dimension of the model dim.
      dropout_rate: a float between 0 and 1.
      initializer_scale: a positive float, the scale for the initializers of the
        separable convolutional filters.
      norm_epsilon: a small positive float, the epsilon for the layer norm.
    """
    self._dropout_rate = dropout_rate
    self._norm_epsilon = norm_epsilon
    self._conv3x1 = transformer_layers.Conv1DLayer(
        filter_size=3, output_size=int(d_model / 2), activation="relu")
    self._sep_conv9x1 = transformer_layers.SeparableConv1DLayer(
        min_relative_pos=-4,
        max_relative_pos=4,
        output_size=int(d_model / 2),
        depthwise_filter_initializer_scale=initializer_scale,
        pointwise_filter_initializer_scale=initializer_scale)

  def call(self, context, x, losses=None):
    """Call the layer."""
    model_dim = context.model.model_dim

    # Note that the left output dim can also be thought of the hidden dimension
    # in the feedforward network.
    with tf.variable_scope("conv1x1"):
      hidden_dim = mtf.Dimension(model_dim.name, 4 * model_dim.size)
      left_state = mtf.layers.dense(
          x,
          reduced_dims=x.shape.dims[-1:],
          new_dims=[hidden_dim],
          activation=mtf.relu,
          use_bias=False,
          variable_dtype=context.variable_dtype)
      left_state = _dropout(left_state, context, self._dropout_rate)

    with tf.variable_scope("conv3x1"):
      right_state = self._conv3x1.call(context, x, losses)
      right_state = _dropout(right_state, context, self._dropout_rate)
      right_state = _pad_channels_dim(right_state, hidden_dim.size)

    hidden_state = left_state + right_state
    hidden_state = mtf.layers.layer_norm(
        hidden_state,
        epsilon=self._norm_epsilon,
        dim=hidden_state.shape.dims[-1])

    with tf.variable_scope("sep_conv9x1"):
      output = self._sep_conv9x1.call(context, hidden_state, losses)
      output = _dropout(output, context, self._dropout_rate)
      return _pad_channels_dim(output, model_dim.size)


@gin.configurable
class DecoderAttentionLayer(transformer.TransformerLayer):
  """The attention layers custom to the evolved transformer decoder.

  This layer consists of applying both a self attention and enc-dec attention to
  the input and then summing their outputs.
  """

  def __init__(self, base_num_heads):
    """Create an DecoderAttentionLayer.

    Args:
      base_num_heads: a positive integer, the base number of heads the attention
        layers are using.
    """
    self._self_attention = transformer_layers.SelfAttention(num_heads=2 *
                                                            base_num_heads)
    self._enc_dec_attention = transformer_layers.EncDecAttention(
        num_heads=base_num_heads)

  def call(self, context, x, losses=None):
    """Call the layer."""
    with tf.variable_scope("self_attention"):
      left_state = self._self_attention.call(context, x, losses)
    with tf.variable_scope("enc_dec_attention"):
      right_state = self._enc_dec_attention.call(context, x, losses)
    return left_state + right_state


@gin.configurable
class DecoderConvolutionalLayer(transformer.TransformerLayer):
  """The convolutional layers custom to the evolved transformer decoder.

  The input is passed through a 11x1 separable convolution followed by a ReLU on
  the left branch while it goes through a 7x1 separable convolution on the right
  branch. The outputs of the branches are summed and then passed through a layer
  norm. The output of the layer norm then goes through separable 7x1
  convolution.
  """

  def __init__(self,
               d_model,
               dropout_rate,
               initializer_scale=1.0,
               norm_epsilon=1e-6):
    """Create an DecoderConvolutionalLayer.

    Args:
      d_model: a positive integer, the dimension of the model dim.
      dropout_rate: a float between 0 and 1.
      initializer_scale: a positive float, the scale for the initializers of the
        separable convolutional filters.
      norm_epsilon: a small positive float, the epsilon for the layer norm.
    """
    self._d_model = d_model
    self._dropout_rate = dropout_rate
    self._norm_epsilon = norm_epsilon
    self._sep_conv11x1 = transformer_layers.SeparableConv1DLayer(
        min_relative_pos=-10,
        max_relative_pos=0,
        output_size=int(2 * d_model),
        depthwise_filter_initializer_scale=initializer_scale,
        pointwise_filter_initializer_scale=initializer_scale,
        activation="relu")
    self._sep_conv7x1_1 = transformer_layers.SeparableConv1DLayer(
        min_relative_pos=-6,
        max_relative_pos=0,
        output_size=int(d_model / 2),
        depthwise_filter_initializer_scale=initializer_scale,
        pointwise_filter_initializer_scale=initializer_scale)
    self._sep_conv7x1_2 = transformer_layers.SeparableConv1DLayer(
        min_relative_pos=-6,
        max_relative_pos=0,
        output_size=d_model,
        depthwise_filter_initializer_scale=initializer_scale,
        pointwise_filter_initializer_scale=initializer_scale)

  def call(self, context, x, losses=None):
    """Call the layer."""
    with tf.variable_scope("sep_conv11x1"):
      left_state = self._sep_conv11x1.call(context, x, losses)
      left_state = _dropout(left_state, context, self._dropout_rate)

    with tf.variable_scope("sep_conv7x1_1"):
      right_state = self._sep_conv7x1_1.call(context, x, losses)
      right_state = _dropout(right_state, context, self._dropout_rate)
      right_state = _pad_channels_dim(right_state,
                                      left_state.shape.dims[-1].size)

    hidden_state = left_state + right_state
    hidden_state = mtf.layers.layer_norm(
        hidden_state,
        epsilon=self._norm_epsilon,
        dim=hidden_state.shape.dims[-1])

    with tf.variable_scope("sep_conv7x1_2"):
      output = self._sep_conv7x1_2.call(context, hidden_state, losses)
      return _dropout(output, context, self._dropout_rate)


def _pad_channels_dim(tensor, size):
  channels_dim = tensor.shape.dims[-1]
  if channels_dim.size > size:
    raise ValueError("Cannot pad to size of {} when the original size "
                     "of {} is bigger".format(size, channels_dim.size))
  elif channels_dim.size == size:
    return tensor
  else:
    return mtf.pad(tensor, [0, size - channels_dim.size], channels_dim.name)


def _dropout(x, context, dropout_rate):
  if context.train and dropout_rate > 0:
    return mtf.dropout(
        x,
        rate=dropout_rate,
        noise_shape=mtf.Shape(context.batch_dims + x.shape.dims[-1:]))
  else:
    return x
