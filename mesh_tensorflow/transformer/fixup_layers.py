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

"""Layers used for Fixup initialization.

See: https://arxiv.org/abs/1901.09321 for the paper.
"""

import math
import gin
import mesh_tensorflow as mtf
from mesh_tensorflow.transformer import attention
from mesh_tensorflow.transformer import transformer
from mesh_tensorflow.transformer import transformer_layers
import tensorflow.compat.v1 as tf


def get_single_scalar_bias(x, name: str):
  """Simple helper method to return a scalar bias.

  This is used as the `shift` in FixUp initialization and should be before
  every projection or convolution.

  Args:
    x: A mtf variable, used to know which mesh and dtype to use.
    name: The name of the bias.

  Returns:
    A (trainable) mtf Scalar.
  """
  single_dimension = mtf.Dimension("single_bias", 1)
  return mtf.get_variable(
      x.mesh,
      name,
      mtf.Shape([single_dimension]),
      initializer=tf.zeros_initializer(),
      dtype=x.dtype)


def dense_product_fixup(x,
                        reduced_dims,
                        new_dims,
                        kernel_initializer,
                        activation_functions=None,
                        name="dense_product",
                        **kwargs):
  """Wrapper around dense_product that is explicit about kernel initialization.

  Args:
    x: a Tensor
    reduced_dims: a list of Dimensions.
    new_dims: a list of Dimensions.
    kernel_initializer: The kernel initializer to use for the dense product. For
      fixup, this is the initializer scaled according to the number of encoder
      and decoder layers.
    activation_functions: a list of activation functions (or a singleton)
      Each can be a either: - a callable function from Tensor to Tensor - a
        string function name from namespace mtf) - None or "linear", meaning no
        activation function
    name: an optional string
    **kwargs: additional kwargs for mtf.layers.dense()

  Returns:
    Component wise product of dense layers with fixup init.
  """
  return mtf.layers.dense_product(
      x,
      reduced_dims,
      new_dims,
      activation_functions,
      name,
      kernel_initializer=kernel_initializer,
      **kwargs)


class AttentionParamsFixup(attention.AttentionParams):
  """Create attention parameters with Fixup initialization.

  See class docstring for DenseReluDenseFixup for details.

  For SelfAttention layer, m = 4, i.e., 4 weight matrix multiplications. See
  https://github.com/hongyi-zhang/Fixup/issues/8#issuecomment-505750941.
  So the scaling factor for SelfAttention layer is num_blocks**(-1/6).

  Attributes:
    mesh: a Mesh
    query_input_dim: a Dimension
    memory_input_dim: a Dimension
    output_dim: a Dimension
    key_dim: a Dimension
    value_dim: a Dimension
    query_heads_dims: a list of Dimension
    memory_heads_dims: a list of Dimension
    variable_dtype: a mtf.VariableDType
    shared_kv: a boolean
    fold_scaling_into_initializer: a boolean
    num_blocks: an integer specifying the number of TransformerLayer objects.
      For a vanilla Transformer model with 12 encoder layers and 12 decoder
      layers, the number of blocks is 2 * 12 + 3 * 12 = 60 where each encoder
      layer has 2 blocks, SelfAttention and Feedforward block and decoder
      additionally has the encoder-decoder attention block.
    o_init_fixup: a tf.initializer for the self.wo.
    init_fixup: a tf.initializer for the self.wq, self.wk, self.wv and self.wkv.
  """

  def __init__(
      self,
      mesh,
      query_input_dim,
      memory_input_dim,
      output_dim,
      key_dim,
      value_dim,
      query_heads_dims,
      memory_heads_dims,
      variable_dtype,
      shared_kv=False,
      fold_scaling_into_initializer=False,
      num_blocks=None,
      default_init="he",
      init_distribution="uniform",
      **kwargs):

    self.num_blocks = num_blocks
    self.default_init = default_init
    self.init_distribution = init_distribution

    if mtf.layers.unit_scaling_convention():
      raise ValueError(
          "Fixup initialization is not compatible with unit scaling convention."
      )

    if fold_scaling_into_initializer:
      raise ValueError("Fixup initialization is not compatible with "
                       "`fold_scaling_into_initializer.")

    super(AttentionParamsFixup, self).__init__(
        mesh,
        query_input_dim,
        memory_input_dim,
        output_dim,
        key_dim,
        value_dim,
        query_heads_dims,
        memory_heads_dims,
        variable_dtype,
        shared_kv=shared_kv,
        fold_scaling_into_initializer=fold_scaling_into_initializer,
        **kwargs)

  def init_weights(self):
    o_init_fixup = tf.initializers.zeros()

    # Since tf.initializers.variance_scaling returns sqrt(3 * scale / n), (note
    # that scale is inside sqrt), we need to square the scale factor. Hence the
    # exponent is -1/3 instead of -1/6 as described in the class docstring.
    if self.default_init == "glorot":
      init_fixup = tf.initializers.variance_scaling(
          mode="fan_avg",
          distribution=self.init_distribution,
          scale=math.pow(self.num_blocks, -1. / 3))
    elif self.default_init == "he":
      init_fixup = tf.initializers.variance_scaling(
          mode="fan_in",
          distribution=self.init_distribution,
          scale=2 * math.pow(self.num_blocks, -1. / 3))
    else:
      raise ValueError(
          ("Unsupported default initialization. Only 'glorot' and 'he'"
           " initializations are supported."))

    if not self.no_query:
      self.wq = mtf.get_variable(
          self.mesh,
          "q_fixup",
          self.q_shape,
          initializer=init_fixup,
          dtype=self.variable_dtype)

    if self.shared_kv:
      self.wkv = mtf.get_variable(
          self.mesh,
          "kv_fixup",
          self.k_shape,
          initializer=init_fixup,
          dtype=self.variable_dtype)
    else:
      self.wk = mtf.get_variable(
          self.mesh,
          "k_fixup",
          self.k_shape,
          initializer=init_fixup,
          dtype=self.variable_dtype)

      self.wv = mtf.get_variable(
          self.mesh,
          "v_fixup",
          self.v_shape,
          initializer=init_fixup,
          dtype=self.variable_dtype)

    self.wo = mtf.get_variable(
        self.mesh,
        "o_fixup",
        self.o_shape,
        initializer=o_init_fixup,
        dtype=self.variable_dtype)


@gin.configurable
class DenseReluDenseFixup(transformer.TransformerLayer):
  """Two dense layers with ReLU or other activation on hidden layer.

  Implements weights initialization in https://arxiv.org/abs/1901.09321.

  tf.initializers.variance_scaling from Uniform(-limit, limit) where limit =
  sqrt(3 * scale / fan).

  Using scale = 2 and fan = fan_in makes it He initializer from
  https://arxiv.org/abs/1502.01852

  Using scale = 1 and fan = fan_avg makes it Glorot initializer from
  http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf

  Fixup initialization multiplies an extra scaling factor to these standard
  initializers. In general, this factor is num_blocks^{1/(2m-2)} where num
  blocks is the number of TransformerLayer objects and m is the number of matrix
  multiplication. This is equal to `L` in https://arxiv.org/abs/1901.09321.

  For DenseReluDense layer, m = 2 (up-projection and
  down-projection) and the extra factor is 1/sqrt(num_blocks).In order to use
  tf.initializers.variance_scaling for Fixup initialization, we need to set the
  scale and mode arguments properly.

  For He initializer, we want to sample from Uniform(-limit_fixup, limit_fixup)
  where limit_fixup = sqrt(3 * 2 / fan_in) * 1/sqrt(num_blocks) = sqrt(3 * (2 /
  num_blocks) / fan_in). In other words, the scale = 2 / num_blocks with mode =
  fan_in.

  For Glorot initializer, we want to sample from Uniform(-limit_fixup,
  limit_fixup) where limit_fixup = sqrt(3 / fan_avg) * 1/sqrt(num_blocks) =
  sqrt(3 * (1 / num_blocks) / fan_avg). In other words, the scale = 1 /
  num_blocks with mode = fan_avg.

  Note that these settings apply equally for both truncated normal and uniform
  distributions from which we sample the weights.

  Attributes:
    hidden_size: an integer - size of the hidden layer
    dropout_rate: a floating-point number
    activation: an activation function or a list of activation functions. see
      documentation for mtf.layers.dense_product()
    use_bias: a boolean, whether to use bias in the dense layers.
    num_blocks: an integer specifying the number of TransformerLayer objects.
      For a vanilla Transformer model with 12 encoder layers and 12 decoder
      layers, the number of blocks is 2 * 12 + 3 * 12 = 60 where each encoder
      layer has 2 blocks, SelfAttention and Feedforward block and decoder
      additionally has the encoder-decoder attention block.
    downproject_initializer: a tf.initializer for d_model to d_ff projection.
    upproject_initializer: a tf.initializer for d_ff to d_model.
  """

  def __init__(
      self,
      hidden_size=4096,
      dropout_rate=0.0,
      activation="relu",
      use_bias=False,
      default_init="he",
      init_distribution="uniform",
      num_blocks=gin.REQUIRED):

    self.hidden_size = hidden_size
    self.dropout_rate = dropout_rate
    self.activation = activation
    self.use_bias = use_bias

    self.downproject_initializer = tf.initializers.zeros()
    if default_init == "glorot":
      self.upproject_initializer = tf.initializers.variance_scaling(
          mode="fan_avg",
          distribution=init_distribution,
          scale=1.0 / num_blocks)
    elif default_init == "he":
      self.upproject_initializer = tf.initializers.variance_scaling(
          mode="fan_in", distribution=init_distribution, scale=2.0 / num_blocks)
    else:
      raise ValueError(
          "Unsupported default initialization. Only 'glorot' and 'he'"
          " initializations are supported.")

  def call(self, context, x, losses=None):
    """Call the layer."""
    io_channels = x.shape.dims[-1]
    hidden_channels = mtf.Dimension("d_ff", self.hidden_size)

    h = dense_product_fixup(
        x,
        reduced_dims=x.shape.dims[-1:],
        new_dims=hidden_channels,
        activation_functions=self.activation,
        use_bias=self.use_bias,
        variable_dtype=context.variable_dtype,
        name="wi",
        kernel_initializer=self.upproject_initializer,
        expert_dims=context.model.ensemble_dims)
    if context.train and self.dropout_rate != 0.0:
      h = mtf.dropout(
          h, 1.0 - self.dropout_rate, noise_shape=h.shape - context.length_dim)
    shift = get_single_scalar_bias(x, "shift")
    h_res = mtf.add(h, shift)
    h = mtf.reshape(h_res, h.shape)
    return mtf.layers.dense(
        h,
        io_channels,
        use_bias=self.use_bias,
        activation=None,
        variable_dtype=context.variable_dtype,
        reduced_dims=h.shape.dims[-1:],
        name="wo",
        expert_dims=context.model.ensemble_dims,
        kernel_initializer=self.downproject_initializer)


@gin.configurable
class SelfAttentionFixup(transformer_layers.SelfAttention):
  """Multi-head self-attention layer with the Fixup initialization."""

  def __init__(self,
               num_blocks=gin.REQUIRED,
               default_init="glorot",
               init_distribution="uniform",
               **kwargs):
    # Any arg in `kwargs` should be defined in SelfAttention constructor.
    super(SelfAttentionFixup, self).__init__(**kwargs)
    self.num_blocks = num_blocks
    self.default_init = default_init
    self.init_distribution = init_distribution

  def make_params(self, context):
    if self.num_heads == 1:
      query_heads_dims = None
      memory_heads_dims = None
    elif self.num_memory_heads == 0:
      query_heads_dims = [mtf.Dimension("heads", self.num_heads)]
      memory_heads_dims = query_heads_dims
    elif self.num_memory_heads == 1:
      query_heads_dims = [mtf.Dimension("heads", self.num_heads)]
      memory_heads_dims = None
    else:
      if self.num_heads % self.num_memory_heads != 0:
        raise ValueError("num_memory_heads must divide num_heads")
      memory_heads_dims = [mtf.Dimension("heads", self.num_memory_heads)]
      query_heads_dims = memory_heads_dims + [
          mtf.Dimension("query_heads", self.num_heads // self.num_memory_heads)
      ]

    return AttentionParamsFixup(
        context.mesh,
        query_input_dim=context.model.model_dim,
        memory_input_dim=context.model.model_dim,
        output_dim=context.model.model_dim,
        key_dim=self.kv_dim,
        value_dim=self.kv_dim,
        query_heads_dims=query_heads_dims,
        memory_heads_dims=memory_heads_dims,
        variable_dtype=context.variable_dtype,
        shared_kv=self.shared_kv,
        ensemble_dim=context.model.ensemble_dim,
        combine_dims=self.combine_dims,
        keep_query_heads_dims=self.keep_query_heads_dims,
        fold_scaling_into_initializer=self.fold_scaling_into_initializer,
        num_blocks=self.num_blocks,
        default_init=self.default_init,
        init_distribution=self.init_distribution)


@gin.configurable
class EncDecAttentionFixup(transformer_layers.EncDecAttention):
  """Multi-head attention over encoder output with Fixup initialization."""

  def __init__(self, relative_attention_type=None, **kwargs):
    super(EncDecAttentionFixup, self).__init__(
        relative_attention_type=relative_attention_type, **kwargs)

  def _get_memory_antecedent(self, context):
    return context.encoder_output

  def call(self, context, x, losses=None):
    """Call the layer."""
    return transformer_layers.enc_dec_attention(
        self, self._get_memory_antecedent(context), context, x, losses)


@gin.configurable
def sublayer_fixup_scale(x, layer_stack, context):
  """Multiply by single one-initialized scalar."""
  del layer_stack
  dim = mtf.Dimension("single_scale", 1)
  fixup_weight = mtf.get_variable(
      x.mesh, "fixup_scale_weight", shape=mtf.Shape([dim]),
      dtype=context.variable_dtype,
      initializer=tf.constant_initializer(1.))
  return mtf.reshape(x * fixup_weight, x.shape)


@gin.configurable
def sublayer_fixup_shift(x, layer_stack, context):
  """Shift by single zero-initialized scalar."""
  del layer_stack
  dim = mtf.Dimension("single_bias", 1)
  fixup_bias = mtf.get_variable(
      x.mesh, "fixup_bias", shape=mtf.Shape([dim]),
      dtype=context.variable_dtype,
      initializer=tf.zeros_initializer())
  res = mtf.add(x, fixup_bias)
  res = mtf.reshape(res, x.shape)
  return res
