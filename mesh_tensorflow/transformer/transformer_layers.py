# coding=utf-8
# Copyright 2018 The Mesh TensorFlow Authors.
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

"""Layers for the Transformer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mesh_tensorflow as mtf
from mesh_tensorflow.transformer import transformer

import tensorflow as tf


class DenseReluDense(transformer.TransformerLayer):
  """Two fully-connected layers with feed-forward activation."""

  def __init__(self, hidden_size=2048, dropout_rate=0.0):
    """Create a DenseReluDense.

    Args:
      hidden_size: an integer - size of the hidden layer
      dropout_rate: a floating-point number
    """
    self.hidden_size = hidden_size
    self.dropout_rate = 0.0

  def call(self, context, x, losses=None):
    """Call the layer."""
    io_channels = x.shape.dims[-1]
    hidden_channels = mtf.Dimension("d_ff", self.hidden_size)
    h = mtf.layers.dense(x, hidden_channels,
                         use_bias=False, activation=mtf.relu,
                         variable_dtype=context.variable_dtype,
                         name="wi")
    if context.train and self.dropout_rate != 0.0:
      h = mtf.dropout(h, 1.0 - self.dropout_rate,
                      noise_shape=h.shape - context.length_dim)
    return mtf.layers.dense(h, io_channels, use_bias=False, activation=None,
                            variable_dtype=context.variable_dtype,
                            name="wo")


class SelfAttention(transformer.TransformerLayer):
  """Multi-head self-attention layer."""

  def __init__(self,
               num_heads=8,
               key_value_size=128,
               dropout_rate=0.0):
    """Create a SelfAttention Layer.

    Args:
      num_heads: an integer
      key_value_size: an integer
      dropout_rate: a floating-point number
    """
    self.num_heads = num_heads
    self.key_value_size = key_value_size
    self.dropout_rate = dropout_rate

  def call(self, context, x, losses=None):
    """Call the layer."""
    wq, wk, wv, wo = mtf.layers.multihead_attention_params(
        context.mesh, self.heads_dim, context.model_dim, self.kv_dim,
        context.variable_dtype)
    memory_length = mtf.Dimension("memory_length", context.length_dim.size)
    q = mtf.einsum([x, wq], reduced_dims=[context.model_dim])
    if context.mode == "incremental":
      m = x
    else:
      m = mtf.rename_dimension(x, context.length_dim.name, "memory_length")
    k = mtf.einsum([m, wk], reduced_dims=[context.model_dim])
    v = mtf.einsum([m, wv], reduced_dims=[context.model_dim])
    if context.mode == "incremental":
      old_k, old_v = context.get_states(2)
      one_hot = mtf.one_hot(
          context.position, memory_length, dtype=context.activation_dtype)
      inv_one_hot = 1.0 - one_hot
      k = old_k * inv_one_hot + k * one_hot
      v = old_v * inv_one_hot + v * one_hot
    if context.mode == "incremental" or context.mode == "first_part":
      context.record_new_states([k, v])
    masks = []
    if context.autoregressive:
      masks.append(mtf.cast(
          mtf.less(
              context.position,
              mtf.range(context.mesh, memory_length, dtype=tf.int32)),
          context.activation_dtype) * -1e9)
    if (context.sequence_id is not None and
        isinstance(context.sequence_id, mtf.Tensor) and
        context.length_dim in context.sequence_id.shape):
      masks.append(mtf.cast(
          mtf.not_equal(
              context.sequence_id,
              mtf.layers.rename_length_to_memory_length(
                  context.sequence_id)),
          context.activation_dtype) * -1e9)
    mask = mtf.add_n(masks) if masks else None

    o = mtf.layers.dot_product_attention_v2(
        q, k, v,
        memory_length,
        self.kv_dim,
        self.kv_dim,
        mask,
        self.dropout_rate if context.train else 0.0,
        [context.length_dim])
    return mtf.einsum([o, wo], x.shape, reduced_dims=[
        self.heads_dim, self.kv_dim])

  @property
  def heads_dim(self):
    return mtf.Dimension("heads", self.num_heads)

  @property
  def kv_dim(self):
    return mtf.Dimension("d_kv", self.key_value_size)


class EncDecAttention(transformer.TransformerLayer):
  """Multi-head self-attention layer."""

  def __init__(self,
               num_heads=8,
               key_value_size=128,
               dropout_rate=0.0):
    """Create a EncDecAttention Layer.

    Args:
      num_heads: an integer
      key_value_size: an integer
      dropout_rate: a floating-point number
    """
    self.num_heads = num_heads
    self.key_value_size = key_value_size
    self.dropout_rate = dropout_rate

  def call(self, context, x, losses=None):
    """Call the layer."""
    memory_input_dim = context.encoder_output.shape[-1]
    if memory_input_dim != context.model_dim:
      raise NotImplementedError(
          "TODO(noam): support different model_dim in encoder and decoder.")
    wq, wk, wv, wo = mtf.layers.multihead_attention_params(
        context.mesh, self.heads_dim, context.model_dim, self.kv_dim,
        context.variable_dtype)
    q = mtf.einsum([x, wq], reduced_dims=[context.model_dim])
    if context.mode == "incremental":
      k, v, memory_length = context.get_constant_state()
    else:
      m = context.encoder_output
      memory_length, = [d for d in m.shape.dims if d.name == "memory_length"]
      k = mtf.einsum([m, wk], reduced_dims=[context.model_dim])
      v = mtf.einsum([m, wv], reduced_dims=[context.model_dim])
      if context.mode == "first_part":
        context.record_constant_state((k, v, memory_length))
    if context.encoder_sequence_id and context.sequence_id:
      mask = mtf.cast(
          mtf.not_equal(
              context.sequence_id, context.encoder_sequence_id),
          context.activation_dtype) * -1e9
    else:
      mask = None
    o = mtf.layers.dot_product_attention_v2(
        q, k, v,
        memory_length,
        self.kv_dim,
        self.kv_dim,
        mask,
        dropout=self.dropout_rate if context.train else 0.0,
        dropout_broadcast_dims=[context.length_dim])
    return mtf.einsum([o, wo], x.shape, reduced_dims=[
        self.heads_dim, self.kv_dim])

  @property
  def heads_dim(self):
    return mtf.Dimension("heads", self.num_heads)

  @property
  def kv_dim(self):
    return mtf.Dimension("d_kv", self.key_value_size)


class LocalSelfAttention(transformer.TransformerLayer):
  """Multi-head local self-attention layer."""

  def __init__(self,
               num_heads=8,
               key_value_size=128,
               window_size=128):
    """Create a LocalSelfAttention Layer.

    Args:
      num_heads: an integer
      key_value_size: an integer
      window_size: an integer
    """
    self.num_heads = num_heads
    self.key_value_size = key_value_size
    self.window_size = window_size

  def call(self, context, x, losses=None):
    """Call the layer."""
    params = mtf.layers.multihead_attention_params(
        context.mesh, self.heads_dim, context.model_dim, self.kv_dim,
        context.variable_dtype)
    if context.mode == "incremental":
      prev_k, prev_v = context.get_states(2)
      y, new_k, new_v = mtf.layers.masked_local_attention_1d_incremental(
          x, prev_k, prev_v, context.position, params=params)
      context.record_new_states([new_k, new_v])
      return y
    else:
      kv = []
      y = mtf.layers.masked_local_attention_1d(
          x, self.kv_dim,
          self.heads_dim,
          self.window_size,
          params=params,
          return_kv=kv)
      if context.mode == "first_part":
        k = kv[0]
        v = kv[1]
        window_dim = mtf.Dimension("window", self.window_size)
        mesh = k.mesh
        window_pos = mtf.range(mesh, window_dim, tf.int32)
        pos = mtf.range(mesh, context.length_dim, tf.int32)
        select_recent = mtf.cast(
            mtf.equal(window_pos, mtf.mod(pos, self.window_size)), k.dtype)
        select_recent *= mtf.cast(
            mtf.less(pos, context.initial_position), k.dtype)
        select_recent *= mtf.cast(
            mtf.greater_equal(
                pos, context.initial_position - self.window_size), k.dtype)
        state_shape = k.shape.dims[:-2] + [window_dim, self.kv_dim]
        k_state = mtf.einsum(
            [k, select_recent], output_shape=state_shape,
            reduced_dims=[context.length_dim])
        v_state = mtf.einsum(
            [v, select_recent], output_shape=state_shape,
            reduced_dims=[context.length_dim])
        context.new_states.extend([k_state, v_state])
      return y

  @property
  def heads_dim(self):
    return mtf.Dimension("heads", self.num_heads)

  @property
  def kv_dim(self):
    return mtf.Dimension("d_kv", self.key_value_size)
