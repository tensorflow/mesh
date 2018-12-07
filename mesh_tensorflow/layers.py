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

"""Layers implemented in Mesh TensorFlow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from mesh_tensorflow import ops_with_redefined_builtins as mtf

import tensorflow as tf


def dense(x, output_dim, reduced_dims=None, expert_dims=None,
          use_bias=True, activation=None,
          master_dtype=tf.float32,
          slice_dtype=tf.float32,
          variable_dtype=None,
          name=None):
  """Dense layer doing (kernel*x + bias) computation.

  Args:
    x: a mtf.Tensor of shape [..., reduced_dims].
    output_dim: a mtf.Dimension
    reduced_dims: an optional list of mtf.Dimensions of x to be reduced. If
      omitted, we reduce the last dimension.
    expert_dims: an optional list of mtf.Dimension which represent different
      experts. Different experts get different weights.
    use_bias: a boolean, whether to add bias.
    activation: an optional function from mtf.Tensor to mtf.Tensor
    master_dtype: a tf.dtype (deprecated - use variable_dtype)
    slice_dtype: a tf.dtype (deprecated - use variable_dtype)
    variable_dtype: a mtf.VariableDType
    name: a string. variable scope.

  Returns:
    a mtf.Tensor of shape [..., output_dim].
  """
  if variable_dtype is None:
    variable_dtype = mtf.VariableDType(master_dtype, slice_dtype, x.dtype)
  if expert_dims is None:
    expert_dims = []
  if reduced_dims is None:
    reduced_dims = x.shape.dims[-1:]
  w_shape = mtf.Shape(expert_dims + reduced_dims + [output_dim])
  output_shape = mtf.Shape(
      [d for d in x.shape.dims if d not in reduced_dims] + [output_dim])

  with tf.variable_scope(name, default_name="dense"):
    stddev = mtf.list_product(d.size for d in reduced_dims) ** -0.5
    w = mtf.get_variable(
        x.mesh,
        "kernel",
        w_shape,
        initializer=tf.random_normal_initializer(stddev=stddev),
        dtype=variable_dtype)
    w = mtf.cast(w, x.dtype)
    y = mtf.einsum([x, w], output_shape)
    if use_bias:
      b = mtf.get_variable(
          x.mesh,
          "bias",
          mtf.Shape(expert_dims + [output_dim]),
          initializer=tf.zeros_initializer(),
          dtype=variable_dtype)
      y += b
    if activation is not None:
      y = activation(y)
    return y


def layer_norm(x, dim, epsilon=1e-6, name="layer_prepostprocess"):
  """Layer normalization over dimension dim.

  Args:
    x: a mtf.Tensor whose shape contains dim.
    dim: a mtf.Dimension
    epsilon: a floating point number
    name: a string. variable scope.

  Returns:
    a mtf.Tensor with same shape as x.
  """
  with tf.variable_scope(name + "/layer_norm"):
    scale = mtf.get_variable(
        x.mesh,
        "layer_norm_scale",
        mtf.Shape([dim]),
        initializer=tf.ones_initializer(),
        activation_dtype=x.dtype)
    bias = mtf.get_variable(
        x.mesh,
        "layer_norm_bias",
        mtf.Shape([dim]),
        initializer=tf.zeros_initializer(),
        activation_dtype=x.dtype)
    reduced_shape = x.shape - dim
    mean = mtf.reduce_mean(x, output_shape=reduced_shape)
    variance = mtf.reduce_mean(mtf.square(x - mean), output_shape=reduced_shape)
    norm_x = (x - mean) * mtf.rsqrt(variance + epsilon)
    return norm_x * scale + bias


def batch_norm(x, is_training, momentum, epsilon=1e-9,
               init_zero=False, name=None):
  """Batch normalization.

  Args:
    x: a mtf.Tensor whose shape contains [batch_dim, ..., dim]
    is_training: a boolean, whether mode is training.
    momentum: a floating point number, specifying batch norm decay value.
    epsilon: a floating point number.
    init_zero: a boolean, whether to initialize scale with 0's or 1's.
    name: a string. variable scope.

  Returns:
    a mtf.Tensor with same shape as x.
  """
  with tf.variable_scope(name, default_name="batch_norm", values=[x]):
    if init_zero:
      gamma_initializer = tf.zeros_initializer()
    else:
      gamma_initializer = tf.ones_initializer()

    norm_dim = x.shape.dims[0:3]
    reduced_shape = x.shape - norm_dim

    scale = mtf.get_variable(
        x.mesh,
        "batch_norm_scale",
        reduced_shape,
        initializer=gamma_initializer,
        activation_dtype=x.dtype)
    bias = mtf.get_variable(
        x.mesh,
        "batch_norm_bias",
        reduced_shape,
        initializer=tf.zeros_initializer(),
        activation_dtype=x.dtype)

    moving_mean = mtf.get_variable(
        x.mesh, "moving_mean", reduced_shape,
        initializer=tf.random_normal_initializer(stddev=1.0),
        activation_dtype=x.dtype,
        trainable=False)
    moving_variance = mtf.get_variable(
        x.mesh, "moving_variance",
        reduced_shape, initializer=tf.ones_initializer(),
        activation_dtype=x.dtype,
        trainable=False)

    # At training time, calculate mean and variance and normalize across batch
    # dim.
    if is_training:
      mean = mtf.reduce_mean(x, output_shape=reduced_shape)
      variance = mtf.reduce_mean(
          mtf.square(x - mean), output_shape=reduced_shape)

      norm_x = (x - mean) * mtf.rsqrt(variance + epsilon)

      # Update running mean and running variance.
      moving_mean = mtf.assign(
          moving_mean, momentum * moving_mean + (1-momentum) * mean)
      moving_variance = mtf.assign(
          moving_variance,
          momentum * moving_variance + (1 - momentum) * variance)
    else:
      # At eval and test time, use the running mean and variance.
      norm_x = (x - moving_mean) * mtf.rsqrt(moving_variance + epsilon)

    return (norm_x * scale) + bias


def softmax_cross_entropy_with_logits(logits, targets, vocab_dim, z_loss=0.0):
  """Per-example softmax loss.

  if z_loss is nonzero, we add a loss equal to z_loss*log(z)^2, where z is the
  partition function.  Example value: z_loss=1e-4.  Two uses of z_loss are:
  - To keep the logits from drifting too far from zero, which can cause
     unacceptable roundoff errors in bfloat16.
  - To encourage the logits to be normalized log-probabilities.

  Args:
    logits: a mtf.Tensor whose shape contains vocab_dim
    targets: a mtf.Tensor with the same shape as logits
    vocab_dim: a mtf.Dimension
    z_loss: a float

  Returns:
    a mtf.Tensor whose shape is equal to logits.shape - vocab_dim

  Raises:
    ValueError: if the shapes do not match.
  """
  if logits.shape != targets.shape:
    raise ValueError(
        "logits shape must equal targets shape"
        "logits=%s targets=%s" % (logits.to_string, targets.to_string))
  if vocab_dim not in logits.shape.dims:
    raise ValueError("vocab_dim must be in logits.shape.dims")
  log_z = mtf.reduce_logsumexp(logits, vocab_dim)
  log_softmax = logits - log_z
  loss = mtf.negative(
      mtf.reduce_sum(log_softmax * targets, reduced_dim=vocab_dim))
  if z_loss != 0:
    loss += z_loss * mtf.square(log_z)
  return loss


def sigmoid_cross_entropy_with_logits(logits, targets):
  """Sigmoid cross-entropy loss.

  Args:
    logits: a mtf.Tensor
    targets: a mtf.Tensor with the same shape as logits

  Returns:
    a mtf.Tensor whose shape is equal to logits.shape

  Raises:
    ValueError: if the shapes do not match.
  """
  if logits.shape != targets.shape:
    raise ValueError(
        "logits shape must equal targets shape"
        "logits=%s targets=%s" % (logits.to_string, targets.to_string))
  x = logits
  z = targets
  return mtf.relu(x) - x * z + mtf.log(1 + mtf.exp(-mtf.abs(x)))


def weights_nonzero(targets, dtype=tf.float32):
  def my_fn(x):
    return tf.cast(tf.not_equal(x, 0), dtype)
  return mtf.cwise(my_fn, [targets], output_dtype=dtype, name="weights_nonzero")


def dense_relu_dense(x,
                     hidden_channels,
                     dropout=0.0,
                     dropout_broadcast_dims=None,
                     master_dtype=tf.float32,
                     slice_dtype=tf.float32, name=None):
  """Hidden layer with ReLU activation followed by linear projection.

  The output has the same number of channels as the input.

  Args:
    x: a mtf.Tensor
    hidden_channels: a mtf.Dimension - channels in the hidden layer
    dropout: an optional float
    dropout_broadcast_dims: an optional list of mtf.Dimension
    master_dtype: a tf.dtype
    slice_dtype: a tf.dtype
    name: an optional string

  Returns:
    a mtf.Tensor with the same shape as x.
  """
  with tf.variable_scope(name, default_name="dense_relu_dense"):
    io_channels = x.shape.dims[-1]
    h = dense(x, hidden_channels,
              use_bias=False, activation=mtf.relu,
              master_dtype=master_dtype, slice_dtype=slice_dtype, name="wi")
    if dropout != 0.0:
      h = mtf.dropout(h, 1.0 - dropout,
                      noise_shape=h.shape - dropout_broadcast_dims)
    return dense(h, io_channels, use_bias=False, activation=None,
                 master_dtype=master_dtype, slice_dtype=slice_dtype,
                 name="wo")


def local_1d_halo_exchange(k, v, num_w_blocks, w_dim, mask_right):
  """Halo exchange for keys and values for Local 1D attention."""
  if num_w_blocks is not None:
    if mask_right:
      k = mtf.left_halo_exchange(k, num_w_blocks, w_dim, w_dim.size)
      v = mtf.left_halo_exchange(v, num_w_blocks, w_dim, w_dim.size)
    else:
      k = mtf.halo_exchange(k, num_w_blocks, w_dim, w_dim.size)
      v = mtf.halo_exchange(v, num_w_blocks, w_dim, w_dim.size)
  else:
    if mask_right:
      k = mtf.pad(k, [w_dim, None], w_dim.name)
      v = mtf.pad(v, [w_dim, None], w_dim.name)
    else:
      k = mtf.pad(k, [w_dim, w_dim], w_dim.name)
      v = mtf.pad(v, [w_dim, w_dim], w_dim.name)
  return k, v


def local_self_attention_spatial_blocks(
    query_antecedent,
    kv_channels,
    heads,
    memory_w_dim=None,
    mask_right=False,
    master_dtype=tf.float32,
    slice_dtype=tf.float32,
    name=None):
  """Attention to the source position and a neighborhood to the left or right.

  The sequence is divided into blocks of length block_size.
  Attention for a given query position can only see memory positions
  less than or equal to the query position, in the corresponding block
  and the previous block.

  Args:
    query_antecedent: a mtf.Tensor with shape
      [batch, num_h_blocks, num_w_blocks, h_dim, w_dim, io_channels]
      must have the same size as query_length, but a different name.
    kv_channels: a mtf.Dimension (the size of the key and value vectors)
    heads: a mtf.Dimension (the number of heads)
    memory_w_dim: mtf Dimension, for the memory width block.
    mask_right: bool, flag specifying whether we mask out attention to the right
      for the decoder.
    master_dtype: a tf.dtype
    slice_dtype: a tf.dtype
    name: an optional string.

  Returns:
    a Tensor of shape
        [batch, num_h_blocks, num_w_blocks, h_dim, w_dim, io_channels]

  Raises:
    ValueError: if channels or depth don't match.
  """
  with tf.variable_scope(
      name, default_name="multihead_attention",
      values=[query_antecedent]):

    w_dim, io_channels = query_antecedent.shape.dims[-2:]
    batch, num_w_blocks = query_antecedent.shape.dims[:2]
    wq, wk, wv, wo = multihead_attention_vars(
        query_antecedent.mesh, heads, io_channels, kv_channels,
        master_dtype, slice_dtype, query_antecedent.dtype)

    # Rename dimensions for the memory height and width.
    memory_antecedent = mtf.rename_dimension(
        query_antecedent, w_dim.name, "memory_" + w_dim.name)
    memory_w_dim = memory_antecedent.shape.dims[-2]

    # Call einsum over the query and memory to get query q, keys k and values v.
    q = mtf.einsum(
        [query_antecedent, wq],
        mtf.Shape([batch, heads, num_w_blocks, w_dim, kv_channels]))
    k = mtf.einsum(
        [memory_antecedent, wk],
        mtf.Shape([batch, heads, num_w_blocks, memory_w_dim, kv_channels]))
    v = mtf.einsum(
        [memory_antecedent, wv],
        mtf.Shape([batch, heads, num_w_blocks, memory_w_dim, kv_channels]))

    # Halo exchange for memory blocks.
    k, v = local_1d_halo_exchange(k, v, num_w_blocks, memory_w_dim, mask_right)

    # Calculate the causal mask to avoid peeking into the future. We compute
    # this once and reuse it for all blocks since the block_size is known.
    mask = None
    if mask_right:
      mask = attention_bias_local_block(
          query_antecedent.mesh, w_dim, memory_w_dim)

    output = dot_product_attention(q, k, v, mask=mask)

    return mtf.einsum(
        [output, wo], mtf.Shape([batch, num_w_blocks, w_dim, io_channels]))


def masked_local_attention_1d(x,
                              kv_channels,
                              heads,
                              window_size=128,
                              master_dtype=tf.float32,
                              slice_dtype=tf.float32,
                              length_per_split=None,
                              return_kv=None,
                              params=None,
                              name=None):
  """Attention to the source position and a neighborhood to the left of it.

  Attention for a given query position p can only see memory positions
  in the range (p - window_size, p].

  Args:
    x: a mtf.Tensor with shape batch_dims + [length, io_channels]
    kv_channels: a mtf.Dimension (the size of the key and value vectors)
    heads: a mtf.Dimension (the number of heads)
    window_size: an integer
    master_dtype: a tf.dtype (deprecated - use params arg)
    slice_dtype: a tf.dtype (deprecated - use params arg)
    length_per_split: an optional integer indicating the part of the length
      dimension per processor.  You can omit if the length dimension is not
      split.
    return_kv: an optional list onto which to append the computed k and v.
    params: an optional quadruple of Tensors (see multihead_attention_params())
    name: an optional string.

  Returns:
    a Tensor with the same shape as x

  Raises:
    ValueError: if channels or depth don't match.
  """
  with tf.variable_scope(
      name, default_name="masked_local_attention_1d", values=[x]):

    batch_dims = x.shape.dims[:-2]
    length, io_channels = x.shape.dims[-2:]
    if params is None:
      wq, wk, wv, wo = multihead_attention_vars(
          x.mesh, heads, io_channels, kv_channels,
          master_dtype, slice_dtype, x.dtype)
    else:
      wq, wk, wv, wo = params

    # Get query q, keys k and values v.
    qkv_shape = mtf.Shape(batch_dims + [heads, length, kv_channels])
    q = mtf.einsum([x, wq], qkv_shape)
    k = mtf.einsum([x, wk], qkv_shape)
    v = mtf.einsum([x, wv], qkv_shape)
    if return_kv is not None:
      return_kv.extend([k, v])

    # Choose a suitable block size.
    # We choose the greatest divisor of length_per_split less than or equal
    # to max(window_size, 128)
    if length_per_split is None:
      length_per_split = length.size
    block_length = max(window_size, 128)
    while length_per_split % block_length != 0:
      block_length -= 1

    query_block_length = mtf.Dimension("query_block_length", block_length)
    memory_block_length = mtf.Dimension("memory_block_length", block_length)
    # The num_blocks dimension gets the same name as the length dimension,
    # so it will be split in the same way.
    num_blocks = mtf.Dimension(length.name, length.size // block_length)
    q_shape = batch_dims + [heads, num_blocks, query_block_length, kv_channels]
    kv_shape = batch_dims + [
        heads, num_blocks, memory_block_length, kv_channels]
    q = mtf.reshape(q, q_shape)
    k = mtf.reshape(k, kv_shape)
    v = mtf.reshape(v, kv_shape)
    # augment the keys and values for each block with keys and values for
    # the previous window_size timesteps.
    k = mtf.left_halo_exchange(k, num_blocks, memory_block_length, window_size)
    v = mtf.left_halo_exchange(v, num_blocks, memory_block_length, window_size)
    padded_memory_block_length = mtf.Dimension(
        "memory_block_length", window_size + block_length)
    mpos = mtf.range(x.mesh, padded_memory_block_length, tf.float32)
    qpos = mtf.range(x.mesh, query_block_length, tf.float32) + window_size
    # prevent looking forward
    mask = mtf.cast(mtf.greater(mpos, qpos), x.dtype) * -1e9
    # prevent looking >=block_length timesteps backward
    mask += mtf.cast(mtf.less_equal(mpos, qpos - block_length), x.dtype) * -1e9
    # Note: The first window_size-1 positions can see back into pre-time
    # where all the keys and values are zero.  We could mask this out, but we
    # don't.
    o = dot_product_attention(q, k, v, mask=mask)
    o = mtf.reshape(o, batch_dims + [heads, length, kv_channels])
    return mtf.einsum([o, wo], mtf.Shape(batch_dims + [length, io_channels]))


def masked_local_attention_1d_incremental(x,
                                          prev_k,
                                          prev_v,
                                          step_num,
                                          master_dtype=None,
                                          slice_dtype=None,
                                          params=None,
                                          name=None):
  """Incremental local self-attention (one decode step).

  Incremental version of masked_local_attention_1d()

  Args:
    x: a mtf.Tensor with shape [batch..., io_channels]
    prev_k: mtf.Tensor with shape
       [batch..., heads, window_length, kv_channels]
    prev_v: mtf.Tensor with shape
       [batch..., heads, window_length, kv_channels]
    step_num: mtf Scalar with dtype tf.int32
    master_dtype: a tf.dtype (deprecated)
    slice_dtype: a tf.dtype (deprecated)
    params: a quadruple of Tensors (see multihead_attention_params())
    name: an optional string.

  Returns:
    y: A mtf.Tensor with shape [batch..., io_channels]
    new_k: mtf.Tensor with shape
       [batch..., heads, window_length, kv_channels]
    new_v: mtf.Tensor with shape
       [batch..., heads, window_length, kv_channels]

  Raises:
    ValueError: if the dimensions do not match.
  """
  batch_dims = x.shape.dims[:-1]
  io_channels = x.shape.dims[-1]
  heads, window_length, kv_channels = prev_k.shape.dims[-3:]
  with tf.variable_scope(name, default_name="masked_local_attention_1d"):
    if params is None:
      wq, wk, wv, wo = multihead_attention_vars(
          x.mesh, heads, io_channels, kv_channels,
          master_dtype, slice_dtype, x.dtype)
    else:
      wq, wk, wv, wo = params
    q = mtf.einsum([x, wq], mtf.Shape(batch_dims + [heads, kv_channels]))
    k = mtf.einsum([x, wk], mtf.Shape(batch_dims + [heads, kv_channels]))
    v = mtf.einsum([x, wv], mtf.Shape(batch_dims + [heads, kv_channels]))
    current_position = mtf.equal(
        mtf.range(x.mesh, window_length, dtype=tf.int32),
        mtf.mod(step_num, window_length.size))
    k = mtf.where(current_position, k, prev_k, output_shape=prev_k.shape)
    v = mtf.where(current_position, v, prev_v, output_shape=prev_v.shape)
    o = dot_product_attention(q, k, v, mask=None)
    y = mtf.einsum([o, wo], x.shape)
    return y, k, v


def local_2d_halo_exchange(k, v, num_h_blocks, h_dim,
                           num_w_blocks, w_dim, mask_right):
  """Halo exchange for keys and values for Local 2D attention."""
  for blocks_dim, block_size_dim, halo_size in [
      (num_h_blocks, h_dim, h_dim.size),
      (num_w_blocks, w_dim, w_dim.size)]:
    # shape of k is [num_h_blocks, num_w_blocks, h_dim, w_dim, kv_channels]
    if halo_size > 0:
      if blocks_dim is not None:
        if mask_right:
          k = mtf.left_halo_exchange(k, blocks_dim, block_size_dim, halo_size)
          v = mtf.left_halo_exchange(v, blocks_dim, block_size_dim, halo_size)
        else:
          k = mtf.halo_exchange(k, blocks_dim, block_size_dim, halo_size)
          v = mtf.halo_exchange(v, blocks_dim, block_size_dim, halo_size)
      else:
        if mask_right:
          k = mtf.pad(k, [halo_size, None], block_size_dim.name)
          v = mtf.pad(v, [halo_size, None], block_size_dim.name)
        else:
          k = mtf.pad(k, [halo_size, halo_size], block_size_dim.name)
          v = mtf.pad(v, [halo_size, halo_size], block_size_dim.name)
  return k, v


def local_2d_self_attention_spatial_blocks(query_antecedent,
                                           kv_channels,
                                           heads,
                                           memory_h_dim=None,
                                           memory_w_dim=None,
                                           mask_right=False,
                                           master_dtype=tf.float32,
                                           slice_dtype=tf.float32,
                                           name=None):
  """Attention to the source position and a neighborhood to the left or right.

  The sequence is divided into blocks of length block_size.
  Attention for a given query position can only see memory positions
  less than or equal to the query position, in the corresponding block
  and the previous block.

  Args:
    query_antecedent: a mtf.Tensor with shape [batch, num_h_blocks,
      num_w_blocks, h_dim, w_dim, io_channels] must have the same size as
      query_length, but a different name.
    kv_channels: a mtf.Dimension (the size of the key and value vectors)
    heads: a mtf.Dimension (the number of heads)
    memory_h_dim: mtf Dimension, for the memory height block.
    memory_w_dim: mtf Dimension, for the memory width block.
    mask_right: bool, flag specifying whether we mask out attention to the right
      for the decoder.
    master_dtype: a tf.dtype
    slice_dtype: a tf.dtype
    name: an optional string.

  Returns:
    a Tensor of shape
        [batch, num_h_blocks, num_w_blocks, h_dim, w_dim, io_channels]

  Raises:
    ValueError: if channels or depth don't match.
  """
  with tf.variable_scope(
      name, default_name="multihead_attention", values=[query_antecedent]):

    h_dim, w_dim, io_channels = query_antecedent.shape.dims[-3:]
    batch, num_h_blocks, num_w_blocks = query_antecedent.shape.dims[:3]
    wq, wk, wv, wo = multihead_attention_vars(
        query_antecedent.mesh, heads, io_channels, kv_channels,
        master_dtype, slice_dtype, query_antecedent.dtype)

    # Rename dimensions for the memory height and width.
    memory_antecedent = mtf.rename_dimension(query_antecedent, h_dim.name,
                                             "memory_" + h_dim.name)
    memory_antecedent = mtf.rename_dimension(memory_antecedent, w_dim.name,
                                             "memory_" + w_dim.name)
    memory_h_dim, memory_w_dim = memory_antecedent.shape.dims[-3:-1]

    # Call einsum over the query and memory to get query q, keys k and values v.
    q = mtf.einsum([query_antecedent, wq],
                   mtf.Shape([
                       batch, heads, num_h_blocks, num_w_blocks, h_dim, w_dim,
                       kv_channels
                   ]))
    k = mtf.einsum([memory_antecedent, wk],
                   mtf.Shape([batch, heads, num_h_blocks, num_w_blocks,
                              memory_h_dim, memory_w_dim, kv_channels]))
    v = mtf.einsum([memory_antecedent, wv],
                   mtf.Shape([batch, heads, num_h_blocks, num_w_blocks,
                              memory_h_dim, memory_w_dim, kv_channels]))

    # Halo exchange for memory blocks.
    k, v = local_2d_halo_exchange(k, v, num_h_blocks, memory_h_dim,
                                  num_w_blocks, memory_w_dim, mask_right)

    # Calculate the causal mask to avoid peeking into the future. We compute
    # this once and reuse it for all blocks since the block_size is known.
    mask = None
    if mask_right:
      mask = attention_bias_local_2d_block(query_antecedent.mesh, h_dim, w_dim,
                                           memory_h_dim, memory_w_dim)

    output = dot_product_attention(q, k, v, mask=mask)

    return mtf.einsum(
        [output, wo],
        mtf.Shape(
            [batch, num_h_blocks, num_w_blocks, h_dim, w_dim, io_channels]))


def rename_length_to_memory_length(
    x, length_name="length", memory_length_name="memory_length"):
  return mtf.rename_dimension(x, length_name, memory_length_name)


def multihead_attention_vars(
    mesh, heads, io_channels, kv_channels,
    master_dtype, slice_dtype, activation_dtype):
  """Deprecated version of multihead_attention_params with combine=True."""
  return multihead_attention_params(
      mesh, heads, io_channels, kv_channels,
      mtf.VariableDType(master_dtype, slice_dtype, activation_dtype),
      combine=True)


def multihead_attention_params(mesh, heads, io_channels, kv_channels,
                               variable_dtype, combine=False):
  """Create Parameters for Multihead Attention.

  If the combine flag is set to True, then we create only one variable
  which stacks together all of the parameters.  Otherwise, we create four
  separate variables.

  Args:
    mesh: a Mesh
    heads: a Dimension
    io_channels: a Dimension
    kv_channels: a Dimension
    variable_dtype: a mtf.VariableDType
    combine: a boolean

  Returns:
    wq: a Tensor with shape [heads, io_channels, kv_channels]
    wk: a Tensor with shape [heads, io_channels, kv_channels]
    wv: a Tensor with shape [heads, io_channels, kv_channels]
    wo: a Tensor with shape [heads, io_channels, kv_channels]
  """
  qkvo = mtf.Dimension("qkvo", 4)
  qk_stddev = (io_channels.size ** -0.5) * (kv_channels.size ** -0.25)
  v_stddev = io_channels.size ** -0.5
  o_stddev = (io_channels.size * heads.size) ** -0.5
  if combine:
    def qkvo_initializer(shape,
                         dtype=None,
                         partition_info=None,
                         verify_shape=None):
      del partition_info, verify_shape
      return tf.random_normal(shape, dtype=dtype) * tf.reshape(
          tf.cast([qk_stddev, qk_stddev, v_stddev, o_stddev],
                  dtype or tf.float32), [4, 1, 1, 1])
    var = mtf.get_variable(
        mesh, "qkvo", mtf.Shape([qkvo, heads, io_channels, kv_channels]),
        initializer=qkvo_initializer, dtype=variable_dtype)
    return mtf.unstack(var, qkvo)
  else:
    return [mtf.get_variable(
        mesh, name, mtf.Shape([heads, io_channels, kv_channels]),
        initializer=tf.random_normal_initializer(stddev=stddev),
        dtype=variable_dtype) for name, stddev in zip(
            ["q", "k", "v", "o"],
            [qk_stddev, qk_stddev, v_stddev, o_stddev])]


def dot_product_attention_v2(q,
                             k,
                             v,
                             memory_length_dim,
                             k_dim,
                             v_dim,
                             mask=None,
                             dropout=0.0,
                             dropout_broadcast_dims=None,
                             extra_logit=None):
  """Dot-product attention - doesn't use positional dimensions.

  k_dim is a Dimension representing the channels in the queries and keys
  v_dim is a Dimension representing the channels in values
  memory_length_dim is a Dimension representing the different key/value pairs.

  Dimensions of q: other_query_dims + {k_dim}
  Dimensions of k: other_memory_dims + {memory_length_dim, k_dim}
  Dimensions of v: other_memory_dims + {memory_length_dim, v_dim}
  other_memory_dims is a subset of other_query_dims

  Typically, other_query_dims={batch, heads, length}
  Typically, other_memory_dims={batch, heads}

  Args:
    q: a Tensor
    k: a Tensor
    v: a Tensor
    memory_length_dim: a Dimension
    k_dim: a Dimension
    v_dim: a Dimension
    mask: mask Tensor (see attention_mask())
    dropout: a float.
    dropout_broadcast_dims: an optional list of mtf.Dimension
    extra_logit: an optional scalar or tensor

  Returns:
    Tensor with shape q.shape - k_dim + v_dim
  """
  logits_shape = q.shape - k_dim + memory_length_dim
  logits = mtf.einsum([q, k], logits_shape, reduced_dims=[k_dim])
  if mask is not None:
    logits += mask
  weights = mtf.softmax(logits, memory_length_dim, extra_logit=extra_logit)
  if dropout != 0.0:
    weights = mtf.dropout(
        weights, 1.0 - dropout,
        noise_shape=weights.shape - dropout_broadcast_dims)
  outputs_shape = q.shape - k_dim + v_dim
  outputs = mtf.einsum(
      [weights, v], outputs_shape, reduced_dims=[memory_length_dim])
  return outputs


def dot_product_attention(q,
                          k,
                          v,
                          mask,
                          dropout=0.0,
                          dropout_broadcast_dims=None,
                          extra_logit=None):
  """Dot-product attention.

  Args:
    q: Tensor with shape [...., length_q, depth_k]. Typically leading dimensions
      are [batch, heads].
    k: Tensor with shape [..., length_kv, depth_k]. Leading dimensions must
      match with q.
    v: Tensor with shape [..., length_kv, depth_v] Leading dimensions must
      match with q.
    mask: mask Tensor (see attention_mask())
    dropout: a float.
    dropout_broadcast_dims: an optional list of mtf.Dimension
    extra_logit: an optional scalar or tensor

  Returns:
    Tensor with shape [..., length_q, depth_v].
  """
  length_kv = k.shape.dims[-2]
  logits_shape = mtf.Shape(q.shape.dims[:-1] + [length_kv])
  logits = mtf.einsum([q, k], logits_shape)
  if mask is not None:
    logits += mask
  weights = mtf.softmax(logits, length_kv, extra_logit=extra_logit)
  if dropout != 0.0:
    weights = mtf.dropout(
        weights, 1.0 - dropout,
        noise_shape=weights.shape - dropout_broadcast_dims)
  depth_v = v.shape.dims[-1]
  outputs_shape = mtf.Shape(q.shape.dims[:-1] + [depth_v])
  outputs = mtf.einsum([weights, v], outputs_shape)
  return outputs


def multihead_attention(query_antecedent,
                        memory_antecedent,
                        mask,
                        kv_channels,
                        heads,
                        dropout=0.0,
                        dropout_broadcast_dims=None,
                        master_dtype=tf.float32,
                        slice_dtype=tf.float32,
                        name="multihead_attention"):
  """Multihead scaled-dot-product attention with input/output transformations.

  In order to use only one variable containing the four weight matrices
  packed together, we insist that the query and memory antecedents have the
  same dimensionality (io_channels) and that the keys and values have the
  same dimensionality (kv_channels).

  Args:
    query_antecedent: a mtf.Tensor with shape
      [<batch_dims>, query_length, io_channels]
    memory_antecedent: a mtf.Tensor with shape
      [batch, memory_length, io_channels] (optional)
    mask: mask Tensor (see attention_mask())
    kv_channels: a mtf.Dimension (the size of the key and value vectors)
    heads: a mtf.Dimension (the number of heads)
    dropout: a floating point value
    dropout_broadcast_dims: an optional list of mtf.Dimension
    master_dtype: a tf.dtype
    slice_dtype: a tf.dtype
    name: an optional string.

  Returns:
    A mtf.Tensor with shape [batch, query_length, io_channels]

  Raises:
    ValueError: if the dimensions do not match.
  """
  batch_dims = query_antecedent.shape.dims[:-2]
  query_length, io_channels = query_antecedent.shape.dims[-2:]
  with tf.variable_scope(name,
                         default_name="multihead_attention",
                         values=[query_antecedent, memory_antecedent]):
    wq, wk, wv, wo = multihead_attention_vars(
        query_antecedent.mesh, heads, io_channels, kv_channels,
        master_dtype, slice_dtype, query_antecedent.dtype)
    if memory_antecedent is None:
      memory_antecedent = rename_length_to_memory_length(
          query_antecedent, query_length.name)
    memory_batch_dims = memory_antecedent.shape.dims[:-2]
    memory_length, memory_channels = memory_antecedent.shape.dims[-2:]
    if memory_batch_dims != batch_dims:
      raise ValueError("memory batch must equal query batch")
    if memory_channels != io_channels:
      raise ValueError("memory channels must equal query channels")
    q = mtf.einsum(
        [query_antecedent, wq],
        mtf.Shape(batch_dims + [heads, query_length, kv_channels]))
    k = mtf.einsum(
        [memory_antecedent, wk],
        mtf.Shape(batch_dims + [heads, memory_length, kv_channels]))
    v = mtf.einsum(
        [memory_antecedent, wv],
        mtf.Shape(batch_dims + [heads, memory_length, kv_channels]))
    o = dot_product_attention(
        q, k, v, mask, dropout, dropout_broadcast_dims)
    return mtf.einsum(
        [o, wo], mtf.Shape(batch_dims + [query_length, io_channels]))


def multihead_self_attention_incremental(query_antecedent,
                                         prev_k,
                                         prev_v,
                                         step_num,
                                         master_dtype,
                                         slice_dtype,
                                         name="multihead_attention"):
  """Incremental self-attention (one decode step).

  In order to use only one variable containing the four weight matrices
  packed together, we insist that the query and memory antecedents have the
  same dimensionality (io_channels) and that the keys and values have the
  same dimensionality (kv_channels).

  Args:
    query_antecedent: a mtf.Tensor with shape [batch..., io_channels]
    prev_k: mtf.Tensor with shape [batch..., heads, memory_length, kv_channels]
    prev_v: mtf.Tensor with shape [batch..., heads, memory_length, kv_channels]
    step_num: mtf Scalar with dtype tf.int32
    master_dtype: a tf.dtype
    slice_dtype: a tf.dtype
    name: an optional string.

  Returns:
    y: A mtf.Tensor with shape [batch..., io_channels]
    new_k: mtf.Tensor with shape [batch..., heads, memory_length, kv_channels]
    new_v: mtf.Tensor with shape [batch..., heads, memory_length, kv_channels]

  Raises:
    ValueError: if the dimensions do not match.
  """
  batch_dims = query_antecedent.shape.dims[:-1]
  io_channels = query_antecedent.shape.dims[-1]
  heads, memory_length, kv_channels = prev_k.shape.dims[-3:]
  with tf.variable_scope(name, default_name="multihead_attention"):
    wq, wk, wv, wo = multihead_attention_vars(
        query_antecedent.mesh, heads, io_channels, kv_channels,
        master_dtype, slice_dtype, query_antecedent.dtype)
    memory_antecedent = query_antecedent
    q = mtf.einsum(
        [query_antecedent, wq],
        mtf.Shape(batch_dims + [heads, kv_channels]))
    k = mtf.einsum(
        [memory_antecedent, wk],
        mtf.Shape(batch_dims + [heads, kv_channels]))
    v = mtf.einsum(
        [memory_antecedent, wv],
        mtf.Shape(batch_dims + [heads, kv_channels]))
    k = prev_k + mtf.multiply(
        k, mtf.one_hot(step_num, memory_length, dtype=prev_k.dtype),
        output_shape=prev_k.shape)
    v = prev_v + mtf.multiply(
        v, mtf.one_hot(step_num, memory_length, dtype=prev_v.dtype),
        output_shape=prev_v.shape)

    mask = mtf.cast(
        mtf.greater(mtf.range(
            query_antecedent.mesh, memory_length, dtype=tf.int32), step_num),
        q.dtype) * -1e9
    o = dot_product_attention(q, k, v, mask)
    y = mtf.einsum([o, wo], query_antecedent.shape)
    return y, k, v


def multihead_encdec_attention_incremental(query_antecedent,
                                           wq, wo, k, v,
                                           mask,
                                           name="multihead_attention"):
  """Incremental attention over encoder (one decode step).

  In order to use only one variable containing the four weight matrices
  packed together, we insist that the query and memory antecedents have the
  same dimensionality (io_channels) and that the keys and values have the
  same dimensionality (kv_channels).

  memory_dims is a subset of query_dims

  Args:
    query_antecedent: a mtf.Tensor with shape query_dims + [io_channels]
    wq: a mtf.Tensor with shape [heads, io_channels, kv_channels]
    wo: a mtf.Tensor with shape [heads, io_channels, kv_channels]
    k: memory_dims + [heads, memory_length, kv_channels]
    v: memory_dims + [heads, memory_length, kv_channels]
    mask: mask Tensor (see attention_mask())
    name: an optional string.

  Returns:
    A mtf.Tensor with shape [batch, qlen, io_channels]
  """
  heads, _, kv_channels = k.shape.dims[-3:]
  query_dims = query_antecedent.shape.dims[:-1]
  with tf.variable_scope(name, default_name="multihead_attention"):
    q = mtf.einsum(
        [query_antecedent, wq],
        mtf.Shape(query_dims + [heads, kv_channels]))
    o = dot_product_attention(q, k, v, mask)
    return mtf.einsum([o, wo], query_antecedent.shape)


def attention_mask_ignore_padding(inputs, dtype=tf.float32):
  """Bias for encoder-decoder attention.

  Args:
    inputs: a mtf.Tensor with shape [..., length_dim]
    dtype: a tf.dtype

  Returns:
    a mtf.Tensor with shape [..., memory_length_dim]
  """
  inputs = rename_length_to_memory_length(inputs)
  return mtf.cast(mtf.equal(inputs, 0), dtype) * -1e9


def attention_mask_autoregressive(query_pos, dtype=tf.float32):
  """Bias for self-attention where attention to the right is disallowed.

  Args:
    query_pos: a mtf.Tensor with shape [..., length_dim]
    dtype: a tf.dtype

  Returns:
    a mtf.Tensor with shape [..., length_dim, memory_length_dim]
  """
  memory_pos = rename_length_to_memory_length(query_pos)
  return mtf.cast(mtf.less(query_pos, memory_pos), dtype) * -1e9


def attention_mask_same_segment(
    query_segment, memory_segment=None, dtype=tf.float32):
  """Bias for attention where attention between segments is disallowed.

  Args:
    query_segment: a mtf.Tensor with shape [..., length_dim]
    memory_segment: a mtf.Tensor with shape [..., memory_length_dim]
    dtype: a tf.dtype

  Returns:
    a mtf.Tensor with shape [..., length_dim, memory_length_dim]
  """
  memory_segment = rename_length_to_memory_length(
      memory_segment or query_segment)
  return mtf.cast(mtf.not_equal(query_segment, memory_segment), dtype) * -1e9


def attention_bias_local_block(mesh, block_length, memory_length,
                               dtype=tf.int32):
  """Bias for attention for local blocks where attention to right is disallowed.

  Create the bias matrix by using two separate masks, one for the memory part
  which doesn't overlap with the query and second which interacts with the query
  and should be disallowed to look to the right of the current query position.

  Args:
    mesh: a MeshTensorflow object
    block_length: a mtf.Dimension
    memory_length: a mtf.Dimension
    dtype: a tf.dtype

  Returns:
    a mtf.Tensor with shape [block_length, memory_length]
  """
  memory_length = mtf.Dimension(memory_length.name, block_length.size)
  memory_mask = mtf.zeros(mesh, [block_length, memory_length], dtype=dtype)

  mask = mtf.cast(mtf.less(mtf.range(mesh, block_length, dtype=dtype),
                           mtf.range(mesh, memory_length, dtype=dtype)),
                  dtype=dtype)
  mask = mtf.cast(
      mtf.concat([memory_mask, mask], memory_length.name),
      dtype=tf.float32) * -1e9
  return mask


def attention_bias_local_2d_block(mesh,
                                  h_dim,
                                  w_dim,
                                  memory_h_dim,
                                  memory_w_dim,
                                  dtype=tf.int32):
  """Bias for attention for local blocks where attention to right is disallowed.

  Create the bias matrix by using two separate masks, one for the memory part
  which doesn't overlap with the query and second which interacts with the query
  and should be disallowed to look to the right of the current query position.

  Args:
    mesh: a MeshTensorflow object
    h_dim: a mtf.Dimension
    w_dim: a mtf.Dimension
    memory_h_dim: a mtf.Dimension
    memory_w_dim: a mtf.Dimension
    dtype: a tf.dtype

  Returns:
    a mtf.Tensor with shape [block_length, memory_length]
  """
  memory_height = mtf.Dimension(memory_h_dim.name, h_dim.size)
  memory_width = mtf.Dimension(memory_w_dim.name, w_dim.size)
  mask_top_visible = mtf.zeros(mesh, [h_dim, memory_height], dtype=dtype)
  mask_left_visible = mtf.zeros(mesh, [w_dim, memory_width], dtype=dtype)
  mask_query = mtf.greater(
      mtf.range(mesh, memory_height, dtype=tf.int32),
      mtf.range(mesh, memory_width, dtype=dtype))
  width_mask = mtf.concat([mask_left_visible, mask_query], memory_width.name)
  mask = mtf.cast(
      mtf.concat([mask_top_visible, width_mask], memory_height.name),
      dtype=tf.float32) * -1e9
  return mask


def multiplicative_jitter(x, epsilon=1e-2):
  """Multiply values by a random number between 1-epsilon and 1+epsilon.

  Makes models more resilient to rounding errors introduced by bfloat16.
  This seems particularly important for logits.

  Args:
    x: a mtf.Tensor
    epsilon: a floating point value

  Returns:
    a mtf.Tensor with the same type and shape as x.
  """
  if epsilon == 0:
    return x
  return x * mtf.random_uniform(
      x.mesh, x.shape, minval=1.0 - epsilon, maxval=1.0+epsilon, dtype=x.dtype)


def multihead_self_attention_memory_compressed(x,
                                               mask_right,
                                               compression_factor,
                                               kv_channels,
                                               heads,
                                               dropout=0.0,
                                               dropout_broadcast_dims=None,
                                               master_dtype=tf.float32,
                                               slice_dtype=tf.float32,
                                               name="multihead_attention"):
  """Memory-compressed self-attention.

  The memory is first average-pooled (strided) to make it shorter by
  a factor of compression_factor.

  Args:
    x: a mtf.Tensor with shape
      [<batch_dims>, query_length, io_channels]
    mask_right: a boolean
    compression_factor: an integer
    kv_channels: a mtf.Dimension (the size of the key and value vectors)
    heads: a mtf.Dimension (the number of heads)
    dropout: a floating point value
    dropout_broadcast_dims: an optional list of mtf.Dimension
    master_dtype: a tf.dtype
    slice_dtype: a tf.dtype
    name: an optional string.

  Returns:
    A mtf.Tensor with shape [batch, query_length, io_channels]

  Raises:
    ValueError: if the dimensions do not match.
  """
  batch_dims = x.shape.dims[:-2]
  length, io_channels = x.shape.dims[-2:]
  with tf.variable_scope(name,
                         default_name="compressed_attention",
                         values=[x]):
    wq, wk, wv, wo = multihead_attention_vars(
        x.mesh, heads, io_channels, kv_channels,
        master_dtype, slice_dtype, x.dtype)
    memory_antecedent = compress_mean(x, length, compression_factor)
    memory_antecedent = rename_length_to_memory_length(memory_antecedent)
    memory_length = memory_antecedent.shape.dims[-2]
    q = mtf.einsum(
        [x, wq],
        mtf.Shape(batch_dims + [heads, length, kv_channels]))
    k = mtf.einsum(
        [memory_antecedent, wk],
        mtf.Shape(batch_dims + [heads, memory_length, kv_channels]))
    v = mtf.einsum(
        [memory_antecedent, wv],
        mtf.Shape(batch_dims + [heads, memory_length, kv_channels]))
    if mask_right:
      query_pos = mtf.range(x.mesh, length, dtype=tf.int32)
      memory_pos = (
          mtf.range(x.mesh, memory_length, dtype=tf.int32) * compression_factor
          + (compression_factor - 1))
      mask = mtf.cast(mtf.greater(memory_pos, query_pos), x.dtype) * -1e9
    else:
      mask = None
    o = dot_product_attention(
        q, k, v, mask, dropout, dropout_broadcast_dims, extra_logit=0.0)
    return mtf.einsum(
        [o, wo], mtf.Shape(batch_dims + [length, io_channels]))


def compress_mean(x, dim, compression_factor):
  """Compress by taking group means.

  Args:
    x: a Tensor
    dim: a dimension in x.shape
    compression_factor: an integer

  Returns:
    a Tensor
  """
  dims = x.shape.dims
  pos = dims.index(dim)
  compressed_dim = mtf.Dimension(dim.name, dim.size // compression_factor)
  compression_factor_dim = mtf.Dimension(
      "compression_factor", compression_factor)
  new_shape = (
      dims[:pos] + [compressed_dim, compression_factor_dim] + dims[pos + 1:])
  x = mtf.reshape(x, new_shape)
  x = mtf.reduce_mean(x, reduced_dim=compression_factor_dim)
  return x


def embedding_weights(
    mesh, vocab_dim, output_dim, variable_dtype, name="embedding"):
  return mtf.get_variable(
      mesh, name, mtf.Shape([vocab_dim, output_dim]),
      dtype=variable_dtype, initializer=tf.random_normal_initializer())


def embedding(indices, vocab_dim, output_dim, variable_dtype, name="embedding"):
  """Embedding layer."""
  weights = embedding_weights(
      indices.mesh, vocab_dim, output_dim, variable_dtype, name)
  return mtf.gather(weights, indices, vocab_dim)
