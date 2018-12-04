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

"""MTF implementation of Transformer sequence/seq2seq model.

This implmentation is meant to be extensible, allowing users to define their
own custom layers.  It is meant to eventually replace the existing
mesh-tensorflow Transformer implementation in the Tensor2Tensor library.

Supported so far:
 - autoregressive single-stack Transformer (e.g. a simple language model)
 - non-autoregressive single-stack Transformer (e.g. BERT)
 - fast autoregressive sampling with temperature
 - mixture of experts layer
 - local attetion layer
 - wrapper for tensor2tensor (MtfTransformer2)

Not yet supported:  TODO(noam)
 - two-stack Transformer (e.g. machine-translation model)
 - beam search
 - shared embedding / shared embedding and softmax weights
 - compressed attention layer
 - training binary without tensor2tensor
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import mesh_tensorflow as mtf

import tensorflow as tf


class TransformerLayer(object):
  """Abstract base class for transformer layers.

  The point of this object hierarchy is to make Transformer extensible.  You can
  configure a Transformer with your own custom layers without changing the base
  library.

  Transformer layers should subclass TransformerLayer.  In the constructor, the
  subclasses simply record their hyperparmeters.  Subclasses must implement a
  call() method, representing a call to that layer.  The call method is passed
  an input tensor and a Context object.  Variables should be created inside of
  the call().

  Examples of subclasses can be found in transformer_layers.py.

  In addition to other global hyperparameters, the Context has a "mode", which
  might be tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL, or another
  special value.  Autoregressive decoding uses the special modes "first_part"
  and "incremental".  In "first_part" mode, the known first part of the sequence
  is passed through all layers so that they can create any necessary initial
  states.  In "incremental" mode (which is called from the body of a while
  loop), the input consists of only one position.  Layers with recurrent states
  must generate both output and the new states.
  """

  def call(self, context, x):
    """Call the layer.

    Args:
      context: a Context
      x: an input Tensor

    Returns:
      y: a Tensor
    """
    raise NotImplementedError("Not implemented")

  def to_json(self):
    return json.dumps(self, cls=json.JSONEncoder)


class Context(object):
  """Extra information that layers need at call time."""

  def __init__(self,
               mesh,
               batch_dims,
               length_dim,
               model_dim,
               variable_dtype,
               beam_dim=None,
               mode=tf.estimator.ModeKeys.TRAIN,
               autoregressive=False,
               position=None,
               states=None,
               new_states=None,
               losses=None,
               initial_position=None,
               layer_outputs=None):
    """Create a context.

    Args:
      mesh: a mtf.Mesh
      batch_dims: a list of mtf.Dimension
      length_dim: a mtf.Dimension
      model_dim: a mtf.Dimension
      variable_dtype: a mtf.VariableDType
      beam_dim: an optional mtf.Dimension
      mode: either a tf.estimator.ModeKeys or one of the follwing:
        "first_part"
        "incremental"
      autoregressive: a boolean
      position: an optional Tensor
      states: an optional list of Tensors ("incremental" mode)
      new_states: an optional list of Tensors onto which to append new states
         ("first_part" and "incremental" modes)
      losses: an optional list of Tensors onto which to append losses
      initial_position: an optional Tensor ("first_part" mode)
      layer_outputs: an optional list onto which to append layer outputs
    """
    self.mesh = mesh
    self.batch_dims = batch_dims
    self.length_dim = length_dim
    self.variable_dtype = variable_dtype
    self.beam_dim = beam_dim
    self.model_dim = model_dim
    self.mode = mode
    self.autoregressive = autoregressive
    if position is None:
      self.position_is_default = True
      self.position = mtf.range(mesh, length_dim, dtype=tf.int32)
    else:
      self.position_is_default = False
      self.position = position
    self.states = states
    self.new_states = new_states
    self.losses = losses
    self.initial_position = initial_position
    self.layer_outputs = layer_outputs

  @property
  def train(self):
    return self.mode == tf.estimator.ModeKeys.TRAIN

  def next_states(self, n):
    return self.states[len(self.new_states):len(self.new_states) + n]

  @property
  def activation_dtype(self):
    return self.variable_dtype.activation_dtype


class LayerStack(TransformerLayer):
  """A stack of layers with residual connections and layer norms."""

  def __init__(self,
               layers,
               dropout_rate=0.0,
               norm_epsilon=1e-6):
    """Create a LayerStack.

    Args:
      layers: a list of TransformerLayer
      dropout_rate: a floating-point number
      norm_epsilon: a floating-point number
    """
    self._layers = layers
    self._dropout_rate = dropout_rate
    self._norm_epsilon = norm_epsilon

  def call(self, context, x):
    """Call the layer stack."""
    x = self._dropout(context, x)
    if context.layer_outputs is not None:
      context.layer_outputs.append(x)
    for lnum, layer in enumerate(self._layers):
      with tf.variable_scope("layer_%03d" % lnum):
        norm_x = self._layer_norm(context, x)
        with tf.variable_scope(layer.__class__.__name__):
          y = layer.call(context, norm_x)
        x += self._dropout(context, y)
      if context.layer_outputs is not None:
        context.layer_outputs.append(x)
    x = self._layer_norm(context, x, name="final_layer_norm")
    x = self._dropout(context, x)
    return x

  def _dropout(self, context, x):
    if context.train and self._dropout_rate > 0:
      return mtf.dropout(
          x, keep_prob=1.0 - self._dropout_rate,
          noise_shape=mtf.Shape(context.batch_dims + [context.model_dim]))
    else:
      return x

  def _layer_norm(self, context, x, name=None):
    with tf.variable_scope(name, default_name="layer_norm"):
      scale = mtf.get_variable(
          context.mesh, "scale", mtf.Shape([context.model_dim]),
          initializer=tf.ones_initializer(),
          dtype=context.variable_dtype)
      variance = mtf.reduce_mean(mtf.square(x), reduced_dim=context.model_dim)
    return x * mtf.rsqrt(variance + self._norm_epsilon) * scale

  @property
  def num_layers(self):
    return len(self.layers)

  @property
  def layers(self):
    return self._layers


class Transformer(object):
  """A Transformer model with only one layer stack, e.g. a language model."""

  def __init__(self,
               layer_stack,
               d_model,
               input_vocab_size,
               output_vocab_size,
               autoregressive,
               max_length):
    self.layer_stack = layer_stack
    self.model_dim = mtf.Dimension("d_model", d_model)
    self.input_vocab_dim = mtf.Dimension("vocab", input_vocab_size)
    self.output_vocab_dim = mtf.Dimension("vocab", output_vocab_size)
    self.autoregressive = autoregressive
    self.max_length_dim = mtf.Dimension("max_length", max_length)

  def _call_internal(self, context, inputs, targets=None):
    """Compute logits based on inputs (all positions in parallel).

    Also updates context if applicable.

    Args:
      context: a Context
      inputs: a Tensor
      targets: an optional Tensor

    Returns:
      logits: a Tensor with shape [<batch_dims>, length_dim, output_vocab_dim]
    """
    mesh = inputs.mesh
    x = mtf.layers.embedding(
        inputs, self.input_vocab_dim, self.model_dim, context.variable_dtype)
    pos_emb_var = mtf.get_variable(
        mesh, "pos_emb",
        mtf.Shape([self.max_length_dim, self.model_dim]),
        initializer=tf.random_normal_initializer(),
        dtype=context.variable_dtype)
    if context.position_is_default:
      pos_emb = mtf.rename_dimension(
          mtf.slice(pos_emb_var, 0, context.length_dim.size,
                    self.max_length_dim.name),
          self.max_length_dim.name, context.length_dim.name)
    else:
      pos_emb = mtf.gather(
          pos_emb_var, context.position, self.max_length_dim,
          output_shape=x.shape)
    x += pos_emb
    x = self.layer_stack.call(context, x)
    logits = mtf.layers.dense(
        x, self.output_vocab_dim, use_bias=False,
        variable_dtype=context.variable_dtype,
        name="logits")
    if context.train:
      logits = mtf.layers.multiplicative_jitter(logits, epsilon=1e-2)
    if targets is not None and context.losses is not None:
      loss = mtf.layers.softmax_cross_entropy_with_logits(
          logits,
          mtf.one_hot(
              targets, self.output_vocab_dim,
              dtype=context.activation_dtype),
          self.output_vocab_dim)
      weights = mtf.layers.weights_nonzero(
          targets, dtype=context.activation_dtype)
      loss = mtf.reduce_mean(loss * weights)
      context.losses.append(loss)
    return logits

  def call_simple(
      self,
      inputs,
      targets,
      compute_loss,
      mode=tf.estimator.ModeKeys.TRAIN,
      variable_dtype=mtf.VariableDType(tf.float32)):
    """Compute logits based on inputs (all positions in parallel).

    This is called during training and evaluation.

    Args:
      inputs: a Tensor with shape [<batch_dims>, length_dim]
        For training autoregressive models this should be equal to
        mtf.shift(targets, offset=1, dim=length_dim, wrap=False)
      targets: an optional Tensor with shape [<batch_dims>, length_dim]
      compute_loss: a boolean
      mode: a tf.estimator.ModeKeys
      variable_dtype: a mtf.VariableDType

    Returns:
      logits: a Tensor with shape [<batch_dims>, output_vocab_dim]
      loss: an optional Scalar (if compute_loss=True)
    """
    if compute_loss:
      if targets is None:
        raise ValueError("cannot comupte losses without targets")
      losses = []
    else:
      losses = None
    context = Context(
        mesh=inputs.mesh,
        batch_dims=inputs.shape.dims[:-1],
        length_dim=inputs.shape.dims[-1],
        model_dim=self.model_dim,
        variable_dtype=variable_dtype,
        mode=mode,
        autoregressive=self.autoregressive,
        losses=losses)
    logits = self._call_internal(context, inputs, targets)
    if compute_loss:
      loss = mtf.add_n(context.losses)
    else:
      loss = None
    return logits, loss

  def sample_autoregressive(self,
                            inputs,
                            stop_at_token=1,
                            max_steps=None,
                            temperature=1.0,
                            variable_dtype=mtf.VariableDType(tf.float32)):
    """Sample randomly one token at a time.

    The inputs represent partial sequences to be continued.  The first tokens
    of each sequence are nonzero representing the given partial sequences
    and the last tokens of each sequence are zeros, representing what needs
    to be filled in.

    Args:
      inputs: a Tensor with shape [<batch_dims>, length_dim]
      stop_at_token: an optional integer eos id.  Stop when we produce it.
      max_steps: an optional integer
      temperature: an optional floating point value between 0 and 1
      variable_dtype: a mtf.VariableDType

    Returns:
      a Tensor with shape [<batch_dims>, length_dim]
    """
    del max_steps  # TODO(noam): implement
    if not self.autoregressive:
      raise ValueError("must be autoregressive")

    batch_dims = inputs.shape.dims[:-1]
    length_dim = inputs.shape.dims[-1]
    initial_position = mtf.reduce_sum(
        mtf.to_int32(mtf.not_equal(inputs, 0)), reduced_dim=length_dim)

    context_first_part = Context(
        mesh=inputs.mesh,
        batch_dims=batch_dims,
        length_dim=length_dim,
        model_dim=self.model_dim,
        variable_dtype=variable_dtype,
        mode="first_part",
        autoregressive=self.autoregressive,
        new_states=[],
        initial_position=initial_position)

    shifted_inputs = mtf.shift(inputs, offset=1, dim=length_dim, wrap=False)
    with tf.variable_scope("transformer"):
      logits = self._call_internal(context_first_part, shifted_inputs)
    del logits
    initial_states = context_first_part.new_states

    def cond_fn(position, ids, *unused_states):
      """Should we run another loop iteration."""
      past_end = mtf.greater_equal(position, length_dim.size)
      is_done = past_end
      if stop_at_token is not None:
        has_eos = mtf.reduce_any(
            mtf.equal(ids, stop_at_token), reduced_dim=length_dim)
        is_done = mtf.logical_or(is_done, has_eos)
      all_done = mtf.reduce_all(is_done)
      return mtf.logical_not(all_done)

    def body_fn(position, ids, *states):
      """One step in the decode loop."""
      context_incremental = Context(
          mesh=inputs.mesh,
          batch_dims=batch_dims,
          length_dim=length_dim,
          model_dim=self.model_dim,
          variable_dtype=variable_dtype,
          mode="incremental",
          autoregressive=self.autoregressive,
          position=position,
          states=states,
          new_states=[])
      inputs_this_step = mtf.gather(ids, position - 1, length_dim)
      with tf.variable_scope("transformer", reuse=True):
        logits = self._call_internal(context_incremental, inputs_this_step)
      ids_this_step = mtf.sample_with_temperature(
          logits, self.output_vocab_dim, temperature)
      new_position = position + 1
      new_ids = ids + ids_this_step * mtf.one_hot(
          position, length_dim, dtype=tf.int32)
      return [new_position, new_ids] + context_incremental.new_states
    while_loop_inputs = [initial_position, inputs] + initial_states
    final_position, outputs = mtf.while_loop(
        cond_fn, body_fn, while_loop_inputs)[:2]
    del final_position
    return outputs
