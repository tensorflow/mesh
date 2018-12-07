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

The interface is for the user to create a Unitransformer or Bitransformer
object and then call its methods (call_simple, sample_autoregressive, etc.)
The Unitransformer or Bitransformer is configured by creating a LayerStack
object contiaining instances of TransformerLayer.  Users can subclass
TransformerLayer to create new types of layers.

Supported so far:
 - autoregressive single-stack Transformer (e.g. a simple language model)
 - encoder-decoder models (requires two Transformers)
 - non-autoregressive single-stack Transformer (e.g. BERT)
 - fast autoregressive sampling with temperature
 - beam search
 - mixture of experts layer
 - local attetion layer
 - wrapper for tensor2tensor (MtfTransformer2)
 - shared embedding / shared embedding and softmax weights

Not yet supported:  TODO(noam)
 - compressed attention layer
 - training binary without tensor2tensor

TODO(noam): move Unitransformer and Bitransformer classes to top of file.
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
  and "incremental".

  In "first_part" mode, the known first part of the sequence is passed through
  all layers so that they can create any necessary initial states.

  In "incremental" mode (which is called from the body of a while loop), the
  input consists of only one position.  Layers with recurrent states must
  generate both output and the new states.
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
  """Extra information that layers need at call time.

  This structure is created by Unitransformer and is passed to the layers.
  It contains information that may be necessary to some layers in some
  modes.

  In "first_part" and "incremental" modes, some layers modify the context
  by producing and consuming "states" and "constant_states".  The "states"
  are loop variables that change at each decoding step.  The "constant_states"
  are produced once in "first_part" mode and read in the iterative decoding
  step.
  """

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
               sequence_id=None,
               states=None,
               new_states=None,
               losses=None,
               initial_position=None,
               layer_outputs=None,
               encoder_output=None,
               encoder_sequence_id=None,
               constant_states=None,
               shared_params=None):
    """Create a context.

    Args:
      mesh: a mtf.Mesh
      batch_dims: a list of mtf.Dimension
      length_dim: a mtf.Dimension
      model_dim: a mtf.Dimension
      variable_dtype: a mtf.VariableDType
      beam_dim: an optional mtf.Dimension (present in beam search)
      mode: either a tf.estimator.ModeKeys or one of the follwing:
        "first_part"
        "incremental"
      autoregressive: a boolean - controls whether attention layers shold mask
        out the future.
      position: an optional Tensor - represents position in the sequence
      sequence_id: an optional int32 Tensor aligned with position - used to
        separate out different sequences which have been concatenated
        to form a single training example.  Also used to mark padding.
        Id 0 is used for padding, and different positive values
        are used for the different sequences.
      states: an optional list of Tensors representing loop variables
        (consumed in "incremental" mode)
      new_states: an optional list of Tensors onto which to append the new
         values of loop variables.
         (produced in "first_part" and "incremental" modes)
      losses: an optional list of Tensors onto which to append losses
      initial_position: an optional Tensor ("first_part" mode)
      layer_outputs: an optional list onto which to append layer outputs
      encoder_output: an optional Tensor (output of the encoder stack)
      encoder_sequence_id: an optional int32 Tensor (similar to sequence_id)
        but aligned with the encoder output.
      constant_states: an optional list of structures produced during
        "first_part" mode and consumed during "incremental" mode.
      shared_params: an optional dictionary which can be populated by
        parameters that are shared between Transformers - e.g. between the
        encoder and decoder Unitransformers in a Bitransformer.
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
    self.sequence_id = sequence_id
    self.states = states
    self.new_states = new_states
    self.losses = losses
    self.initial_position = initial_position
    self.layer_outputs = layer_outputs
    self.encoder_output = encoder_output
    self.encoder_sequence_id = encoder_sequence_id
    self.constant_states = constant_states
    self.next_constant_state = 0
    self.shared_params = shared_params or {}

  @property
  def train(self):
    return self.mode == tf.estimator.ModeKeys.TRAIN

  @property
  def activation_dtype(self):
    return self.variable_dtype.activation_dtype

  def get_states(self, n):
    """Get the next n recurrent states.

    Called by layers in "incremental" mode.

    Args:
      n: an integer
    Returns:
      a list of n Tensors
    """
    return self.states[len(self.new_states):len(self.new_states) + n]

  def record_new_states(self, new_states):
    """Record the new values of recurrent states.

    Called by layers in "first_part" or "incremental" mode.

    Args:
      new_states: a list of Tensors
    """
    self.new_states.extend(new_states)

  def record_constant_state(self, s):
    """Record state in "first_part" mode to be read in "incremental" mode.

    This is to record state that is computed once and does not change
    at every decoding step.

    Args:
      s: a structure
    """
    self.constant_states.append(s)

  def get_constant_state(self):
    """Read state that was written in "first_part" mode.

    Returns:
      a structure
    """
    ret = self.constant_states[self.next_constant_state]
    self.next_constant_state += 1
    return ret


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


class Unitransformer(object):
  """A Transformer model with only one layer stack, e.g. a language model.

  This class is also used as part of Bitransformer, which contains two
  Unitransformers.
  """

  def __init__(self,
               layer_stack,
               d_model,
               input_vocab_size,
               output_vocab_size,
               autoregressive,
               max_length,
               shared_embedding_and_softmax_weights=False,
               label_smoothing=0.0,
               z_loss=1e-4,
               name="transformer"):
    self.layer_stack = layer_stack
    self.model_dim = mtf.Dimension("d_model", d_model)
    self.input_vocab_dim = mtf.Dimension("vocab", input_vocab_size)
    if output_vocab_size:
      self.output_vocab_dim = mtf.Dimension("vocab", output_vocab_size)
    else:
      self.output_vocab_dim = None
      if autoregressive:
        raise ValueError("autoregressive Transformer needs output vocabulary")
    self.autoregressive = autoregressive
    self.max_length_dim = mtf.Dimension("max_length", max_length)
    self.shared_embedding_and_softmax_weights = (
        shared_embedding_and_softmax_weights)
    self.label_smoothing = label_smoothing
    self.z_loss = z_loss
    self.name = name

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
    if "embedding" in context.shared_params:
      embedding_weights = context.shared_params["embedding"]
    else:
      embedding_weights = mtf.layers.embedding_weights(
          mesh, self.input_vocab_dim, self.model_dim, context.variable_dtype,
          name="embedding")
    x = mtf.gather(embedding_weights, inputs, self.input_vocab_dim)
    if "positional_embedding" in context.shared_params:
      pos_emb_var = context.shared_params["positional_embedding"]
    else:
      pos_emb_var = mtf.layers.embedding_weights(
          mesh, self.max_length_dim, self.model_dim, context.variable_dtype,
          "positional_embedding")
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
    if self.output_vocab_dim is None:
      return x
    if self.shared_embedding_and_softmax_weights:
      logits = mtf.einsum(
          [x * (self.model_dim ** -0.5), embedding_weights],
          reduced_dims=[self.model_dim])
    else:
      logits = mtf.layers.dense(
          x, self.output_vocab_dim, use_bias=False,
          variable_dtype=context.variable_dtype,
          name="logits")
    if targets is not None and context.losses is not None:
      off_value = self.label_smoothing / self.output_vocab_dim.size
      on_value = 1.0 - self.label_smoothing + off_value
      soft_targets = mtf.one_hot(
          targets, self.output_vocab_dim,
          dtype=context.activation_dtype,
          on_value=on_value,
          off_value=off_value)
      loss = mtf.layers.softmax_cross_entropy_with_logits(
          logits, soft_targets, self.output_vocab_dim,
          z_loss=self.z_loss if context.train else 0.0)
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
      variable_dtype=mtf.VariableDType(tf.float32),
      sequence_id=None,
      encoder_output=None,
      encoder_sequence_id=None,
      shared_params=None):
    """Compute logits based on inputs (all positions in parallel).

    This is called during training and evaluation.

    Args:
      inputs: an int32 Tensor with shape [<batch_dims>, length_dim]
        For training autoregressive models this should be equal to
        mtf.shift(targets, offset=1, dim=length_dim, wrap=False)
      targets: an optional int32 Tensor with shape [<batch_dims>, length_dim]
      compute_loss: a boolean
      mode: a tf.estimator.ModeKeys
      variable_dtype: a mtf.VariableDType
      sequence_id: an optional Tensor
      encoder_output: an optional Tensor
      encoder_sequence_id: an optional Tensor
      shared_params: an optional dictionary

    Returns:
      logits: a Tensor with shape [<batch_dims>, output_vocab_dim]
      loss: an optional Scalar (if compute_loss=True)
    """
    context = Context(
        mesh=inputs.mesh,
        batch_dims=inputs.shape.dims[:-1],
        length_dim=inputs.shape.dims[-1],
        model_dim=self.model_dim,
        variable_dtype=variable_dtype,
        mode=mode,
        autoregressive=self.autoregressive,
        losses=[] if compute_loss else None,
        sequence_id=sequence_id,
        encoder_output=encoder_output,
        encoder_sequence_id=encoder_sequence_id,
        shared_params=shared_params)
    with tf.variable_scope(self.name):
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
                            variable_dtype=mtf.VariableDType(tf.float32),
                            encoder_output=None,
                            encoder_sequence_id=None,
                            shared_params=None):
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
      encoder_output: an optional Tensor
      encoder_sequence_id: an optional Tensor
      shared_params: an optional dictionary

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
    sequence_id = 1 if encoder_sequence_id is not None else None

    context_first_part = Context(
        mesh=inputs.mesh,
        batch_dims=batch_dims,
        length_dim=length_dim,
        model_dim=self.model_dim,
        variable_dtype=variable_dtype,
        mode="first_part",
        autoregressive=self.autoregressive,
        new_states=[],
        initial_position=initial_position,
        sequence_id=sequence_id,
        encoder_output=encoder_output,
        encoder_sequence_id=encoder_sequence_id,
        constant_states=[],
        shared_params=shared_params)

    shifted_inputs = mtf.shift(inputs, offset=1, dim=length_dim, wrap=False)
    with tf.variable_scope(self.name):
      logits = self._call_internal(context_first_part, shifted_inputs)
    del logits
    initial_states = context_first_part.new_states
    constant_states = context_first_part.constant_states

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
          new_states=[],
          sequence_id=sequence_id,
          encoder_output=encoder_output,
          encoder_sequence_id=encoder_sequence_id,
          constant_states=constant_states,
          shared_params=shared_params)
      inputs_this_step = mtf.gather(ids, position - 1, length_dim)
      with tf.variable_scope(self.name, reuse=True):
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

  def beam_search(self,
                  inputs,
                  decode_length,
                  variable_dtype=mtf.VariableDType(tf.float32),
                  encoder_output=None,
                  encoder_sequence_id=None,
                  alpha=0.6,
                  shared_params=None):
    """Beam search.

    Args:
      inputs: an int32 zero-Tensor with shape
         [<batch_dims>, beam_dim, length_dim].
      decode_length: an int32 mtf scalar.  Maximum decode length.
      variable_dtype: a mtf.VariableDType
      encoder_output: an optional Tensor
      encoder_sequence_id: an optional Tensor
      alpha: a floating point value (length bonus)
      shared_params: an optional dictionary

    Returns:
      a Tensor with shape [<batch_dims>, beam_dim, length_dim]
    """
    if not self.autoregressive:
      raise ValueError("must be autoregressive")

    batch_dims = inputs.shape.dims[:-2]
    if len(batch_dims) != 1:
      raise NotImplementedError(
          "beam search supports exactly one batch dimension.")
    beam_dim = inputs.shape.dims[-2]
    length_dim = inputs.shape.dims[-1]
    initial_position = mtf.reduce_sum(
        mtf.to_int32(mtf.not_equal(inputs, 0)), reduced_dim=length_dim)
    sequence_id = 1 if encoder_sequence_id is not None else None

    context_first_part = Context(
        mesh=inputs.mesh,
        batch_dims=batch_dims + [beam_dim],
        length_dim=length_dim,
        model_dim=self.model_dim,
        variable_dtype=variable_dtype,
        mode="first_part",
        autoregressive=self.autoregressive,
        new_states=[],
        initial_position=initial_position,
        sequence_id=sequence_id,
        encoder_output=encoder_output,
        encoder_sequence_id=encoder_sequence_id,
        constant_states=[],
        shared_params=shared_params)

    shifted_inputs = mtf.shift(inputs, offset=1, dim=length_dim, wrap=False)
    with tf.variable_scope(self.name):
      logits = self._call_internal(context_first_part, shifted_inputs)
    del logits
    initial_states = context_first_part.new_states
    constant_states = context_first_part.constant_states

    def logits_fn(step_num, ids, states):
      """logits_fn for mtf.beam_search.beam_search()."""
      context_incremental = Context(
          mesh=inputs.mesh,
          batch_dims=batch_dims + [beam_dim],
          length_dim=length_dim,
          model_dim=self.model_dim,
          variable_dtype=variable_dtype,
          mode="incremental",
          autoregressive=self.autoregressive,
          position=step_num,
          states=states,
          new_states=[],
          sequence_id=sequence_id,
          encoder_output=encoder_output,
          encoder_sequence_id=encoder_sequence_id,
          constant_states=constant_states,
          shared_params=shared_params)
      inputs_this_step = mtf.gather(ids, step_num - 1, length_dim)
      with tf.variable_scope(self.name, reuse=True):
        logits = self._call_internal(context_incremental, inputs_this_step)
      return logits, context_incremental.new_states

    beams, unused_scores = mtf.beam_search.beam_search(
        logits_fn,
        inputs,
        alpha,
        states=initial_states,
        decode_length=decode_length,
        use_tpu=True,
        dtype=variable_dtype.activation_dtype)
    return mtf.gather(
        beams, mtf.constant(inputs.mesh, 0, dtype=tf.int32), beam_dim)


class Bitransformer(object):
  """A Transformer model with only one layer stack, e.g. a language model."""

  def __init__(self,
               encoder_layer_stack,
               decoder_layer_stack,
               encoder_d_model,
               decoder_d_model,
               input_vocab_size,
               output_vocab_size,
               max_length,
               shared_embedding=True,
               label_smoothing=0.0,
               z_loss=1e-4,
               encoder_name="encoder",
               decoder_name="decoder"):
    self.encoder = Unitransformer(
        encoder_layer_stack,
        encoder_d_model,
        input_vocab_size=input_vocab_size,
        output_vocab_size=None,
        autoregressive=False,
        max_length=max_length,
        name=encoder_name)
    self.decoder = Unitransformer(
        decoder_layer_stack,
        decoder_d_model,
        input_vocab_size=output_vocab_size,
        output_vocab_size=output_vocab_size,
        autoregressive=True,
        max_length=max_length,
        label_smoothing=label_smoothing,
        z_loss=z_loss,
        name=decoder_name)
    self.shared_embedding = shared_embedding

  def _shared_params(self, mesh, variable_dtype):
    """Create parameters that are shared between encoder and decoder.

    Args:
      mesh: a Mesh
      variable_dtype: a VariableDType
    Returns:
      a dictionary
    """
    shared_params = {}
    if self.shared_embedding:
      with tf.variable_scope("shared"):
        compatible = (
            self.encoder.model_dim == self.decoder.model_dim and
            self.encoder.input_vocab_dim == self.decoder.input_vocab_dim)
        if not compatible:
          raise ValueError(
              "shared_embedding requires encoder and decoder to have identical"
              " d_model and vocabulary sizes")
        shared_params["embedding"] = mtf.layers.embedding_weights(
            mesh,
            self.encoder.input_vocab_dim,
            self.encoder.model_dim,
            variable_dtype,
            name="embedding")
        shared_params["positional_embedding"] = mtf.layers.embedding_weights(
            mesh, self.encoder.max_length_dim, self.encoder.model_dim,
            variable_dtype, "positional_embedding")
    return shared_params

  def call_simple(
      self,
      inputs,
      targets,
      compute_loss,
      mode=tf.estimator.ModeKeys.TRAIN,
      variable_dtype=mtf.VariableDType(tf.float32),
      encoder_sequence_id=None,
      decoder_sequence_id=None):
    """Compute logits based on inputs (all positions in parallel).

    This is called during training and evaluation.

    Args:
      inputs: an int32 Tensor with shape [<batch_dims>, length_dim]
      targets: an optional int32 Tensor with shape [<batch_dims>, length_dim]
      compute_loss: a boolean
      mode: a tf.estimator.ModeKeys
      variable_dtype: a mtf.VariableDType
      encoder_sequence_id: an optional Tensor
      decoder_sequence_id: an optional Tensor

    Returns:
      logits: a Tensor with shape [<batch_dims>, output_vocab_dim]
      loss: an optional Scalar (if compute_loss=True)
    """
    shared_params = self._shared_params(inputs.mesh, variable_dtype)
    encoder_output, encoder_loss = self.encoder.call_simple(
        inputs,
        None,
        compute_loss,
        mode=mode,
        variable_dtype=variable_dtype,
        sequence_id=encoder_sequence_id,
        shared_params=shared_params)
    encoder_output = mtf.layers.rename_length_to_memory_length(encoder_output)
    if encoder_sequence_id is not None:
      encoder_sequence_id = mtf.layers.rename_length_to_memory_length(
          encoder_sequence_id)
    length_dim = targets.shape.dims[-1]
    shifted_targets = mtf.shift(targets, offset=1, dim=length_dim, wrap=False)
    logits, loss = self.decoder.call_simple(
        shifted_targets,
        targets,
        compute_loss,
        mode=mode,
        variable_dtype=variable_dtype,
        sequence_id=decoder_sequence_id,
        encoder_output=encoder_output,
        encoder_sequence_id=encoder_sequence_id,
        shared_params=shared_params)
    if loss is not None and encoder_loss is not None:
      loss += encoder_loss
    return logits, loss

  def decode(self,
             inputs,
             variable_dtype=mtf.VariableDType(tf.float32),
             beam_size=1,
             alpha=0.6,
             temperature=1.0,
             decode_length_multiplier=1.5,
             decode_length_constant=10):
    """Sampling or beam search.

    TODO(noam): should we make the output length dimension different from the
    input length dimension?

    Args:
      inputs: a Tensor with shape [<batch_dims>, beam_dim, length_dim]
      variable_dtype: a mtf.VariableDType
      beam_size: an integer >= 1
      alpha: a floating point value (length bonus for beam search)
      temperature: a value between 0 and 1 (must be 0 if beam_size > 1)
      decode_length_multiplier: a float
      decode_length_constant: a float

    Returns:
      a Tensor with shape [<batch_dims>, beam_dim, length_dim]
    """
    shared_params = self._shared_params(inputs.mesh, variable_dtype)
    encoder_sequence_id = mtf.minimum(inputs, 1)
    encoder_output, encoder_loss = self.encoder.call_simple(
        inputs=inputs,
        targets=None,
        compute_loss=False,
        mode=tf.estimator.ModeKeys.PREDICT,
        variable_dtype=variable_dtype,
        sequence_id=encoder_sequence_id,
        shared_params=shared_params)
    del encoder_loss
    encoder_output = mtf.layers.rename_length_to_memory_length(encoder_output)
    encoder_sequence_id = mtf.layers.rename_length_to_memory_length(
        encoder_sequence_id)
    if beam_size == 1:
      ids_shape = inputs.shape
      partial_targets = mtf.constant(inputs.mesh, 0, ids_shape, dtype=tf.int32)
      return self.decoder.sample_autoregressive(
          partial_targets,
          temperature=temperature,
          variable_dtype=variable_dtype,
          encoder_output=encoder_output,
          encoder_sequence_id=encoder_sequence_id,
          shared_params=shared_params)
    else:
      if temperature != 0:
        raise ValueError(
            "don't know how to beam search with nonzero temperature")
      # beam search
      beam_dim = mtf.Dimension("beam", beam_size)
      batch_dims = inputs.shape[:-1]
      length_dim = inputs.shape[-1]
      ids_shape = mtf.Shape(batch_dims + [beam_dim, length_dim])
      partial_targets = mtf.constant(inputs.mesh, 0, ids_shape, dtype=tf.int32)
      input_length = mtf.reduce_sum(
          mtf.to_float(mtf.cast(inputs, tf.bool)),
          reduced_dim=length_dim)
      max_input_length = mtf.reduce_max(input_length)
      decode_length = mtf.cast(
          max_input_length * decode_length_multiplier
          + decode_length_constant, tf.int32)
      return self.decoder.beam_search(
          partial_targets,
          decode_length,
          variable_dtype=variable_dtype,
          encoder_output=encoder_output,
          encoder_sequence_id=encoder_sequence_id,
          alpha=alpha,
          shared_params=shared_params)
