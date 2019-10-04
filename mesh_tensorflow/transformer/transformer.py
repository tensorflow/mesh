# coding=utf-8
# Copyright 2019 The Mesh TensorFlow Authors.
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
import gin
import mesh_tensorflow as mtf

import tensorflow.compat.v1 as tf


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

  def set_name(self, name):
    self._name = name

  @property
  def name(self):
    return getattr(self, "_name", None)


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
               model,
               mesh,
               batch_dims,
               length_dim,
               variable_dtype,
               beam_dim=None,
               mode=tf.estimator.ModeKeys.TRAIN,
               position=None,
               position_is_default=False,
               sequence_id=None,
               subsequence_id=None,
               states=None,
               new_states=None,
               losses=None,
               initial_position=None,
               layer_outputs=None,
               encoder_output=None,
               encoder_sequence_id=None,
               constant_states=None,
               shared_params=None,
               encoder_layer_outputs=None,
               write_priority=None,
               read_priority=None,
               inputs=None,
               encoder_inputs=None):
    """Create a context.

    Args:
      model: a pointer back at the unitransformer object
      mesh: a mtf.Mesh
      batch_dims: a list of mtf.Dimension
      length_dim: a mtf.Dimension
      variable_dtype: a mtf.VariableDType
      beam_dim: an optional mtf.Dimension (present in beam search)
      mode: either a tf.estimator.ModeKeys or one of the follwing:
        "first_part"
        "incremental"
      position: an optional Tensor - represents position in the sequence.
        Passing None means that the position should be considered to be the
        index in the Tensor (along length_dim).
      position_is_default: a boolean - is the position equal to
        mtf.range(mesh, length_dim, tf.int32).  This allows a shortcut in
        embedding lookup, as we can just slice the embedding variable.
      sequence_id: an optional int32 Tensor aligned with position - used to
        separate out different sequences which have been concatenated
        to form a single training example.  Also used to mark padding.
        Id 0 is used for padding, and different positive values
        are used for the different sequences.
      subsequence_id: an optional int32 Tensor - used to represent multiple
        targets corresponding to the same input. Should only be provided when
        being called as a decoder. If provided, then position should line up
        with this rather than sequence_id. The sequence_id will represent the
        groups of sub-targets corresponding to each input.
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
      encoder_layer_outputs: optional - readonly list of tensor activations when
        decoding, one per each input layer + the embedding layer
      write_priority: an optional Tensor
        in self-attention, position a can see position b iff
        read_priority[a] >= write_priority[b]
      read_priority: an optional Tensor
      inputs: an optional int32 Tensor with the input token ids
      encoder_inputs: an optional int32 Tensor with the input token ids to the
        encoder half of the Bitransformer of which this Unitransformer is the
        decoder.
    """
    self.model = model
    self.mesh = mesh
    self.batch_dims = batch_dims
    self.length_dim = length_dim
    self.variable_dtype = variable_dtype
    self.beam_dim = beam_dim
    self.mode = mode
    self.position = position
    self.position_is_default = position_is_default
    self.sequence_id = sequence_id
    self.subsequence_id = subsequence_id
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
    self.layer_index = 0
    self.encoder_layer_outputs = encoder_layer_outputs
    # put values here to share them between layers
    self.cache = {}
    self.write_priority = write_priority
    self.read_priority = read_priority
    self.inputs = inputs
    self.encoder_inputs = encoder_inputs

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

  @property
  def nonpadding(self):
    """Tensor with zeros in padding positions and ones elsewhere."""
    if self.sequence_id is None:
      return None
    if self.sequence_id == 1:
      return 1
    else:
      return mtf.cast(
          mtf.not_equal(self.sequence_id, 0), self.activation_dtype)


@gin.configurable
class LayerStack(TransformerLayer):
  """A stack of layers with residual connections and layer norms."""

  def __init__(self, layers, dropout_rate=0.0, norm_epsilon=1e-6):
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
    if isinstance(context.sequence_id, mtf.Tensor):
      # We use this mask to zero out the padding regions at each layer.
      # This "fixes" a bug where extreme values leak from the padding into the
      # non-padding regions.
      # TODO(noam): undertand this better and make a more principled fix.
      mask = mtf.cast(
          mtf.not_equal(context.sequence_id, 0), context.activation_dtype)
    else:
      mask = None
    x = self._dropout(context, x)
    if context.layer_outputs is not None:
      context.layer_outputs.append(x)
    for lnum, layer in enumerate(self._layers):
      with tf.variable_scope(layer.name or ""):
        norm_x = self._layer_norm(context, (x * mask) if mask else x)
        with tf.variable_scope(layer.__class__.__name__):
          y = layer.call(context, norm_x)
          if y.shape != x.shape:
            raise ValueError("Layer %s returned misshaped output x=%s y=%s"
                             % (layer.__class__.__name__, x, y))
        x += self._dropout(context, y)
      if context.layer_outputs is not None and lnum != len(self._layers) - 1:
        context.layer_outputs.append(x)
      context.layer_index += 1
    x = self._layer_norm(context, x, name="final_layer_norm")
    x = self._dropout(context, x)
    if mask:
      x *= mask
    if context.layer_outputs is not None:
      context.layer_outputs.append(x)
    return x

  def _dropout(self, context, x):
    if context.train and self._dropout_rate > 0:
      return mtf.dropout(
          x, keep_prob=1.0 - self._dropout_rate,
          noise_shape=mtf.Shape(context.batch_dims + [context.model.model_dim]))
    else:
      return x

  def _layer_norm(self, context, x, name=None):
    """Layer normalization.

    Args:
      context: a Context
      x: a Tensor
      name: an optional string
    Returns:
      a Tensor
    """
    with tf.variable_scope(name, default_name="layer_norm"):
      scale_shape = [context.model.model_dim]
      if context.model.ensemble_dim:
        scale_shape = [context.model.ensemble_dim] + scale_shape
      scale = mtf.get_variable(
          context.mesh, "scale", mtf.Shape(scale_shape),
          initializer=tf.ones_initializer(),
          dtype=context.variable_dtype)
      variance = mtf.reduce_mean(
          mtf.square(x), reduced_dim=context.model.model_dim)
    return x * mtf.rsqrt(variance + self._norm_epsilon) * scale

  @property
  def num_layers(self):
    return len(self.layers)

  @property
  def layers(self):
    return self._layers


@gin.configurable
class Unitransformer(object):
  """A Transformer model with only one layer stack, e.g. a language model.

  This class is also used as part of Bitransformer, which contains two
  Unitransformers.
  """

  def __init__(self,
               layer_stack,
               d_model=1024,
               input_vocab_size=gin.REQUIRED,
               output_vocab_size=gin.REQUIRED,
               autoregressive=gin.REQUIRED,
               max_length=gin.REQUIRED,
               shared_embedding_and_softmax_weights=False,
               label_smoothing=0.0,
               z_loss=1e-4,
               name="transformer",
               layout=None,
               mesh_shape=None,
               vocab_divisor=128,
               ensemble=None,
               loss_fn=None,
               positional_embedding=True,
               input_full_attention=False,
               loss_on_targets_only=False,
               loss_denominator=None):
    """Create a Unitransformer.

    Args:
      layer_stack: a LayerStack
      d_model: an integer
      input_vocab_size: an integer
      output_vocab_size: an integer
      autoregressive: a boolean
      max_length: an integer
      shared_embedding_and_softmax_weights: a boolean
      label_smoothing: a float
      z_loss: a float
      name: a string
      layout: optional - an input to mtf.convert_to_layout_rules
        Some layers (e.g. MoE layers) cheat by looking at layout and mesh_shape
      mesh_shape: optional - an input to mtf.convert_to_shape
        Some layers (e.g. MoE layers) cheat by looking at layout and mesh_shape
      vocab_divisor: an integer
      ensemble: an optional integer (for creating an ensemble of models)
      loss_fn: an optional function to override self._compute_loss
      positional_embedding: a boolean
      input_full_attention: a boolean
        This is an option for seq-to-seq as a language model.  Each example
        consists of [<inputs>, EOS=1, <targets>, EOS=1].  In the self-attention
        layers, positions in the inputs portion of the sequence can see the
        entire inputs portion, while positions in the targets portion of the
        sequence cannot see future positions.
      loss_on_targets_only: a boolean
        This is an option for seq-to-seq as a language model.  Each example
        consists of [<inputs>, EOS=1, <targets>, EOS=1].  We zero-out the
        loss for the inputs portion of the example.
      loss_denominator: an optional float.  The default behavior is to
        compute the mean loss across all tokens in the batch, making the
        denomiator the size of the targets tensor (omitting ensemble
        dimensions).
        Passing a float here provides an alternative denomiator.
        One use case is that when fine-tuning a model using a much smaller
        batch size than the original training batch, one might want to use the
        same denominator as was used for the pretraining.  This complication
        might be avoided by always using loss_denominator = 1.0.
    """
    self.layer_stack = layer_stack
    self.model_dim = mtf.Dimension("d_model", d_model)
    self.input_vocab_dim = mtf.Dimension(
        "vocab", _round_up_to_multiple(input_vocab_size, vocab_divisor))
    if output_vocab_size:
      self.output_vocab_dim = mtf.Dimension(
          "vocab", _round_up_to_multiple(output_vocab_size, vocab_divisor))
    else:
      self.output_vocab_dim = None
      if autoregressive:
        raise ValueError("autoregressive Transformer needs output vocabulary")
    self.autoregressive = autoregressive
    if positional_embedding:
      self.max_length_dim = mtf.Dimension("max_length", max_length)
    else:
      self.max_length_dim = None
    self.shared_embedding_and_softmax_weights = (
        shared_embedding_and_softmax_weights)
    self.label_smoothing = label_smoothing
    self.z_loss = z_loss
    self.name = name
    self.layout = layout
    self.mesh_shape = mesh_shape
    self.ensemble_dim = (
        mtf.Dimension("ensemble", ensemble) if ensemble else None)
    if loss_fn:
      self._compute_loss = loss_fn
    self.positional_embedding = positional_embedding
    self.input_full_attention = input_full_attention
    self.loss_on_targets_only = loss_on_targets_only
    self._loss_denominator = loss_denominator
    if self.input_full_attention and not self.autoregressive:
      raise ValueError(
          "input_full_attention only makes sense with autoregressive")

  @property
  def fully_autoregressive(self):
    return self.autoregressive and not self.input_full_attention

  def _compute_loss(self, context, logits, targets, output_vocab_dim):
    """Regular cross entropy loss.

    Args:
      context: a Context
      logits: a Tensor, the logits from the decoder
      targets: an Tensor
      output_vocab_dim: a Dimension

    Returns:
      A 0-dimensional tensor of the loss.
    """
    off_value = self.label_smoothing / output_vocab_dim.size
    on_value = 1.0 - self.label_smoothing + off_value
    soft_targets = mtf.one_hot(
        targets,
        output_vocab_dim,
        dtype=context.activation_dtype,
        on_value=on_value,
        off_value=off_value)
    loss = mtf.layers.softmax_cross_entropy_with_logits(
        logits,
        soft_targets,
        output_vocab_dim,
        z_loss=self.z_loss if context.train else 0.0)
    weights = mtf.layers.weights_nonzero(
        targets, dtype=context.activation_dtype)
    if self.loss_on_targets_only:
      weights *= mtf.cast(mtf.logical_not(text2self_inputs_mask(targets)),
                          dtype=context.activation_dtype)
    return mtf.reduce_sum(loss * weights) / self.loss_denominator(targets)

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
    if self.ensemble_dim and self.ensemble_dim not in inputs.shape.dims:
      # Training an ensemble where all models are trained on the same examples.
      inputs = mtf.broadcast(inputs, [self.ensemble_dim] + inputs.shape.dims)
      if targets:
        targets = mtf.broadcast(
            targets, [self.ensemble_dim] + targets.shape.dims)
    if "embedding" in context.shared_params:
      embedding_weights = context.shared_params["embedding"]
    else:
      embedding_weights = mtf.layers.embedding_weights(
          mesh, self.input_vocab_dim, self.model_dim, context.variable_dtype,
          name="embedding", ensemble_dim=self.ensemble_dim)
    x = mtf.gather(embedding_weights, inputs, self.input_vocab_dim)
    if self.positional_embedding:
      if "positional_embedding" in context.shared_params:
        pos_emb_var = context.shared_params["positional_embedding"]
      else:
        pos_emb_var = mtf.layers.embedding_weights(
            mesh, self.max_length_dim, self.model_dim, context.variable_dtype,
            "positional_embedding", ensemble_dim=self.ensemble_dim)
      if (context.length_dim is not None and
          context.length_dim.size > self.max_length_dim.size):
        message = (
            "Length dimenison exceeds size of positional embedding table. "
            "length_dim.size > max_length_dim.size %s vs %s."
            % (context.length_dim, self.max_length_dim))
        if context.position_is_default:
          # Definitely getting overflow in this case.
          raise ValueError(message)
        else:
          tf.logging.warning(
              message +
              " This may be OK if there are several shorter sequences packed "
              "together.  Otherwise, the later positions will get zeros.")
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
          [x * (self.model_dim.size ** -0.5), embedding_weights],
          reduced_dims=[self.model_dim])
    else:
      logits = mtf.layers.dense(
          x, self.output_vocab_dim, use_bias=False,
          variable_dtype=context.variable_dtype,
          name="logits")
    if targets is not None and context.losses is not None:
      context.losses.append(
          self._compute_loss(context, logits, targets, self.output_vocab_dim))
    if self.ensemble_dim:
      logits = reduce_ensemble_logits(
          logits, self.ensemble_dim, self.output_vocab_dim)
    return logits

  def loss_denominator(self, targets):
    """Denominator applied to losses.

    This is usually the size of the targets tensor (omitting ensemble
    dimensions).  Alternitively, it is an override value passed to the
    class constructor.

    Args:
      targets: a mtf.Tensor
    Returns:
      a float
    """
    if self._loss_denominator is not None:
      return float(self._loss_denominator)
    else:
      ret = float(targets.shape.size)
      if self.ensemble_dim:
        # The ensembling should not decrease the gradient to each model
        ret /= self.ensemble_dim.size
      return float(ret)

  def call_simple(self,
                  inputs,
                  targets,
                  compute_loss,
                  mode=tf.estimator.ModeKeys.TRAIN,
                  variable_dtype=mtf.VariableDType(tf.float32),
                  sequence_id=None,
                  subsequence_id=None,
                  position=None,
                  encoder_output=None,
                  encoder_sequence_id=None,
                  encoder_inputs=None,
                  shared_params=None,
                  layer_outputs=None,
                  encoder_layer_outputs=None):
    """Compute logits based on inputs (all positions in parallel).

    This is called during training and evaluation.

    Args:
      inputs: an int32 Tensor with shape [<batch_dims>, length_dim] For training
        autoregressive models this should be equal to mtf.shift(targets,
        offset=1, dim=length_dim, wrap=False)
      targets: an optional int32 Tensor with shape [<batch_dims>, length_dim]
      compute_loss: a boolean
      mode: a tf.estimator.ModeKeys
      variable_dtype: a mtf.VariableDType
      sequence_id: an optional Tensor
      subsequence_id: an optional Tensor
      position: an optional Tensor
      encoder_output: an optional Tensor
      encoder_sequence_id: an optional Tensor
      encoder_inputs: an optional Tensor
      shared_params: an optional dictionary
      layer_outputs: an optional list to append Tensor layer activations to
      encoder_layer_outputs: optional - readonly list of tensor activations when
        decoding, one per each input layer + the embedding layer

    Returns:
      logits: a Tensor with shape [<batch_dims>, output_vocab_dim]
      loss: an optional Scalar (if compute_loss=True)
    """
    batch_dims = inputs.shape.dims[:-1]
    length_dim = inputs.shape.dims[-1]
    length_range = mtf.range(inputs.mesh, length_dim, dtype=tf.int32)
    if not self.positional_embedding:
      # To make relative attention faster, we drop the information about the
      #   position in the subsequence.  The relative attention code then
      #   assumes that the positions are given by index in the tensor,
      #   which still leads to the correct computation of relative position.
      position = None
    if position is None:
      position_is_default = True
      position = length_range
    else:
      position_is_default = False
    if self.input_full_attention:
      # The inputs part of each sequence can fully attend within itself.
      full_attention_region = text2self_inputs_mask(targets)
      # We can include one additional position to the right - the position
      #   where the final EOS of the inputs is read and the first target token
      #   is predicted.
      full_attention_region = mtf.logical_or(
          full_attention_region,
          mtf.shift(full_attention_region, offset=1, dim=length_dim, wrap=False)
      )
      # We set read_priority and write_priority to 0 in the full-attention
      #   region and equal to the position elsewhere.
      read_priority = write_priority = length_range * mtf.cast(
          mtf.logical_not(full_attention_region), tf.int32)
    elif self.autoregressive:
      # Vanilla autoregressive model - each position can see previous positions.
      read_priority = write_priority = length_range
    else:
      read_priority = write_priority = None
    context = Context(
        model=self,
        mesh=inputs.mesh,
        batch_dims=batch_dims,
        length_dim=length_dim,
        variable_dtype=variable_dtype,
        mode=mode,
        losses=[] if compute_loss else None,
        sequence_id=sequence_id,
        subsequence_id=subsequence_id,
        position=position,
        position_is_default=position_is_default,
        encoder_output=encoder_output,
        encoder_sequence_id=encoder_sequence_id,
        shared_params=shared_params,
        layer_outputs=layer_outputs,
        encoder_layer_outputs=encoder_layer_outputs,
        write_priority=write_priority,
        read_priority=read_priority,
        inputs=inputs,
        encoder_inputs=encoder_inputs)
    with tf.variable_scope(self.name):
      logits = self._call_internal(context, inputs, targets)
    if compute_loss:
      loss = mtf.add_n(context.losses)
    else:
      loss = None
    return logits, loss

  @gin.configurable(module="Unitransformer")
  def sample_autoregressive(self,
                            partial_sequences,
                            stop_at_token=1,
                            max_steps=None,
                            temperature=0.0,
                            variable_dtype=mtf.VariableDType(tf.float32),
                            encoder_output=None,
                            encoder_sequence_id=None,
                            encoder_inputs=None,
                            shared_params=None,
                            has_partial_sequences=True,
                            encoder_layer_outputs=None,
                            never_end=False,
                            remove_partial_sequences=False,
                            sampling_keep_top_k=-1):
    """Sample randomly one token at a time.

    The partial_sequences represent partial sequences to be continued.  The
    first tokens of each sequence are nonzero representing the given partial
    sequences and the last tokens of each sequence are zeros, representing what
    needs to be filled in.

    If there are no partial sequences (you want to sample from the beginning),
    then pass partial_sequences=mtf.zeros(mesh, shape, dtype=tf.int32) and
    has_partial_sequences=False (so we can skip computation).

    Args:
      partial_sequences: an int32 Tensor with shape [<batch_dims>, length_dim]
      stop_at_token: an optional integer eos id.  Stop when we produce it.
      max_steps: an optional integer, the max number of steps to decode.
      temperature: an optional floating point value between 0.0 and 1.0 0.0
        means argmax, 1.0 means sample according to predicted distribution.
      variable_dtype: a mtf.VariableDType
      encoder_output: an optional Tensor
      encoder_sequence_id: an optional Tensor
      encoder_inputs: an optional Tensor
      shared_params: an optional dictionary
      has_partial_sequences: a boolean
      encoder_layer_outputs: optional - readonly list of tensor activations when
        decoding, one per each input layer + the embedding layer
      never_end: a boolean - if set, then avoid generating stop_at_token
      remove_partial_sequences: a boolean - whether to remove the partial
        sequences from the output
      sampling_keep_top_k: an integer - if not -1, only sample from the top k
        logits.

    Returns:
      a Tensor with shape [<batch_dims>, length_dim]
    """
    if not self.autoregressive:
      raise ValueError("must be autoregressive")

    inputs = partial_sequences
    batch_dims = inputs.shape.dims[:-1]
    length_dim = inputs.shape.dims[-1]
    initial_position = mtf.reduce_sum(
        mtf.to_int32(mtf.not_equal(inputs, 0)), reduced_dim=length_dim)
    sequence_id = 1 if encoder_sequence_id is not None else None

    length_range = mtf.range(inputs.mesh, length_dim, tf.int32)
    if self.input_full_attention:
      read_priority = write_priority = length_range * mtf.to_int32(
          mtf.greater(length_range, initial_position))
    else:
      read_priority = write_priority = length_range

    context_first_part = Context(
        model=self,
        mesh=inputs.mesh,
        batch_dims=batch_dims,
        length_dim=length_dim,
        variable_dtype=variable_dtype,
        mode="first_part",
        position=length_range,
        position_is_default=True,
        new_states=[],
        initial_position=initial_position,
        sequence_id=sequence_id,
        encoder_output=encoder_output,
        encoder_sequence_id=encoder_sequence_id,
        constant_states=[],
        shared_params=shared_params,
        encoder_layer_outputs=encoder_layer_outputs,
        write_priority=write_priority,
        read_priority=read_priority,
        inputs=inputs,
        encoder_inputs=encoder_inputs)

    shifted_inputs = mtf.shift(inputs, offset=1, dim=length_dim, wrap=False)
    with tf.variable_scope(self.name):
      logits = self._call_internal(context_first_part, shifted_inputs)
    del logits
    constant_states = context_first_part.constant_states
    if not has_partial_sequences:
      initial_states = [
          mtf.zeros_like(t) for t in context_first_part.new_states]
      partial_sequences_eos_count = 0
    else:
      initial_states = context_first_part.new_states
      partial_sequences_eos_count = mtf.reduce_sum(
          mtf.to_int32(mtf.equal(partial_sequences, stop_at_token)),
          reduced_dim=length_dim)

    def cond_fn(position, ids, *unused_states):
      """Should we run another loop iteration."""
      past_end = mtf.greater_equal(position, length_dim.size)
      if max_steps:
        past_end = mtf.logical_or(
            past_end, mtf.greater_equal(position - initial_position, max_steps))

      is_done = past_end
      if stop_at_token is not None:
        eos_count = mtf.reduce_sum(
            mtf.to_int32(mtf.equal(ids, stop_at_token)),
            reduced_dim=length_dim)
        has_additional_eos = mtf.greater(eos_count, partial_sequences_eos_count)
        is_done = mtf.logical_or(is_done, has_additional_eos)
      all_done = mtf.reduce_all(is_done)
      return mtf.logical_not(all_done)

    def body_fn(position, ids, *states):
      """One step in the decode loop."""
      inputs_this_step = mtf.gather(ids, position - 1, length_dim)
      context_incremental = Context(
          model=self,
          mesh=inputs.mesh,
          batch_dims=batch_dims,
          length_dim=length_dim,
          variable_dtype=variable_dtype,
          mode="incremental",
          position=position,
          states=states,
          new_states=[],
          sequence_id=sequence_id,
          encoder_output=encoder_output,
          encoder_sequence_id=encoder_sequence_id,
          constant_states=constant_states,
          shared_params=shared_params,
          encoder_layer_outputs=encoder_layer_outputs,
          write_priority=write_priority,
          read_priority=position,
          inputs=inputs_this_step,
          encoder_inputs=encoder_inputs)
      with tf.variable_scope(self.name, reuse=True):
        logits = self._call_internal(context_incremental, inputs_this_step)
        if never_end:
          logits += mtf.one_hot(
              mtf.constant(logits.mesh, stop_at_token, dtype=tf.int32),
              self.output_vocab_dim, on_value=-1e9, off_value=0.0,
              dtype=logits.dtype)

      # TBD whether this should be before or after never_end:
      # Note for adding top_p sampling in the future, in other code bases, the
      # option to apply temperature is done before the top-k truncation. This
      # implementation does this in the opposite order. For top-k this doesn't
      # matter, but for top_p it will.
      if sampling_keep_top_k != -1:
        if sampling_keep_top_k <= 0:
          raise ValueError("sampling_keep_top_k must either be -1 or positive.")
        k_largest = mtf.nth_largest_element(
            logits, n=sampling_keep_top_k,
            reduced_dim=self.output_vocab_dim)
        logits = mtf.where(mtf.less_equal(logits, k_largest),
                           mtf.ones_like(logits)*-1e6, logits)

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
    if has_partial_sequences and remove_partial_sequences:
      # remove partial sequences from outputs
      partial_length = mtf.reduce_sum(
          mtf.to_int32(mtf.not_equal(partial_sequences, 0)),
          reduced_dim=length_dim)
      outputs = mtf.dynamic_shift(
          outputs, -partial_length, length_dim, wrap=False)
    return outputs

  def beam_search(self,
                  inputs,
                  decode_length,
                  variable_dtype=mtf.VariableDType(tf.float32),
                  encoder_output=None,
                  encoder_sequence_id=None,
                  encoder_inputs=None,
                  alpha=0.6,
                  shared_params=None,
                  encoder_layer_outputs=None):
    """Beam search.

    Args:
      inputs: an int32 zero-Tensor with shape [<batch_dims>, beam_dim,
        length_dim].
      decode_length: an int32 mtf scalar.  Maximum decode length.
      variable_dtype: a mtf.VariableDType
      encoder_output: an optional Tensor
      encoder_sequence_id: an optional Tensor
      encoder_inputs: an optional Tensor
      alpha: a floating point value (length bonus)
      shared_params: an optional dictionary
      encoder_layer_outputs: optional - readonly list of tensor activations when
        decoding, one per each input layer + the embedding layer

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
    length_range = mtf.range(inputs.mesh, length_dim, tf.int32)
    initial_position = mtf.reduce_sum(
        mtf.to_int32(mtf.not_equal(inputs, 0)), reduced_dim=length_dim)
    sequence_id = 1 if encoder_sequence_id is not None else None

    if self.input_full_attention:
      # This only makes sense in the case of beam search with given partial
      # sequences, which is not yet implemented.
      # TODO(noam): implement
      raise NotImplementedError(
          "Beam search for language models not yet implemented")
    else:
      read_priority = write_priority = length_range

    context_first_part = Context(
        model=self,
        mesh=inputs.mesh,
        batch_dims=batch_dims + [beam_dim],
        length_dim=length_dim,
        variable_dtype=variable_dtype,
        mode="first_part",
        position=length_range,
        position_is_default=True,
        new_states=[],
        initial_position=initial_position,
        sequence_id=sequence_id,
        encoder_output=encoder_output,
        encoder_sequence_id=encoder_sequence_id,
        constant_states=[],
        shared_params=shared_params,
        encoder_layer_outputs=encoder_layer_outputs,
        write_priority=write_priority,
        read_priority=read_priority,
        inputs=inputs,
        encoder_inputs=encoder_inputs)

    shifted_inputs = mtf.shift(inputs, offset=1, dim=length_dim, wrap=False)
    with tf.variable_scope(self.name):
      logits = self._call_internal(context_first_part, shifted_inputs)
    del logits
    # There are no partial targets.
    # Replace initial states by zeros to avoid computing them.
    initial_states = [mtf.zeros_like(t) for t in context_first_part.new_states]
    constant_states = context_first_part.constant_states

    def logits_fn(step_num, ids, states):
      """logits_fn for mtf.beam_search.beam_search()."""
      inputs_this_step = mtf.gather(ids, step_num - 1, length_dim)
      context_incremental = Context(
          model=self,
          mesh=inputs.mesh,
          batch_dims=batch_dims + [beam_dim],
          length_dim=length_dim,
          variable_dtype=variable_dtype,
          mode="incremental",
          position=step_num,
          states=states,
          new_states=[],
          sequence_id=sequence_id,
          encoder_output=encoder_output,
          encoder_sequence_id=encoder_sequence_id,
          constant_states=constant_states,
          shared_params=shared_params,
          encoder_layer_outputs=encoder_layer_outputs,
          write_priority=write_priority,
          read_priority=step_num,
          inputs=inputs_this_step,
          encoder_inputs=encoder_inputs)
      with tf.variable_scope(self.name, reuse=True):
        logits = self._call_internal(context_incremental, inputs_this_step)
      return mtf.to_float(logits), context_incremental.new_states

    beams, unused_scores = mtf.beam_search.beam_search(
        logits_fn,
        inputs,
        alpha,
        states=initial_states,
        decode_length=decode_length,
        use_tpu=True,
        dtype=tf.float32,
        mesh_shape=self.mesh_shape,
        layout=self.layout)
    return mtf.gather(
        beams, mtf.constant(inputs.mesh, 0, dtype=tf.int32), beam_dim)


@gin.configurable
class Bitransformer(object):
  """A Transformer sequence-to-sequence model with two layer stacks."""

  def __init__(self, encoder, decoder, shared_embedding=True):
    """Create a Bitransformer.

    Args:
      encoder: a mtf.unitransformer
      decoder: a mtf.unitransformer
      shared_embedding: a boolean
    """
    self.encoder = encoder
    self.decoder = decoder
    self.shared_embedding = shared_embedding

  @property
  def output_vocab_dim(self):
    return self.decoder.output_vocab_dim

  def loss_denominator(self, targets):
    return self.decoder.loss_denominator(targets)

  @property
  def z_loss(self):
    return self.decoder.z_loss

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
        if not (self.encoder.model_dim == self.decoder.model_dim and
                self.encoder.input_vocab_dim == self.decoder.input_vocab_dim):
          raise ValueError(
              "shared_embedding requires encoder and decoder to have identical"
              " d_model and vocabulary sizes")
        shared_params["embedding"] = mtf.layers.embedding_weights(
            mesh,
            self.encoder.input_vocab_dim,
            self.encoder.model_dim,
            variable_dtype,
            name="embedding",
            ensemble_dim=self.encoder.ensemble_dim)
        if (self.encoder.positional_embedding
            and self.decoder.positional_embedding
            and self.encoder.max_length_dim == self.decoder.max_length_dim):
          shared_params["positional_embedding"] = mtf.layers.embedding_weights(
              mesh, self.encoder.max_length_dim, self.encoder.model_dim,
              variable_dtype, "positional_embedding",
              ensemble_dim=self.encoder.ensemble_dim)
    return shared_params

  def call_simple(self,
                  inputs,
                  targets,
                  compute_loss,
                  mode=tf.estimator.ModeKeys.TRAIN,
                  variable_dtype=mtf.VariableDType(tf.float32),
                  encoder_sequence_id=None,
                  decoder_sequence_id=None,
                  decoder_subsequence_id=None,
                  encoder_position=None,
                  decoder_position=None):
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
      decoder_subsequence_id: an optional Tensor
      encoder_position: an optional Tensor
      decoder_position: an optional Tensor

    Returns:
      logits: a Tensor with shape [<batch_dims>, output_vocab_dim]
      loss: an optional Scalar (if compute_loss=True)
    """
    encoder_layer_outputs = []
    shared_params = self._shared_params(inputs.mesh, variable_dtype)
    encoder_output, encoder_loss = self.encoder.call_simple(
        inputs,
        None,
        compute_loss,
        mode=mode,
        variable_dtype=variable_dtype,
        sequence_id=encoder_sequence_id,
        position=encoder_position,
        shared_params=shared_params,
        layer_outputs=encoder_layer_outputs)
    encoder_output = mtf.layers.rename_length_to_memory_length(encoder_output)
    if encoder_sequence_id is not None:
      encoder_sequence_id = mtf.layers.rename_length_to_memory_length(
          encoder_sequence_id)
    length_dim = targets.shape.dims[-1]
    shifted_targets = mtf.shift(targets, offset=1, dim=length_dim, wrap=False)
    # We should have a 0 at the beginning of each sequence rather than the
    # shifted EOS (1) from the previous sequence.
    shifted_targets -= mtf.to_int32(mtf.equal(shifted_targets, 1))
    logits, loss = self.decoder.call_simple(
        shifted_targets,
        targets,
        compute_loss,
        mode=mode,
        variable_dtype=variable_dtype,
        sequence_id=decoder_sequence_id,
        subsequence_id=decoder_subsequence_id,
        encoder_output=encoder_output,
        encoder_sequence_id=encoder_sequence_id,
        encoder_inputs=mtf.layers.rename_length_to_memory_length(inputs),
        position=decoder_position,
        shared_params=shared_params,
        encoder_layer_outputs=encoder_layer_outputs)
    if loss is not None and encoder_loss is not None:
      loss += encoder_loss
    return logits, loss

  @gin.configurable(module="Bitransformer")
  def decode(self,
             inputs,
             variable_dtype=mtf.VariableDType(tf.float32),
             beam_size=1,
             alpha=0.6,
             temperature=0.0,
             decode_length_multiplier=1.5,
             decode_length_constant=10,
             max_decode_length=None):
    """Sampling or beam search.

    TODO(noam): should we make the output length dimension different from the
    input length dimension?

    Args:
      inputs: a Tensor with shape [<batch_dims>, beam_dim, length_dim]
      variable_dtype: a mtf.VariableDType
      beam_size: an integer >= 1
      alpha: a floating point value (length bonus for beam search)
      temperature: a value between 0 and 1 (must be 0 if beam_size > 1)
        0.0 means argmax, 1.0 means sample according to predicted distribution.
      decode_length_multiplier: a float
      decode_length_constant: a float
      max_decode_length: an optional integer

    Returns:
      a Tensor with shape [<batch_dims>, beam_dim, length_dim]
    """
    encoder_layer_outputs = []
    shared_params = self._shared_params(inputs.mesh, variable_dtype)
    encoder_sequence_id = mtf.minimum(inputs, 1)
    encoder_output, encoder_loss = self.encoder.call_simple(
        inputs=inputs,
        targets=None,
        compute_loss=False,
        mode=tf.estimator.ModeKeys.PREDICT,
        variable_dtype=variable_dtype,
        sequence_id=encoder_sequence_id,
        shared_params=shared_params,
        layer_outputs=encoder_layer_outputs)
    del encoder_loss
    encoder_output = mtf.layers.rename_length_to_memory_length(encoder_output)
    encoder_sequence_id = mtf.layers.rename_length_to_memory_length(
        encoder_sequence_id)
    batch_dims = inputs.shape[:-1]
    length_dim = inputs.shape[-1]
    if max_decode_length is None:
      decode_length_dim = length_dim
    else:
      decode_length_dim = mtf.Dimension("length", max_decode_length)
    if beam_size == 1:
      ids_shape = mtf.Shape(batch_dims + [decode_length_dim])
      partial_sequences = mtf.zeros(inputs.mesh, ids_shape, dtype=tf.int32)
      return self.decoder.sample_autoregressive(
          partial_sequences,
          temperature=temperature,
          variable_dtype=variable_dtype,
          encoder_output=encoder_output,
          encoder_sequence_id=encoder_sequence_id,
          encoder_inputs=mtf.layers.rename_length_to_memory_length(inputs),
          shared_params=shared_params,
          has_partial_sequences=False,
          encoder_layer_outputs=encoder_layer_outputs)
    else:
      if temperature != 0:
        raise ValueError(
            "don't know how to beam search with nonzero temperature")
      # beam search
      beam_dim = mtf.Dimension("beam", beam_size)
      ids_shape = mtf.Shape(batch_dims + [beam_dim, decode_length_dim])
      partial_sequences = mtf.zeros(inputs.mesh, ids_shape, dtype=tf.int32)
      input_length = mtf.reduce_sum(
          mtf.to_float(mtf.cast(inputs, tf.bool)),
          reduced_dim=length_dim)
      max_input_length = mtf.reduce_max(input_length)
      decode_length = mtf.cast(
          max_input_length * decode_length_multiplier
          + decode_length_constant, tf.int32)
      return self.decoder.beam_search(
          partial_sequences,
          decode_length,
          variable_dtype=variable_dtype,
          encoder_output=encoder_output,
          encoder_sequence_id=encoder_sequence_id,
          encoder_inputs=inputs,
          alpha=alpha,
          shared_params=shared_params,
          encoder_layer_outputs=encoder_layer_outputs)


@gin.configurable
class StudentTeacher(object):
  """A teacher and a student to be taught via distillation."""

  def __init__(self,
               student,
               teacher,
               temperature=None,
               fraction_soft=None,
               teacher_checkpoint=None):
    """Create a StudentTeacher.

    Args:
      student: a Unitransformer or Bitransformer
      teacher: a Unitransformer or Bitransformer
      temperature: a float, the temperature of the softmax for distilling from
        the teacher. Required only when training.
      fraction_soft: a float between 0 and 1, the contribution of the soft
        target cross entropy to the training loss. The rest of the loss will be
        the cross entropy with the one-hot actual label. Required only when
        training.
      teacher_checkpoint: a string, the path to the teacher checkpoint that we
        wish to use. Required only when training.
    """
    self.student = student
    self.teacher = teacher
    self.temperature = temperature
    self.fraction_soft = fraction_soft
    self.teacher_checkpoint = teacher_checkpoint

  def call_simple(self,
                  inputs,
                  targets,
                  compute_loss,
                  variable_dtype=mtf.VariableDType(tf.float32),
                  **kargs):
    """Compute logits based on inputs (all positions in parallel).

    This is called during training and evaluation.

    Args:
      inputs: an int32 Tensor with shape [<batch_dims>, length_dim] For training
        autoregressive models this should be equal to mtf.shift(targets,
        offset=1, dim=length_dim, wrap=False)
      targets: an optional int32 Tensor with shape [<batch_dims>, length_dim]
      compute_loss: a boolean
      variable_dtype: a mtf.VariableDType
      **kargs: additional arguments to pass to the student.call_simple and
        teacher.call_simple

    Returns:
      logits: a Tensor with shape [<batch_dims>, output_vocab_dim]
      loss: an optional Scalar (if compute_loss=True)
    """
    with tf.variable_scope("student"):
      student_logits, hard_loss = self.student.call_simple(
          inputs,
          targets,
          compute_loss=True,
          variable_dtype=variable_dtype,
          **kargs)
      if not compute_loss:
        return student_logits
      elif self.fraction_soft == 0.0:
        # Do not create the teacher if we do not need it.
        return student_logits, hard_loss

    assert self.student.output_vocab_dim == self.teacher.output_vocab_dim
    assert self.student.z_loss == self.teacher.z_loss
    output_vocab_dim = self.student.output_vocab_dim
    z_loss = self.student.z_loss
    graph = inputs.mesh.graph

    with tf.variable_scope("teacher"):
      teacher_logits, _ = self.teacher.call_simple(
          inputs,
          targets,
          compute_loss=True,
          variable_dtype=variable_dtype,
          **kargs)
    graph.make_variables_untrainable(
        [v for v in graph.trainable_variables if v.name.startswith("teacher/")])

    soft_targets = mtf.softmax(teacher_logits / self.temperature,
                               output_vocab_dim)

    soft_loss = mtf.layers.softmax_cross_entropy_with_logits(
        student_logits / self.temperature,
        mtf.stop_gradient(soft_targets),
        output_vocab_dim,
        z_loss=z_loss)

    # Ignore losses from padding regions.
    weights = mtf.layers.weights_nonzero(
        targets, dtype=variable_dtype.activation_dtype)
    soft_loss = (mtf.reduce_sum(soft_loss * weights) /
                 self.student.loss_denominator(targets))

    loss = (1.0 - self.fraction_soft) * hard_loss \
           + self.temperature**2 * self.fraction_soft * soft_loss

    return student_logits, loss

  def decode(self, *args, **kargs):
    """Sample from the student.

    Args:
       *args: arguments to Unitransformer.sample_autoregressive or
         Bitransformer.decode
       **kargs: arguments to Unitransformer.sample_autoregressive or
         Bitransformer.decode

    Returns:
      a Tensor with the same shape as the output of
      Unitransformer.sample_autoregressive or Bitransformer.decode
    """
    with tf.variable_scope("student"):
      if isinstance(self.student, Unitransformer):
        return self.student.sample_autoregressive(*args, **kargs)
      elif isinstance(self.student, Bitransformer):
        return self.student.decode(*args, **kargs)
      else:
        raise ValueError("unrecognized class")

  def initialize(self):
    """Initialize the teacher model from the checkpoint.

    This function will be called after the graph has been constructed.
    """
    if self.fraction_soft == 0.0:
      # Do nothing if we do not need the teacher.
      return
    vars_to_restore = tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES, scope="teacher")
    tf.train.init_from_checkpoint(
        self.teacher_checkpoint,
        {v.name[len("teacher/"):].split(":")[0]: v for v in vars_to_restore})


# gin-configurable constructors
@gin.configurable
def make_layer_stack(layers=gin.REQUIRED, num_layers=6, block_scope=True):
  """Configurable layer stack.

  Args:
    layers: a list of subclasses of TransformerLayer, or a list of tuples. If an
      entry of this list is a tuple, the first entry of the tuple is assumed to
      be the layer name (string) and the second entry is a subclass of
      TransformerLayer.
    num_layers: an integer
    block_scope: a bool, if True then use scopes of the format
      ```
      block_000/layer_000/...
      block_000/layer_001/...
      ...
      block_001/layer_000/...
      block_001/layer_001/...
      ```
      If False then use scopes of the format
      ```
      layer_000/...
      layer_001/...
      layer_002/...
      ...
      ```
  Returns:
    a LayerStack
  """
  layer_stack = []
  for block in range(num_layers):
    for n, cls in enumerate(layers):
      # Set name to None if it wasn't provided which simplifies the logic below
      name, cls = cls if isinstance(cls, (list, tuple)) else (None, cls)
      if block_scope:
        name = "block_{:03d}/{}".format(block, name or "layer_{:03d}".format(n))
      else:
        name = name or "layer_{:03d}".format(len(layer_stack))
      layer = cls()
      layer.set_name(name)
      layer_stack.append(layer)
  return LayerStack(layer_stack)


@gin.configurable
def make_bitransformer(
    input_vocab_size=gin.REQUIRED,
    output_vocab_size=gin.REQUIRED,
    layout=None,
    mesh_shape=None,
    encoder_name="encoder",
    decoder_name="decoder"):
  """Gin-configurable bitransformer constructor.

  In your config file you need to set the encoder and decoder layers like this:
  encoder/make_layer_stack.layers = [
    @transformer_layers.SelfAttention,
    @transformer_layers.DenseReluDense,
  ]
  decoder/make_layer_stack.layers = [
    @transformer_layers.SelfAttention,
    @transformer_layers.EncDecAttention,
    @transformer_layers.DenseReluDense,
  ]

  Args:
    input_vocab_size: a integer
    output_vocab_size: an integer
    layout: optional - an input to mtf.convert_to_layout_rules
      Some layers (e.g. MoE layers) cheat by looking at layout and mesh_shape
    mesh_shape: optional - an input to mtf.convert_to_shape
      Some layers (e.g. MoE layers) cheat by looking at layout and mesh_shape
    encoder_name: optional - a string giving the Unitransformer encoder name.
    decoder_name: optional - a string giving the Unitransformer decoder name.
  Returns:
    a Bitransformer
  """
  with gin.config_scope("encoder"):
    encoder = Unitransformer(
        layer_stack=make_layer_stack(),
        input_vocab_size=input_vocab_size,
        output_vocab_size=None,
        autoregressive=False,
        name=encoder_name,
        layout=layout,
        mesh_shape=mesh_shape)
  with gin.config_scope("decoder"):
    decoder = Unitransformer(
        layer_stack=make_layer_stack(),
        input_vocab_size=output_vocab_size,
        output_vocab_size=output_vocab_size,
        autoregressive=True,
        name=decoder_name,
        layout=layout,
        mesh_shape=mesh_shape)
  return Bitransformer(encoder, decoder)


@gin.configurable
def make_bi_student_teacher(input_vocab_size=gin.REQUIRED,
                            output_vocab_size=gin.REQUIRED,
                            layout=None,
                            mesh_shape=None):
  """Gin-configurable bitransformer student teacher constructor.

  In your config file you need to set the encoder and decoder layers like this:
    encoder_layers = [
        @mesh_tensorflow.transformer.transformer_layers.SelfAttention,
        @mesh_tensorflow.transformer.transformer_layers.DenseReluDense,
    ]
    decoder_layers = [
        @mesh_tensorflow.transformer.transformer_layers.SelfAttention,
        @mesh_tensorflow.transformer.transformer_layers.EncDecAttention,
        @mesh_tensorflow.transformer.transformer_layers.DenseReluDense,
    ]
    teacher/encoder/transformer.make_layer_stack.layers = %encoder_layers
    teacher/decoder/transformer.make_layer_stack.layers = %decoder_layers
    student/encoder/transformer.make_layer_stack.layers = %encoder_layers
    student/decoder/transformer.make_layer_stack.layers = %decoder_layers

  Args:
    input_vocab_size: a integer
    output_vocab_size: an integer
    layout: optional - an input to mtf.convert_to_layout_rules Some layers (e.g.
      MoE layers) cheat by looking at layout and mesh_shape
    mesh_shape: optional - an input to mtf.convert_to_shape Some layers (e.g.
      MoE layers) cheat by looking at layout and mesh_shape

  Returns:
    a StudentTeacher
  """
  with gin.config_scope("student"):
    student = make_bitransformer(
        input_vocab_size=input_vocab_size,
        output_vocab_size=output_vocab_size,
        layout=layout,
        mesh_shape=mesh_shape)
  with gin.config_scope("teacher"):
    teacher = make_bitransformer(
        input_vocab_size=input_vocab_size,
        output_vocab_size=output_vocab_size,
        layout=layout,
        mesh_shape=mesh_shape)
  return StudentTeacher(student=student, teacher=teacher)


def _round_up_to_multiple(n, divisor):
  return n + -n % divisor


def text2self_inputs_mask(ids, eos_id=1):
  """Binary mask indicating which parts of the ids represent the inputs.

  Assumes that the ids consist of packed sequences where each example is
  represented by two eos-terminated sequences, i.e.
  [<inputs0>, EOS, <targets0>, EOS, <inputs1>, EOS, <targets1>, EOS ...]

  As such, the inputs are the parts where the number of previous EOS tokens
  is even.

  Args:
    ids: an int32 mtf.Tensor with shape [..., length_dim]
    eos_id: an integer
  Returns:
    a boolean mtf.Tensor with the same shape as ids
  """
  length_dim = ids.shape.dims[-1]
  return mtf.equal(mtf.mod(mtf.cumsum(mtf.to_int32(mtf.equal(ids, eos_id)),
                                      length_dim, exclusive=True), 2), 0)


@gin.configurable
def reduce_ensemble_logits_select(logits, ensemble_dim, vocab_dim, model_id=0):
  """Select logits from the model_id-th element of the ensemble."""
  del vocab_dim
  return mtf.gather(logits, model_id % ensemble_dim.size, ensemble_dim)


@gin.configurable
def reduce_ensemble_logits_mean_prob(logits, ensemble_dim, vocab_dim):
  """Probabilities equal to arithmetic mean probability across models."""
  probs = mtf.softmax(logits, reduced_dim=vocab_dim)
  probs = mtf.reduce_mean(probs, reduced_dim=ensemble_dim)
  return mtf.log(mtf.maximum(probs, 1e-20))


@gin.configurable
def reduce_ensemble_logits_mean_logit(logits, ensemble_dim, vocab_dim):
  """Probabilities proportional to geometric mean probability across models."""
  del vocab_dim
  return mtf.reduce_mean(logits, reduced_dim=ensemble_dim)


@gin.configurable
def reduce_ensemble_logits(logits, ensemble_dim, vocab_dim,
                           reduce_fn=reduce_ensemble_logits_mean_prob):
  """Configurable reduction function for decoding from an ensemble.

  reduce_fn is a function which takes:
     a logits tensor containing ensemble_dim (logits from all models)
     ensemble_dim
     vocab_dim
  and returns a logits tensor without ensemble_dim.

  Args:
    logits: a mtf.Tensor containing ensemble_dim
    ensemble_dim: a mtf.Dimension
    vocab_dim: a mtf.Dimension
    reduce_fn: a function
  Returns:
    a mtf.Tensor with shape logits.shape - ensemble_dim
  """
  return reduce_fn(logits, ensemble_dim, vocab_dim)


