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

"""The implementation of a Funnel Transformer in Mesh TensorFlow."""

import gin

import mesh_tensorflow as mtf
from mesh_tensorflow.transformer import transformer
import tensorflow.compat.v1 as tf


@gin.configurable
class FunnelTransformerLayerStack(transformer.TransformerLayer):
  """A stack of layers for FunnelTransformer."""

  def __init__(self,
               layers,
               n_blocks=gin.REQUIRED,
               block_param_size=gin.REQUIRED,
               block_repeat_size=gin.REQUIRED,
               pooling_size=2,
               sublayers_initial=None,
               sublayers_per_layer=None,
               sublayers_final=None,
               pooling_type="mean",
               n_submodules=2):
    """Create a LayerStack specialized for FunnelTransformer.

    The design of this class follows the transformer.LayerStack. See the
    docstring of that class for how the layer stack is built. Here we only
    discuss the features unique to the Funnel Transformer.

    This implementation has subtle difference from the Funnel Transformer
    introduced in https://arxiv.org/abs/2006.03236.

    1. Application to encoder-decoder model.
    The original Funnel Transformer was proposed for the encoder-only
    architectures such as BERT. In A.3 section of the paper, they discuss
    potential extension of the core idea to other model architectures. For
    encoder-decoder models, the authors suggest that the Funnel Transformer idea
    can be used to modify the encoder such that "the key difference compared to
    conventional models is the source side compression Funnel-Transformer
    provides".

    Therefore, we don't modify the decoder, i.e., we use the standard
    transformer.LayerStack and this class is only applicable to the encoder.

    2. Relative attention
    We use the simplified reletive attention scalar from the T5, whereas the
    Funnel Transformer paper uses the relative attention from the Transformer-XL
    (https://arxiv.org/abs/1901.02860).

    3. The order of pooling operation
    In the Funnel Transformer paper, only the query is pooled while key and
    value are kept intact. The resulting attention output has the same length as
    the query, enabling the residual connection.

    In our implementation, we apply the regular SelfAttention and then apply the
    pooling to the output. Since each sequence position in query is
    independently computed, we expect the difference between these
    implmentations to be negligible.

    Args:
      layers: a list of TransformerLayer
      n_blocks: an integer specifying the number of Funnel Transformer blocks.
      block_param_size: a list of integers specifying the number of layers in
        each block.
      block_repeat_size: a list of integers specifying the number of repeated
        layers in each block. The repeated layers share the parameters.
      pooling_size: an integer specifying the pool size
      sublayers_initial: an optional list of sublayer functions
      sublayers_per_layer: an optional list of sublayer functions
      sublayers_final: an optional list of sublayer functions
      pooling_type: a string specifying the pooling type. One of "mean", "max",
        or "min".
      n_submodules: an integer specifying the number of submodules (e.g.,
        SelfAttention and DenseReluDense for each layer of a block.
    """
    if len(block_param_size) != n_blocks:
      raise ValueError(
          "Number of blocks should match the length of block_param_size.")

    if len(block_repeat_size) != n_blocks:
      raise ValueError(
          "Number of blocks should match the length of block_repeat_size.")

    if len(layers) != sum(block_param_size) * n_submodules:
      raise ValueError(
          "Total number of submodules should match the number of layers.")

    self._layers = layers
    self.n_blocks = n_blocks
    self.block_param_size = block_param_size
    self.block_repeat_size = block_repeat_size
    self.pooling_size = pooling_size

    self._sublayers_initial = sublayers_initial
    self._sublayers_per_layer = sublayers_per_layer
    self._sublayers_final = sublayers_final

    if pooling_type == "mean":
      self.pool_fn = mtf.reduce_mean
    elif pooling_type == "max":
      self.pool_fn = mtf.reduce_max
    elif pooling_type == "min":
      self.pool_fn = mtf.reduce_min
    else:
      raise ValueError(
          "Unknown pooling type. Choose among 'mean', 'max' or 'min'")
    self.n_submodules = n_submodules

  def update_context(self, context, x, pool_dim_name):
    """Update the length dimension, sequence_id and position information."""

    pooled_seq_length = x.shape.get_dim_by_name(pool_dim_name).size
    # For position, we slice the first `pooled_seq_length` indices instead of
    # striding. This ensures that the 3rd position before the pooling becomes
    # 2nd position after pooling instead of remembering its position before
    # pooling.
    new_context_position = mtf.slice(
        context.position,
        begin=0,
        size=pooled_seq_length,
        slice_dim_name=pool_dim_name)
    context.position = new_context_position

    pooled_seq_length = x.shape.get_dim_by_name(pool_dim_name).size
    new_length_dim = mtf.Dimension(
        name=pool_dim_name, size=pooled_seq_length)

    new_sequence_id = mtf.stride_tensor_1d(
        context.sequence_id,
        pool_dim=context.length_dim,
        pool_size=self.pooling_size)

    context.length_dim = new_length_dim
    context.sequence_id = new_sequence_id

  def call(self, context, x):
    """Call the layer stack."""
    x = self._call_sublayers(self._sublayers_initial, x, context)
    context.layer_outputs.append(x)

    assert context.layer_index == 0

    for block_idx in range(self.n_blocks):
      for param_idx in range(self.block_param_size[block_idx]):
        # Number of layers to (locally) share parameters.
        cur_repeat_size = self.block_repeat_size[block_idx]
        for repeat_idx in range(cur_repeat_size):
          # context.do_pooling = block_idx > 0 and sub_idx == 0

          # Submodules are transformer.TransformerLayer objects such as
          # SelfAttention and DenseReluDense.
          for submodule_idx in range(self.n_submodules):
            layer = self._layers[context.layer_index]
            name = (f"funnel_block_{block_idx:03d}/"
                    f"param_idx_{param_idx:03d}/"
                    f"submodule_{submodule_idx:03d}")
            # Override the layer name given in transformer.make_layer_stack.
            layer.set_name(name)

            with tf.variable_scope(layer.name or ""):
              x = self._layer_fn(x, layer, context)

            # Do pooling if the current layer
            # 1) does not belong to the first block
            # 2) is the first layer within the current block
            # 3) is the first submodule (typically SelfAttention).
            sub_idx = (param_idx * cur_repeat_size + repeat_idx)
            if block_idx > 0 and sub_idx == 0 and submodule_idx == 0:
              x = mtf.pool_tensor_1d(
                  x,
                  pool_dim=context.length_dim,
                  reduce_fn=self.pool_fn,
                  pool_size=self.pooling_size)
              self.update_context(context, x, pool_dim_name="length")

            if context.layer_index != len(self._layers) - 1:
              context.layer_outputs.append(x)
            context.layer_index += 1

    x = self._call_sublayers(self._sublayers_final, x, context)
    x = transformer.sublayer_mask_padding(x, self, context)
    context.layer_outputs.append(x)
    self.set_context(context)
    return x

  def _call_sublayers(self, sublayers, x, context):
    for s in sublayers:
      x = s(x, self, context)
    return x

  def _layer_fn(self, x, layer, context):
    """Call the layer and its associated sublayers.

    Args:
      x: a Tensor
      layer: a Layer
      context: a Context
    Returns:
      a Tensor
    """
    context.current_layer = layer
    context.current_layer_input = x
    y = self._call_sublayers(self._sublayers_per_layer, x, context)

    # When pooling is done, context.current_layer_input will be updated inside
    # SelfAttentionPoolQ.call method, i.e., x != context.current_layer_input. So
    # we use context.current_layer_input to check the shape consistency.
    if y.shape != context.current_layer_input.shape:
      raise ValueError(
          "Layer %s returned misshaped output x=%s y=%s"
          % (layer.__class__.__name__, x, y))
    return y

  @property
  def layers(self):
    return self._layers

  def set_context(self, context):
    self._context = context

  @property
  def context(self):
    return self._context


@gin.configurable
class BitransformerFunnel(transformer.Bitransformer):
  """Bitransformer with the compressed sequence length in the encoder.

  See base class for details.

  This class updates the encoder's information passed to the decoder in order to
  account for the reduced sequence length.
  """

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
                  decoder_position=None,
                  num_microbatches=1):
    """Compute logits based on inputs (all positions in parallel).

    This is called during training and evaluation.

    This class inherits the trnasformer.Bitransformer with one difference. The
    encoder is Funnel Transformer. So the length dimension is reduced. The
    decoder needs to use the updated `encoder_sequence_id`.

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
      num_microbatches: integer - greater than one if the step has been
        serialized into multiple microbatches to save memory.

    Returns:
      logits: a Tensor with shape [<batch_dims>, output_vocab_dim]
      loss: an optional Scalar (if compute_loss=True)
    """
    # encoder_sequene_id and decoder_sequence_id are used to delineate packed
    # examples but are also necessary to indicate padding where sequence_id==0.
    # If they are absent, then we assume that padding is indicated by zeros in
    # the inputs/targets, and we make up sequence_id tensors to indicate this.
    if encoder_sequence_id is None:
      encoder_sequence_id = mtf.minimum(inputs, 1)
    if decoder_sequence_id is None:
      decoder_sequence_id = mtf.minimum(targets, 1)
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
        layer_outputs=encoder_layer_outputs,
        num_microbatches=num_microbatches)
    encoder_output = mtf.layers.rename_length_to_memory_length(encoder_output)

    # The sequence_id is updated inside the layer_stack due to pooling. So we
    # need to use the updated sequence_id stored in the context.
    encoder_sequence_id = self.encoder.layer_stack.context.sequence_id
    encoder_sequence_id = mtf.layers.rename_length_to_memory_length(
        encoder_sequence_id)

    logits, loss = self.decoder.call_simple(
        transformer.autoregressive_inputs(
            targets, sequence_id=decoder_sequence_id),
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
        encoder_layer_outputs=encoder_layer_outputs,
        num_microbatches=num_microbatches)
    if loss is not None and encoder_loss is not None:
      loss += encoder_loss
    return logits, loss
