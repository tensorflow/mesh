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
"""MeshTensorFlow implementation of BERT.

The code is ported from https://github.com/google-research/bert.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import json
import math
import re
import mesh_tensorflow as mtf
import mesh_tensorflow.transformer.moe as moe
import six

import tensorflow.compat.v1 as tf


class BertConfig(object):
  """Configuration for `BertModel`."""

  def __init__(self,
               vocab_size,
               d_model=768,
               position_signal="embedding",
               max_position_embeddings=512,
               num_blocks=12,
               block_layers="attention,feedforward",
               layer_output_dropout_prob=0.1,
               residual_structure="original",
               use_bias=True,
               attention_num_heads=12,
               attention_head_size=None,
               attention_num_key_heads=None,
               attention_key_head_size=None,
               attention_num_value_heads=None,
               attention_value_head_size=None,
               attention_probs_dropout_prob=0.1,
               feedforward_intermediate_size=3072,
               feedforward_intermediate_act="gelu",
               feedforward_intermediate_dropout_prob=0.0,
               moe_num_experts=32,
               moe_intermediate_size=6144,
               type_vocab_size=16,
               initializer_range=0.02):
    """Constructs BertConfig.

    residual_structure="original"
       TODO(noam): describe
    residual_structure="direct"
       TODO(noam): describe


    Args:
      vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
      d_model: Number of channels in input/output of each layer.
      position_signal: A string specifying the type of position signal.
        Implemented values are "embedding", "relative_attention_bias".
      max_position_embeddings: For models using positional embeddings,
        this is the maximum sequence length that this model might
        ever be used with. Typically set this to something large just in case
        (e.g., 512 or 1024 or 2048).
      num_blocks: Number of (attention+feed-forward) blocks in the Transformer
         encoder.
      block_layers: a comma-separated string specifying the sequence of layers
        in each block.
      layer_output_dropout_prob: The dropout probability for the output of
        each layer.
      residual_structure: a string.  Legal values are "original" and "direct".
      use_bias: a boolean - If true, then we use biases for dense layers and
        in layer normalization, and subtract off the mean in layer
        normalization.
      attention_num_heads: Number of attention heads for each attention layer in
        the Transformer encoder.
      attention_head_size: Size of attention keys and values.  If set to None,
        a default value is used equal to (d_model / attention_num_heads)
      attention_num_key_heads: Number of attention key heads.
      attention_key_head_size: Size of attention keys.
      attention_num_value_heads: Number of attention value heads.
      attention_value_head_size: Size of attention values.
      attention_probs_dropout_prob: The dropout ratio for the attention
        probabilities.
      feedforward_intermediate_size: The size of the "intermediate" layer in the
         feed-forward layer in the Transformer encoder (a.k.a. d_ff).
      feedforward_intermediate_act: The non-linear activation function
        (function or string) applied to the feedforward intermediate layer
        and the pooler layer.
      feedforward_intermediate_dropout_prob: The dropout probability for
        feed-forward intermediate layer.
      moe_num_experts: an integer - number of experts in moe layer
      moe_intermediate_size: an integer - size of intermediate layer in each
        expert
      type_vocab_size: The vocabulary size of the `token_type_ids` passed into
        `BertModel`.
      initializer_range: The stdev of the truncated_normal_initializer for
        initializing all weight matrices.
    """
    self.vocab_size = vocab_size
    self.d_model = d_model
    self.position_signal = position_signal
    self.max_position_embeddings = max_position_embeddings
    self.num_blocks = num_blocks
    self.block_layers = block_layers.split(",")
    self.layer_output_dropout_prob = layer_output_dropout_prob
    self.residual_structure = residual_structure
    self.use_bias = use_bias
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.attention_num_heads = attention_num_heads
    self.attention_head_size = attention_head_size
    self.attention_num_key_heads = attention_num_key_heads
    self.attention_key_head_size = attention_key_head_size
    self.attention_num_value_heads = attention_num_value_heads
    self.attention_value_head_size = attention_value_head_size
    self.feedforward_intermediate_size = feedforward_intermediate_size
    self.feedforward_intermediate_act = feedforward_intermediate_act
    self.feedforward_intermediate_dropout_prob = (
        feedforward_intermediate_dropout_prob)
    self.moe_num_experts = moe_num_experts
    self.moe_intermediate_size = moe_intermediate_size
    self.type_vocab_size = type_vocab_size
    self.initializer_range = initializer_range
    if self.position_signal not in ["embedding", "relative_attention_bias"]:
      raise ValueError("unknown position_signal")
    if self.residual_structure not in ["original", "direct"]:
      raise ValueError("unknown residual_structure")

  @classmethod
  def from_dict(cls, json_object):
    """Constructs a `BertConfig` from a Python dictionary of parameters."""
    # Dictionary for compatibility for tf BertConfig files.
    hparam_name_conversion = {
        "hidden_size": "d_model",
        "num_hidden_layers": "num_blocks",
        "num_attention_heads": "attention_num_heads",
        "intermediate_size": "feedforward_intermediate_size",
        "hidden_act": "feedforward_intermediate_act",
        "hidden_dropout_prob": "layer_output_dropout_prob",
    }
    config = BertConfig(vocab_size=None)
    for (key, value) in six.iteritems(json_object):
      config.__dict__[hparam_name_conversion.get(key, key)] = value
    return config

  @classmethod
  def from_json_file(cls, json_file):
    """Constructs a `BertConfig` from a json file of parameters."""
    with tf.gfile.GFile(json_file, "r") as reader:
      text = reader.read()
    return cls.from_dict(json.loads(text))

  def to_dict(self):
    """Serializes this instance to a Python dictionary."""
    output = copy.deepcopy(self.__dict__)
    return output

  def to_json_string(self):
    """Serializes this instance to a JSON string."""
    return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class BertModel(object):
  """BERT model ("Bidirectional Encoder Representations from Transformers")."""

  def __init__(self,
               config,
               is_training,
               input_ids,
               input_mask=None,
               token_type_ids=None,
               scope=None,
               mesh_shape="",
               layout=""):
    self.config = copy.deepcopy(config)
    del config
    if not is_training:
      self.config.layer_output_dropout_prob = 0.0
      self.config.attention_probs_dropout_prob = 0.0
      self.config.feedforward_intermediate_dropout_prob = 0.0
    input_shape = input_ids.shape
    assert input_shape.ndims == 2

    self._seq_dim = input_shape.dims[1]
    self._memory_seq_dim = mtf.Dimension("memory_seq", self.seq_dim.size)
    self._extra_losses = []
    mesh = input_ids.mesh

    if token_type_ids is None:
      token_type_ids = mtf.zeros(mesh, input_shape, dtype=tf.int32)

    with tf.variable_scope(scope, default_name="bert"):
      with tf.variable_scope("embeddings"):
        # Perform embedding lookup on the word ids.
        self.embedding_table = mtf.get_variable(
            mesh, "word_embeddings",
            mtf.Shape([self.vocab_dim, self.model_dim]),
            initializer=self.embedding_initializer)
        self.word_embedding_output = mtf.gather(
            self.embedding_table, input_ids, self.vocab_dim)

        # Add positional embeddings and token type embeddings, then layer
        # normalize and perform dropout.
        self.embedding_output = self.word_embedding_output

        token_type_table = mtf.get_variable(
            mesh, "token_type_embeddings",
            mtf.Shape([self.token_type_vocab_dim, self.model_dim]),
            initializer=self.embedding_initializer)
        if token_type_ids is not None:
          self.embedding_output += mtf.gather(
              token_type_table, token_type_ids, self.token_type_vocab_dim)
        if self.config.position_signal == "embedding":
          full_position_table = mtf.get_variable(
              mesh, "position_embeddings",
              mtf.Shape([self.max_position_embeddings_dim, self.model_dim]),
              initializer=self.embedding_initializer)
          short_position_table = mtf.rename_dimension(
              mtf.slice(full_position_table, 0, self.seq_dim.size,
                        self.max_position_embeddings_dim.name),
              self.max_position_embeddings_dim.name, self.seq_dim.name)
          self.embedding_output += short_position_table
        self.embedding_output = self.normalize(self.embedding_output)
        self.embedding_output = mtf.dropout(
            self.embedding_output,
            keep_prob=1.0 - self.config.layer_output_dropout_prob)

      with tf.variable_scope("encoder"):
        attention_biases = []
        if input_mask:
          # [batch_dim, memory_seq_dim]
          attention_biases.append(
              (1.0 - mtf.to_float(mtf.replace_dimensions(
                  input_mask, self.seq_dim, self.memory_seq_dim))) * -10000.0)
        if self.config.position_signal == "relative_attention_bias":
          buckets_dim = mtf.Dimension("buckets", 32)
          rp_bucket = _relative_position_bucket(
              mtf.range(mesh, self.memory_seq_dim, tf.int32)
              - mtf.range(mesh, self.seq_dim, tf.int32),
              num_buckets=buckets_dim.size)
          bias_var = mtf.get_variable(
              mesh, "relative_attention_bias",
              [self.num_heads_dim, buckets_dim],
              initializer=tf.zeros_initializer())
          attention_biases.append(mtf.gather(bias_var, rp_bucket, buckets_dim))
        attention_bias = mtf.add_n(attention_biases)
        prev_layer_output = self.embedding_output
        self.all_encoder_layers = []
        for block_num in range(self.config.num_blocks):
          with tf.variable_scope("block_%d" % block_num):
            for layer_idx, layer_type in enumerate(self.config.block_layers):
              layer_name = layer_type
              count = self.config.block_layers[:layer_idx].count(layer_type)
              if count:
                layer_name += "_%d" % count
              with tf.variable_scope(layer_name):
                x = prev_layer_output
                if self.config.residual_structure == "direct":
                  x = self.normalize(x)
                if layer_type == "attention":
                  x = self.self_attention(x, attention_bias)
                elif layer_type == "feedforward":
                  x = self.feedforward(x)
                elif layer_type == "moe":
                  x = self.moe(x, layout, mesh_shape, input_mask, is_training)
                else:
                  raise ValueError("unknown layer type " + layer_type)
                x = mtf.dropout(
                    x, keep_prob=1.0 - self.config.layer_output_dropout_prob)
                layer_output = prev_layer_output + x
                if self.config.residual_structure == "original":
                  layer_output = self.normalize(layer_output)
                prev_layer_output = layer_output
          self.all_encoder_layers.append(layer_output)

      self.sequence_output = prev_layer_output
      if self.config.residual_structure == "direct":
        self.sequence_output = self.normalize(self.sequence_output)

      # The "pooler" converts the encoded sequence tensor of shape
      # [batch_dim, seq_dim, hidden_size] to a tensor of shape
      # [batch_dim, hidden_size]. This is necessary for segment-level
      # (or segment-pair-level) classification tasks where we need a fixed
      # dimensional representation of the segment.
      with tf.variable_scope("pooler"):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token. We assume that this has been pre-trained
        first_token_tensor = mtf.gather(self.sequence_output, 0, self.seq_dim)
        self.pooled_output = mtf.layers.dense(
            first_token_tensor,
            reduced_dims=[self.model_dim],
            new_dims=[self.model_dim],
            activation=mtf.tanh,
            kernel_initializer=self.dense_initializer,
            use_bias=self.config.use_bias)

  def self_attention(self, x, attention_bias):
    """Performs multi-headed self-attention with output projection.

    Args:
      x: output of previous layer
      attention_bias: optional float32 Tensor broadcastable to shape
        x.shape - self.model_dim + self.memory_seq_dim
        to be added to attention logits.
        This may used to mask out padding regions of the memory.

    Returns:
      float Tensor with the same shape as x
    """

    queries = mtf.layers.dense(
        x,
        reduced_dims=[self.model_dim],
        new_dims=[self.num_heads_dim, self.size_per_head_dim],
        kernel_initializer=self.dense_initializer,
        name="query",
        use_bias=self.config.use_bias)
    keys = mtf.layers.dense(
        mtf.replace_dimensions(x, self.seq_dim, self.memory_seq_dim),
        reduced_dims=[self.model_dim],
        new_dims=[self.num_heads_dim, self.size_per_head_dim],
        kernel_initializer=self.dense_initializer,
        name="key",
        use_bias=self.config.use_bias)
    values = mtf.layers.dense(
        mtf.replace_dimensions(x, self.seq_dim, self.memory_seq_dim),
        reduced_dims=[self.model_dim],
        new_dims=[self.num_heads_dim, self.size_per_head_dim],
        kernel_initializer=self.dense_initializer,
        name="value",
        use_bias=self.config.use_bias)

    # Take the dot product between "query" and "key" to get the raw
    # attention scores.
    attention_scores = mtf.einsum(
        [queries, keys], reduced_dims=[self.size_per_head_dim])
    attention_scores *= self.size_per_head_dim.size ** -0.5

    if attention_bias is not None:
      attention_scores += attention_bias

    # Normalize the attention scores to probabilities.
    attention_probs = mtf.softmax(attention_scores, self.memory_seq_dim)

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_probs = mtf.dropout(
        attention_probs,
        keep_prob=1.0 - self.config.attention_probs_dropout_prob)

    output = mtf.einsum([attention_probs, values],
                        reduced_dims=[self.memory_seq_dim])

    # linear transformation back to shape of query_antecedent
    output = mtf.layers.dense(
        output,
        reduced_dims=[self.num_heads_dim, self.size_per_head_dim],
        new_dims=[self.model_dim],
        kernel_initializer=self.dense_initializer,
        name="output",
        use_bias=self.config.use_bias)
    output = mtf.transpose(output, x.shape)
    return output


  def feedforward(self, x):
    intermediate = mtf.layers.dense(
        x, reduced_dims=[self.model_dim],
        new_dims=[self.feedforward_intermediate_dim],
        activation=get_activation(self.config.feedforward_intermediate_act),
        kernel_initializer=self.dense_initializer,
        name="dense_1", use_bias=self.config.use_bias)
    return mtf.layers.dense(
        intermediate,
        reduced_dims=[self.feedforward_intermediate_dim],
        new_dims=[self.model_dim],
        kernel_initializer=self.dense_initializer,
        name="dense_2", use_bias=self.config.use_bias)

  def moe(self, x, layout, mesh_shape, input_mask, is_training):
    """Mixture of experts layer.

    TODO(noam): clean up the mixture-of-experts code in Transformer.

    Args:
      x: layer input
      layout: a mtf.LayoutRules
      mesh_shape: a mtf.Shape
      input_mask: a mtf.Tensor
      is_training: a boolean
    Returns:
      a mtf.Tensor (the layer output)
    """
    hparams = moe.HParams(
        moe_gating="top_2",
        moe_num_experts=self.config.moe_num_experts,
        moe_loss_coef=1e-3,
        moe_hidden_size=self.config.moe_intermediate_size,
        moe_group_size=2048,
        moe_capacity_factor_train=1.25,
        moe_capacity_factor_eval=8.0,
        moe_use_second_place_loss=False,
        moe_second_policy_train="random",
        moe_second_policy_eval="random",
        moe_second_threshold_train=0.2,
        moe_second_threshold_eval=0.2)
    layer_output, loss = moe.transformer_moe_layer_v1(
        inputs=x,
        output_dim=self.model_dim,
        hparams=hparams,
        train=is_training,
        variable_dtype=tf.float32,
        layout=layout,
        mesh_shape=mesh_shape,
        nonpadding=(mtf.cast(input_mask, tf.float32) if input_mask else None),
        activation=get_activation(self.config.feedforward_intermediate_act))
    self._extra_losses.append(loss)
    return layer_output

  def get_masked_lm_output(self, positions, label_ids, label_weights):
    """Get loss and logits for the masked LM."""
    input_tensor = self.get_sequence_output()
    output_weights = self.get_embedding_table()

    # [batch_size, num_position, hidden]
    input_tensor = mtf.gather(input_tensor, positions, self.seq_dim)
    with tf.variable_scope("cls/predictions"):
      # We apply one more non-linear transformation before the output layer.
      # This matrix is not used after pre-training.
      with tf.variable_scope("transform"):
        input_tensor = mtf.layers.dense(
            input_tensor,
            reduced_dims=[self.model_dim],
            new_dims=[self.model_dim],
            activation=get_activation(self.config.feedforward_intermediate_act),
            kernel_initializer=self.dense_initializer,
            use_bias=self.config.use_bias)
        input_tensor = self.normalize(input_tensor)
      # The output weights are the same as the input embeddings, but there is
      # an output-only bias for each token.
      output_bias = mtf.get_variable(
          input_tensor.mesh,
          name="output_bias",
          shape=[self.vocab_dim],
          initializer=tf.zeros_initializer())
      logits = mtf.einsum([input_tensor, output_weights],
                          reduced_dims=[self.model_dim]) + output_bias
      per_example_loss = mtf.layers.softmax_cross_entropy_with_logits(
          logits, label_ids, self.vocab_dim, z_loss=1e-4)
      # The `positions` tensor might be zero-padded (if the sequence is too
      # short to have the maximum number of predictions). The `label_weights`
      # tensor has a value of 1.0 for every real prediction and 0.0 for the
      # padding predictions.
      numerator = mtf.reduce_sum(label_weights * per_example_loss)
      denominator = mtf.reduce_sum(label_weights) + mtf.constant(
          input_tensor.mesh, 1e-5, dtype=tf.float32)
      loss = numerator / denominator
    return (loss, per_example_loss, logits)

  def get_next_sentence_output(self, labels):
    """Get loss and logits for the next sentence prediction."""
    class_dim = mtf.Dimension("class", 2)
    input_tensor = self.get_pooled_output()
    # Simple binary classification. Note that 0 is "next sentence" and 1 is
    # "random sentence". This weight matrix is not used after pre-training.
    logits = mtf.layers.dense(
        input_tensor,
        reduced_dims=[self.model_dim],
        new_dims=[class_dim],
        kernel_initializer=self.dense_initializer,
        name="cls/seq_relationship",
        use_bias=self.config.use_bias)
    per_example_loss = mtf.layers.softmax_cross_entropy_with_logits(
        logits, labels, class_dim, z_loss=1e-4)
    loss = mtf.reduce_mean(per_example_loss)
    return (loss, per_example_loss, logits)

  def get_pooled_output(self):
    return self.pooled_output

  def get_sequence_output(self):
    """Gets final hidden layer of encoder.

    Returns:
      float Tensor of shape [batch_dim, seq_dim, model_dim] corresponding
      to the final hidden of the transformer encoder.
    """
    return self.sequence_output

  def get_all_encoder_layers(self):
    return self.all_encoder_layers

  def get_word_embedding_output(self):
    """Get output of the word(piece) embedding lookup.

    This is BEFORE positional embeddings and token type embeddings have been
    added.

    Returns:
      float Tensor of shape [batch_dim, seq_dim, model_dim] corresponding
      to the output of the word(piece) embedding layer.
    """
    return self.word_embedding_output

  def get_embedding_output(self):
    """Gets output of the embedding lookup (i.e., input to the transformer).

    Returns:
      float Tensor of shape [batch_dim, seq_dim, model_dim] corresponding
      to the output of the embedding layer, after summing the word
      embeddings with the positional embeddings and the token type embeddings,
      then performing layer normalization. This is the input to the transformer.
    """
    return self.embedding_output

  def normalize(self, x):
    return layer_norm(x, self.model_dim,
                      subtract_mean=self.config.use_bias,
                      use_bias=self.config.use_bias)

  def get_embedding_table(self):
    return self.embedding_table

  def get_extra_loss(self):
    return mtf.add_n(self._extra_losses)

  @property
  def vocab_dim(self):
    # pad vocab to a multiple of 128 so as to be splittable.
    # TODO(noam): This creates issues in checkpoint compatibility
    n = self.config.vocab_size
    return mtf.Dimension("vocab", n + (-n % 128))

  @property
  def model_dim(self):
    return mtf.Dimension("hidden", self.config.d_model)

  @property
  def token_type_vocab_dim(self):
    return mtf.Dimension("token_type_vocab", self.config.type_vocab_size)

  @property
  def feedforward_intermediate_dim(self):
    return mtf.Dimension("intermediate",
                         self.config.feedforward_intermediate_size)

  @property
  def num_heads_dim(self):
    return mtf.Dimension("num_heads", self.config.attention_num_heads)

  @property
  def softmax_heads_dims(self):
    return self.num_heads_dim

  @property
  def max_position_embeddings_dim(self):
    return mtf.Dimension("max_position_embeddings",
                         self.config.max_position_embeddings)

  @property
  def seq_dim(self):
    return self._seq_dim

  @property
  def memory_seq_dim(self):
    return self._memory_seq_dim

  @property
  def dense_initializer(self):
    if self.config.initializer_range:
      return tf.truncated_normal_initializer(
          stddev=self.config.initializer_range)
    else:
      return mtf.layers.VarianceScalingInitializer(scale=0.4)

  @property
  def embedding_initializer(self):
    initializer = self.dense_initializer
    if isinstance(initializer, mtf.layers.DenseInitializer):
      # embedding matrix is also used as classifier weight matrix.
      # scale it appropriately.
      return initializer(
          reduced_dims=[self.model_dim], new_dims=[self.vocab_dim])
    else:
      return initializer

  @property
  def size_per_head_dim(self):
    """Dimensionality of attention queries/keys/values."""
    if self.config.attention_head_size:
      attention_head_size = self.config.attention_head_size
    else:
      if self.model_dim.size % self.num_heads_dim.size != 0:
        raise ValueError(
            "The hidden size (%d) is not a multiple of the number of attention "
            "heads (%d)" % (self.model_dim.size, self.num_heads_dim.size))
      attention_head_size = int(self.model_dim.size / self.num_heads_dim.size)
    return mtf.Dimension("attention_head", attention_head_size)

  @property
  def key_dim(self):
    """Dimensionality of attention key."""
    if self.config.attention_key_head_size is None:
      raise ValueError("The key head size is not defined.")
    return mtf.Dimension("d_k", self.config.attention_key_head_size)

  @property
  def key_heads_dims(self):
    """Dimensionality of number of key heads."""
    if self.config.attention_num_key_heads is None:
      raise ValueError("The number of key heads is not defined.")
    return mtf.Dimension("key_heads", self.config.attention_num_key_heads)

  @property
  def value_dim(self):
    """Dimensionality of attention value."""
    if self.config.attention_value_head_size is None:
      raise ValueError("The value head size is not defined.")
    return mtf.Dimension("d_v", self.config.attention_value_head_size)

  @property
  def value_heads_dims(self):
    """Dimensionality of number of value heads."""
    if self.config.attention_num_value_heads is None:
      raise ValueError("The number of value heads is not defined.")
    return mtf.Dimension("value_heads", self.config.attention_num_value_heads)


def get_activation(activation_string):
  """Maps a string to a Python function, e.g., "relu" => `mtf.relu`.

  Args:
    activation_string: String name of the activation function.

  Returns:
    A Python function corresponding to the activation function. If
    `activation_string` is None, empty, or "linear", this will return None.
    If `activation_string` is not a string, it will return `activation_string`.

  Raises:
    ValueError: The `activation_string` does not correspond to a known
      activation.
  """

  # We assume that anything that"s not a string is already an activation
  # function, so we just return it.
  if not isinstance(activation_string, six.string_types):
    return activation_string

  if not activation_string:
    return None

  act = activation_string.lower()
  if act == "linear":
    return None
  elif act == "gelu":
    return mtf.gelu
  elif act == "relu":
    return mtf.relu
  elif act == "tanh":
    return mtf.tanh
  else:
    raise ValueError("Unsupported activation: %s" % act)


def get_assignment_map_from_checkpoint(tvars, init_checkpoint):
  """Compute the union of the current variables and checkpoint variables."""
  assignment_map = {}
  initialized_variable_names = {}

  name_to_variable = collections.OrderedDict()
  for var in tvars:
    name = var.name
    m = re.match("^(.*):\\d+$", name)
    if m is not None:
      name = m.group(1)

    if "global_step" in name or "adam_" in name or "slot_" in name:
      continue
    name_to_variable[name] = var

  tf.logging.info("init_checkpoint:{} ".format(init_checkpoint))
  init_vars = tf.train.list_variables(init_checkpoint)

  assignment_map = collections.OrderedDict()
  for x in init_vars:
    (name, var) = (x[0], x[1])
    if name not in name_to_variable:
      continue
    assignment_map[name] = name
    initialized_variable_names[name] = 1
    initialized_variable_names[name + ":0"] = 1

  return (assignment_map, initialized_variable_names)


def _relative_position_bucket(relative_position,
                              bidirectional=True,
                              num_buckets=32,
                              max_distance=128):
  """Translate relative position to a bucket number for relative attention.

  The relative position is defined as memory_position - query_position, i.e.
  the distance in tokens from the attending position to the attended-to
  position.  If bidirectional=False, then positive relative positions are
  invalid.

  We use smaller buckets for small absolute relative_position and larger buckets
  for larger absolute relative_positions.  All relative positions >=max_distance
  map to the same bucket.  All relative positions <=-max_distance map to the
  same bucket.  This should allow for more graceful generalization to longer
  sequences than the model has been trained on.

  Args:
    relative_position: an int32 Tensor
    bidirectional: a boolean - whether the attention is bidirectional
    num_buckets: an integer
    max_distance: an integer
  Returns:
    a Tensor with the same shape as relative_position, containing int32
      values in the range [0, num_buckets)
  """
  ret = 0
  n = -relative_position
  if bidirectional:
    num_buckets //= 2
    ret += mtf.to_int32(mtf.less(n, 0)) * num_buckets
    n = mtf.abs(n)
  else:
    n = mtf.maximum(n, 0)
  # now n is in the range [0, inf)
  max_exact = num_buckets // 2
  is_small = mtf.less(n, max_exact)
  val_if_large = max_exact + mtf.to_int32(
      mtf.log(mtf.to_float(n) / max_exact)
      / math.log(max_distance / max_exact) * (num_buckets - max_exact))
  val_if_large = mtf.minimum(val_if_large, num_buckets - 1)
  ret += mtf.where(is_small, n, val_if_large)
  return ret


def layer_norm(x, dim, epsilon=1e-6,
               subtract_mean=True,
               use_scale=True,
               use_bias=True,
               name=None):
  """Layer normalization over dimension dim.

  TODO(noam): This is cleaner than the version in mtf.layers
  Move this version into mtf.layers to replace the one there.

  Args:
    x: a mtf.Tensor whose shape contains dim.
    dim: a mtf.Dimension
    epsilon: a floating point number
    subtract_mean: a boolean
    use_scale: a boolean
    use_bias: a boolean
    name: a string used for tf.variable_scope.

  Returns:
    a mtf.Tensor with same shape as x.
  """
  with tf.variable_scope(name, default_name="layer_norm"):
    if subtract_mean:
      x -= mtf.reduce_mean(x, reduced_dim=dim)
    variance = mtf.reduce_mean(mtf.square(x), reduced_dim=dim)
    x *= mtf.rsqrt(variance + epsilon)
    if use_scale:
      x *= mtf.get_variable(
          x.mesh,
          "scale",
          mtf.Shape([dim]),
          initializer=tf.ones_initializer(),
          activation_dtype=x.dtype)
    if use_bias:
      x += mtf.get_variable(
          x.mesh,
          "bias",
          mtf.Shape([dim]),
          initializer=tf.zeros_initializer(),
          activation_dtype=x.dtype)
    return x
