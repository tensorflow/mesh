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

# Lint as: python3
"""MeshTensorFlow implementation of BERT.

The code is ported from https://github.com/google-research/bert.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import google_type_annotations
from __future__ import print_function

import collections
import copy
import json
import re
import mesh_tensorflow as mtf
import six

import tensorflow.compat.v1 as tf


class BertConfig(object):
  """Configuration for `BertModel`."""

  def __init__(self,
               vocab_size,
               hidden_size=768,
               num_hidden_layers=12,
               num_attention_heads=12,
               intermediate_size=3072,
               hidden_act="gelu",
               hidden_dropout_prob=0.1,
               attention_probs_dropout_prob=0.1,
               max_position_embeddings=512,
               type_vocab_size=16,
               initializer_range=0.02):
    """Constructs BertConfig.

    Args:
      vocab_size: Vocabulary size of `inputs_ids` in `BertModel`.
      hidden_size: Size of the encoder layers and the pooler layer.
      num_hidden_layers: Number of hidden layers in the Transformer encoder.
      num_attention_heads: Number of attention heads for each attention layer in
        the Transformer encoder.
      intermediate_size: The size of the "intermediate" (i.e., feed-forward)
        layer in the Transformer encoder.
      hidden_act: The non-linear activation function (function or string) in the
        encoder and pooler.
      hidden_dropout_prob: The dropout probability for all fully connected
        layers in the embeddings, encoder, and pooler.
      attention_probs_dropout_prob: The dropout ratio for the attention
        probabilities.
      max_position_embeddings: The maximum sequence length that this model might
        ever be used with. Typically set this to something large just in case
        (e.g., 512 or 1024 or 2048).
      type_vocab_size: The vocabulary size of the `token_type_ids` passed into
        `BertModel`.
      initializer_range: The stdev of the truncated_normal_initializer for
        initializing all weight matrices.
    """
    self.vocab_size = vocab_size
    self.hidden_size = hidden_size
    self.num_hidden_layers = num_hidden_layers
    self.num_attention_heads = num_attention_heads
    self.hidden_act = hidden_act
    self.intermediate_size = intermediate_size
    self.hidden_dropout_prob = hidden_dropout_prob
    self.attention_probs_dropout_prob = attention_probs_dropout_prob
    self.max_position_embeddings = max_position_embeddings
    self.type_vocab_size = type_vocab_size
    self.initializer_range = initializer_range

  @classmethod
  def from_dict(cls, json_object):
    """Constructs a `BertConfig` from a Python dictionary of parameters."""
    config = BertConfig(vocab_size=None)
    for (key, value) in six.iteritems(json_object):
      config.__dict__[key] = value
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
               scope=None):
    self.config = copy.deepcopy(config)
    del config
    if not is_training:
      self.config.hidden_dropout_prob = 0.0
      self.config.attention_probs_dropout_prob = 0.0
    input_shape = input_ids.shape
    assert input_shape.ndims == 2

    self._seq_dim = input_shape.dims[1]
    self._memory_seq_dim = mtf.Dimension("memory_seq", self.seq_dim.size)
    mesh = input_ids.mesh

    if token_type_ids is None:
      token_type_ids = mtf.zeros(mesh, input_shape, dtype=tf.int32)

    with tf.variable_scope(scope, default_name="bert"):
      with tf.variable_scope("embeddings"):
        # Perform embedding lookup on the word ids.
        self.embedding_table = mtf.get_variable(
            mesh, "word_embeddings",
            mtf.Shape([self.vocab_dim, self.hidden_dim]),
            initializer=self.initializer)
        self.word_embedding_output = mtf.gather(
            self.embedding_table, input_ids, self.vocab_dim)

        # Add positional embeddings and token type embeddings, then layer
        # normalize and perform dropout.
        self.embedding_output = self.word_embedding_output

        token_type_table = mtf.get_variable(
            mesh, "token_type_embeddings",
            mtf.Shape([self.token_type_vocab_dim, self.hidden_dim]),
            initializer=self.initializer)
        if token_type_ids is not None:
          self.embedding_output += mtf.gather(
              token_type_table, token_type_ids, self.token_type_vocab_dim)
        full_position_table = mtf.get_variable(
            mesh, "position_embeddings",
            mtf.Shape([self.max_position_embeddings_dim, self.hidden_dim]),
            initializer=self.initializer)
        short_position_table = mtf.rename_dimension(
            mtf.slice(full_position_table, 0, self.seq_dim.size,
                      self.max_position_embeddings_dim.name),
            self.max_position_embeddings_dim.name, self.seq_dim.name)
        self.embedding_output += short_position_table
        self.embedding_output = mtf.layers.layer_norm(
            self.embedding_output, self.hidden_dim,
            name="emb_postprocessing_layer_norm")
        self.embedding_output = mtf.dropout(
            self.embedding_output,
            keep_prob=1.0 - self.config.hidden_dropout_prob)

      with tf.variable_scope("encoder"):
        # [batch_dim, memory_seq_dim]
        attention_mask = (
            None if input_mask is None else
            mtf.replace_dimensions(
                input_mask, self.seq_dim, self.memory_seq_dim))
        prev_output = self.embedding_output
        self.all_encoder_layers = []
        for layer_idx in range(self.config.num_hidden_layers):
          with tf.variable_scope("layer_%d" % layer_idx):
            layer_input = prev_output
            with tf.variable_scope("attention"):
              attention_output = self.self_attention(
                  layer_input, attention_mask)
              attention_output = mtf.dropout(
                  attention_output,
                  keep_prob=1.0 - self.config.hidden_dropout_prob)
              attention_output = mtf.layers.layer_norm(
                  attention_output + layer_input, self.hidden_dim)

            # The activation is only applied to the "intermediate" hidden layer.
            with tf.variable_scope("intermediate"):
              intermediate_output = mtf.layers.dense(
                  attention_output,
                  reduced_dims=[self.hidden_dim],
                  new_dims=[self.intermediate_dim],
                  activation=get_activation(self.config.hidden_act),
                  kernel_initializer=self.initializer,
                  name="dense")

            # Down-project back to `hidden_size` then add the residual.
            with tf.variable_scope("output"):
              layer_output = mtf.layers.dense(
                  intermediate_output,
                  reduced_dims=[self.intermediate_dim],
                  new_dims=[self.hidden_dim],
                  kernel_initializer=self.initializer,
                  name="dense")
              layer_output = mtf.dropout(
                  layer_output, keep_prob=1.0 - self.config.hidden_dropout_prob)
              layer_output = mtf.layers.layer_norm(
                  layer_output + attention_output, self.hidden_dim)
              prev_output = layer_output
              self.all_encoder_layers.append(layer_output)

      self.sequence_output = self.all_encoder_layers[-1]
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
            reduced_dims=[self.hidden_dim],
            new_dims=[self.hidden_dim],
            activation=mtf.tanh,
            kernel_initializer=self.initializer)

  def self_attention(self, x, attention_mask):
    """Performs multi-headed self-attention with output projection.

    Args:
      x: output of previous layer
      attention_mask: optional int32 Tensor broadcastable to shape
        x.shape - self.hidden_dim + self.memory_seq_dim
        The values should be 1 or 0. The attention scores will effectively be
        set to -infinity for any positions in the mask that are 0, and will be
        unchanged for positions that are 1.
        This is used to mask out padding regions of the memory.

    Returns:
      float Tensor with the same shape as x
    """
    queries = mtf.layers.dense(
        x,
        reduced_dims=[self.hidden_dim],
        new_dims=[self.num_heads_dim, self.size_per_head_dim],
        kernel_initializer=self.initializer,
        name="query")
    keys = mtf.layers.dense(
        mtf.replace_dimensions(x, self.seq_dim, self.memory_seq_dim),
        reduced_dims=[self.hidden_dim],
        new_dims=[self.num_heads_dim, self.size_per_head_dim],
        kernel_initializer=self.initializer,
        name="key")
    values = mtf.layers.dense(
        mtf.replace_dimensions(x, self.seq_dim, self.memory_seq_dim),
        reduced_dims=[self.hidden_dim],
        new_dims=[self.num_heads_dim, self.size_per_head_dim],
        kernel_initializer=self.initializer,
        name="value")

    # Take the dot product between "query" and "key" to get the raw
    # attention scores.
    attention_scores = mtf.einsum(
        [queries, keys], reduced_dims=[self.size_per_head_dim])
    attention_scores *= self.size_per_head_dim.size ** -0.5

    if attention_mask is not None:
      # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
      # masked positions, this operation will create a tensor which is 0.0 for
      # positions we want to attend and -10000.0 for masked positions.
      # Since we are adding it to the raw scores before the softmax, this is
      # effectively the same as removing these entirely.
      attention_scores += (1.0 - mtf.to_float(attention_mask)) * -10000.0

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
        new_dims=[self.hidden_dim],
        kernel_initializer=self.initializer,
        name="output")
    output = mtf.transpose(output, x.shape)
    return output

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
            reduced_dims=[self.hidden_dim],
            new_dims=[self.hidden_dim],
            activation=get_activation(self.config.hidden_act),
            kernel_initializer=self.initializer)
        input_tensor = mtf.layers.layer_norm(input_tensor, self.hidden_dim)
      # The output weights are the same as the input embeddings, but there is
      # an output-only bias for each token.
      output_bias = mtf.get_variable(
          input_tensor.mesh,
          name="output_bias",
          shape=[self.vocab_dim],
          initializer=tf.zeros_initializer())
      logits = mtf.einsum([input_tensor, output_weights],
                          reduced_dims=[self.hidden_dim]) + output_bias
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
        reduced_dims=[self.hidden_dim],
        new_dims=[class_dim],
        kernel_initializer=self.initializer,
        name="cls/seq_relationship")
    per_example_loss = mtf.layers.softmax_cross_entropy_with_logits(
        logits, labels, class_dim, z_loss=1e-4)
    loss = mtf.reduce_mean(per_example_loss)
    return (loss, per_example_loss, logits)

  def get_pooled_output(self):
    return self.pooled_output

  def get_sequence_output(self):
    """Gets final hidden layer of encoder.

    Returns:
      float Tensor of shape [batch_dim, seq_dim, hidden_dim] corresponding
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
      float Tensor of shape [batch_dim, seq_dim, hidden_dim] corresponding
      to the output of the word(piece) embedding layer.
    """
    return self.word_embedding_output

  def get_embedding_output(self):
    """Gets output of the embedding lookup (i.e., input to the transformer).

    Returns:
      float Tensor of shape [batch_dim, seq_dim, hidden_dim] corresponding
      to the output of the embedding layer, after summing the word
      embeddings with the positional embeddings and the token type embeddings,
      then performing layer normalization. This is the input to the transformer.
    """
    return self.embedding_output

  def get_embedding_table(self):
    return self.embedding_table

  @property
  def vocab_dim(self):
    # pad vocab to a multiple of 128 so as to be splittable.
    # TODO(noam): This creates issues in checkpoint compatibility
    n = self.config.vocab_size
    return mtf.Dimension("vocab", n + (-n % 128))

  @property
  def hidden_dim(self):
    return mtf.Dimension("hidden", self.config.hidden_size)

  @property
  def token_type_vocab_dim(self):
    return mtf.Dimension("token_type_vocab", self.config.type_vocab_size)

  @property
  def intermediate_dim(self):
    return mtf.Dimension("intermediate", self.config.intermediate_size)

  @property
  def num_heads_dim(self):
    return mtf.Dimension("num_heads", self.config.num_attention_heads)

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
  def initializer(self):
    return tf.truncated_normal_initializer(stddev=self.config.initializer_range)

  @property
  def size_per_head_dim(self):
    if self.hidden_dim.size % self.num_heads_dim.size != 0:
      raise ValueError(
          "The hidden size (%d) is not a multiple of the number of attention "
          "heads (%d)" % (self.hidden_dim.size, self.num_heads_dim.size))
    attention_head_size = int(self.hidden_dim.size / self.num_heads_dim.size)
    return mtf.Dimension("attention_head", attention_head_size)


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

    if "global_step" in name or "adam_" in name:
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
