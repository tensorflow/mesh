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
"""Different ways to go from token ids to hidden states and states to logits."""

import gin
import mesh_tensorflow as mtf
from mesh_tensorflow.transformer import transformer

import tensorflow.compat.v1 as tf


@gin.configurable
class FactorizedVocabEmbedding(object):
  """Factorizes the embedding matrix with projection to a small inner dimension.

  Like ALBERT (https://arxiv.org/abs/1706.03762).

  Interface matches mesh_tensorflow.transformer VocabEmbedding object.
  """

  def __init__(self,
               mesh,
               vocab_dim,
               output_dim,
               variable_dtype,
               name,
               ensemble_dim,
               inner_dimension_size=gin.REQUIRED):
    """Configurable embedding for the vocabulary.

    Most of the arguments get passed to `mtf.layers.embedding_weights` with an
    option to factorize the embedding matrix.

    Args:
      mesh: a mtf.Mesh
      vocab_dim: a mtf.Dimension
      output_dim: a mtf.Dimension
      variable_dtype: a mtf.VariableDType
      name: a string
      ensemble_dim: a mtf.Dimension
      inner_dimension_size: a positive integer, the size of the inner dimension
        of the embedding matrix
    """
    self._vocab_dim = vocab_dim
    self._output_dim = output_dim
    self._inner_dim = mtf.Dimension("inner_vocab", inner_dimension_size)
    self._factor1 = mtf.layers.embedding_weights(
        mesh=mesh,
        vocab_dim=vocab_dim,
        output_dim=self._inner_dim,
        variable_dtype=variable_dtype,
        name="{}1".format(name),
        ensemble_dim=ensemble_dim,
        initializer=tf.random_normal_initializer(
            stddev=inner_dimension_size**-0.25))
    self._factor2 = mtf.layers.embedding_weights(
        mesh=mesh,
        vocab_dim=self._inner_dim,
        output_dim=output_dim,
        variable_dtype=variable_dtype,
        name="{}2".format(name),
        ensemble_dim=ensemble_dim,
        initializer=tf.random_normal_initializer(
            stddev=inner_dimension_size**-0.25))

  def ids_to_embedding(self, ids, context):
    del context
    tmp = mtf.gather(self._factor1, ids, self._vocab_dim)
    return mtf.einsum([tmp, self._factor2], reduced_dims=[self._inner_dim])

  def hidden_to_logits(self, hidden, context):
    del context
    hidden *= self._output_dim.size**-0.5
    tmp = mtf.einsum([hidden, self._factor2], reduced_dims=[self._output_dim])
    return mtf.einsum([tmp, self._factor1], reduced_dims=[self._inner_dim])


class _Cluster(object):
  """Helper class for adaptive embeddings specifying a cluster of tokens.

  Essentially a wrapper around a vocab embedding for the cluster with additional
  metadata so that we can apply the embedding to the actual ids and hidden
  states.
  """

  def __init__(self, embedding, start_token_id, end_token_id):
    """Cluster constructor.

    Args:
      embedding: a FactorizedVocabEmbedding or transformer.VocabEmbedding, the
        vocab embedding to use for the cluster
      start_token_id: an integer, the inclusive id of the first token in the
        cluster
      end_token_id: an integer, the exclusive id of the last token in the
        cluster
    """
    self._embedding = embedding
    self._start_token_id = start_token_id
    self._end_token_id = end_token_id

  def ids_to_embedding(self, ids, context):
    """Ids to embeddings with ids not in cluster mapped to the zero vector."""
    ids -= self._start_token_id
    # The mtf.gather in the embedding's ids_to_embedding implementation will
    # cause the one hot representations of tokens greater than cluster vocab
    # dimension size to be the zero vector. Thus the embeddings for those tokens
    # will be the zero vector.
    ids = mtf.where(mtf.greater_equal(ids, 0), ids, self._end_token_id)
    return self._embedding.ids_to_embedding(ids, context)

  def hidden_to_logits(self, hidden, context):
    """Returns the logits for tokens within the cluster."""
    return self._embedding.hidden_to_logits(hidden, context)


@gin.configurable
class AdaptiveVocabEmbedding(object):
  """A vocab embedding assigning variable capacity to clusters of tokens.

  Similar to the adaptive input representations in this paper
  (https://arxiv.org/abs/1809.10853). However, they use an adaptive softmax to
  compute logits while this embedding uses a regular softmax.

  The idea is to create clusters of tokens and assign different capacity to
  different clusters by factoring their embedding matrices to different inner
  dimensions.

  The clustering can be done by word frequency with more frequent tokens getting
  higher capacity. In this implementation, token ids of clusters must be
  contiguous in the vocabulary.

  Interface matches mesh_tensorflow.transformer VocabEmbedding object.
  """

  def __init__(self,
               mesh,
               vocab_dim,
               output_dim,
               variable_dtype,
               name,
               ensemble_dim,
               clusters=gin.REQUIRED):
    """Configurable embedding for the vocabulary.

    Most of the arguments get passed to `mtf.layers.embedding_weights`.

    The clustering parameters are specified by the `clusters` argument. It is a
    list of dicts with keys "token_count" and "embedding_size". Token count
    specifies the number of tokens in the cluster, and embedding size specifies
    the hidden dimension size of its embedding.

    For example, let's say we have a vocab size of 500k and pass as clusters:
      [
        {"token_count": 50000,  "embedding_size": 1024},
        {"token_count": 100000, "embedding_size": 256},
        {"token_count": 350000, "embedding_size": 64},
      ]
    Then tokens with ids 0 (inclusive) to 50k (exclusive) will be in the first
    cluster with embedding size of 1024, tokens with ids 50k to 150k will be in
    the second cluster with embedding size of 256, and tokens with ids 150k to
    500k will be in the third cluster with embedding size of 64.

    Args:
      mesh: a mtf.Mesh
      vocab_dim: a mtf.Dimension
      output_dim: a mtf.Dimension
      variable_dtype: a mtf.VariableDType
      name: a string
      ensemble_dim: a mtf.Dimension
      clusters: a list(dict), specification of the clusters

    Raises:
      ValueError: The sum of the token counts across the clusters does not equal
        the vocabulary size.
    """
    self._vocab_dim = vocab_dim
    self._output_dim = output_dim

    token_counts = [cluster["token_count"] for cluster in clusters]
    if sum(token_counts) != vocab_dim.size:
      raise ValueError(
          "The cluster token counts {} do not sum to the vocab size {}.".format(
              token_counts, vocab_dim.size))

    self._clusters = []
    start_token_id = 0
    for i, cluster in enumerate(clusters):
      token_count = cluster["token_count"]
      embedding_size = cluster["embedding_size"]
      cluster_vocab_dim = mtf.Dimension(vocab_dim.name, token_count)

      if embedding_size == self._output_dim.size:
        # In this case we don't need to up project from the embedding space to
        # the model state space.
        cluster_embedding = transformer.VocabEmbedding(
            mesh=mesh,
            vocab_dim=cluster_vocab_dim,
            output_dim=output_dim,
            variable_dtype=variable_dtype,
            name="{}_{}".format(name, i),
            ensemble_dim=ensemble_dim)
      else:
        cluster_embedding = FactorizedVocabEmbedding(
            mesh=mesh,
            vocab_dim=cluster_vocab_dim,
            output_dim=output_dim,
            variable_dtype=variable_dtype,
            name="{}_{}".format(name, i),
            ensemble_dim=ensemble_dim,
            inner_dimension_size=embedding_size)
      self._clusters.append(
          _Cluster(
              embedding=cluster_embedding,
              start_token_id=start_token_id,
              end_token_id=start_token_id + token_count))
      start_token_id += token_count

  def ids_to_embedding(self, ids, context):
    # Ids not in each cluster will be mapped to the zero vector. Since clusters
    # are disjoint, this sum is correct.
    return sum(
        cluster.ids_to_embedding(ids, context) for cluster in self._clusters)

  def hidden_to_logits(self, hidden, context):
    # Each cluster returns the logits for only the tokens with itself, so their
    # concatenation is the full logits.
    return mtf.concat(
        [
            cluster.hidden_to_logits(hidden, context=context)
            for cluster in self._clusters
        ],
        concat_dim_name=self._vocab_dim.name,
    )


@gin.configurable
class MixtureOfSoftmaxes(object):
  """Embedding with the token distributions as a weighted mixture of softmaxes.

  Expressing the token distributions in this way improves expressiveness and
  enables the matrix of token probabilities given all contexts to be high rank.

  The vocab embedding is the same as the default, which is just a simple
  embedding.

  See https://arxiv.org/pdf/1711.03953.pdf for more details.
  """

  def __init__(self,
               mesh: mtf.Mesh,
               vocab_dim: mtf.Dimension,
               output_dim: mtf.Dimension,
               variable_dtype: mtf.VariableDType,
               name: str,
               ensemble_dim: mtf.Dimension,
               num_softmaxes: int = gin.REQUIRED):
    """Configurable embedding for the vocabulary.

    Most of the arguments get passed to `mtf.layers.embedding_weights`.

    Args:
      mesh: the mesh used to layout the tensors.
      vocab_dim: the dimension corresponding to vocabulary.
      output_dim: the dimension corresponding to the model
        hidden states.
      variable_dtype: the datatype information for the
        variables used in the embedding tensors.
      name: a name to base variable names off of.
      ensemble_dim: the dimension used for ensembling.
        Absolutely no guarantees that this code will work with ensembling.
      num_softmaxes: a positive int, the number of components to use in the
        mixture.
    """
    self._vocab_dim = vocab_dim
    self._output_dim = output_dim
    self._copy_output_dim = mtf.Dimension("_{}_copy".format(output_dim.name),
                                          output_dim.size)
    self._components_dim = mtf.Dimension("softmax_components", num_softmaxes)

    self._embedding_weights = mtf.layers.embedding_weights(
        mesh=mesh,
        vocab_dim=vocab_dim,
        output_dim=output_dim,
        variable_dtype=variable_dtype,
        name="{}_embedding_weights".format(name),
        ensemble_dim=ensemble_dim)
    self._mixture_weights = mtf.layers.embedding_weights(
        mesh=mesh,
        vocab_dim=self._components_dim,
        output_dim=output_dim,
        variable_dtype=variable_dtype,
        name="{}_mixture_weights".format(name),
        ensemble_dim=ensemble_dim)
    self._context_weights = mtf.layers.embedding_weights(
        mesh=mesh,
        vocab_dim=self._copy_output_dim,
        output_dim=output_dim,
        variable_dtype=variable_dtype,
        name="{}_context_weights".format(name),
        ensemble_dim=([ensemble_dim] if ensemble_dim else []) +
        [self._components_dim])

  def ids_to_embedding(self, ids: mtf.Tensor, context) -> mtf.Tensor:
    del context
    return mtf.gather(self._embedding_weights, ids, self._vocab_dim)

  def hidden_to_logits(self, hidden: mtf.Tensor,
                       context: transformer.Context) -> mtf.Tensor:
    """Function called by mtf transformer to get the logits.

    Note that we are taking the log of a mixture of softmaxes. The logits will
    then go through a softmax. This could potentially run into numerical
    stability issues. If that happens, try setting the activation_dtype to
    float32.

    Args:
      hidden: hidden model states of the final decoder layer.
      context: the context used for the call to the
        transformer.

    Returns:
      The logits.
    """
    del context
    hidden *= self._output_dim.size**-0.5

    component_prior_logits = mtf.einsum([hidden, self._mixture_weights],
                                        reduced_dims=[self._output_dim])

    component_contexts = mtf.einsum([
        mtf.rename_dimension(hidden, self._output_dim.name,
                             self._copy_output_dim.name),
        self._context_weights,
    ],
                                    reduced_dims=[self._copy_output_dim])
    component_contexts = mtf.tanh(component_contexts)
    component_logits = mtf.einsum([component_contexts, self._embedding_weights],
                                  reduced_dims=[self._output_dim])

    component_prior_logits = mtf.log_softmax(
        component_prior_logits, reduced_dim=self._components_dim)
    component_logits = mtf.log_softmax(
        component_logits, reduced_dim=self._vocab_dim)

    logits = component_prior_logits + component_logits
    logits = mtf.reduce_logsumexp(logits, reduced_dim=self._components_dim)
    return logits


@gin.configurable
class Mixtape(object):
  """Embedding that uses Mixtape in computing logits.

  Expressing the token distributions in this way improves expressiveness and
  enables the matrix of token probabilities given all contexts to be high rank.

  Mixtape has the advantage of added efficiency over other methods such as
  mixture of softmax.

  The vocab embedding is the same as the default, which just a simple embedding.

  See
  https://papers.nips.cc/paper/9723-mixtape-breaking-the-softmax-bottleneck-efficiently.pdf
  for more details.
  """

  def __init__(self,
               mesh: mtf.Mesh,
               vocab_dim: mtf.Dimension,
               output_dim: mtf.Dimension,
               variable_dtype: mtf.VariableDType,
               name: str,
               ensemble_dim: mtf.Dimension,
               extra_ids: int = 0,
               dropout_rate: float = 0.0,
               gate_embedding_size: int = gin.REQUIRED,
               frequent_token_fraction: float = 0.1,
               noise_std_dev: float = 0.0):
    """Configurable embedding for the vocabulary.

    Most of the arguments get passed to `mtf.layers.embedding_weights`.

    Mixtape shares gates for low frequency tokens to improve efficiency. Since
    our vocabs are sorted in decreasing order of frequency with sentinels
    appended to the end, we need to do a little trick to ensure that the
    sentinels are treated as high frequency. If you want to treat the sentinels
    as low frequency tokens, then pass in zero for `extra_ids`.

    Args:
      mesh: the mesh used to layout the tensors.
      vocab_dim: the dimension corresponding to vocabulary.
      output_dim: the dimension corresponding to the model hidden states.
      variable_dtype: the datatype information for the  variables used in the
        embedding tensors.
      name: a name to base variable names off of.
      ensemble_dim: the dimension used for ensembling. Absolutely no guarantees
        that this code will work with ensembling.
      extra_ids: a non-negative integer, the number of sentinels at the end of
        the vocab.
      dropout_rate: a float between 0 and 1, the rate to use for dropout.
      gate_embedding_size: a positive integer, the size to use for embedding for
        the gates. It is usually chosen to be much smaller than d_model.
      frequent_token_fraction: a float between 0 and 1, what fraction of tokens
        to consider as high frequency and not share gates for.
      noise_std_dev: a non-negative float, the standard deviation of the
        Gaussian noise to add to the pre-activation priors.
    """
    self._extra_ids = extra_ids
    self._dropout_rate = dropout_rate
    self._noise_std_dev = noise_std_dev
    self._mesh = mesh
    self._vocab_dim = vocab_dim
    self._frequent_vocab_dim = mtf.Dimension(
        vocab_dim.name, int(frequent_token_fraction * vocab_dim.size))
    self._rare_vocab_dim = mtf.Dimension(
        vocab_dim.name, vocab_dim.size - self._frequent_vocab_dim.size)
    self._output_dim = output_dim
    self._copy_output_dim = mtf.Dimension("_{}_copy".format(output_dim.name),
                                          output_dim.size)
    self._pre_gates_dim = mtf.Dimension("gates", 3)
    self._gates_dim = mtf.Dimension("gates", 4)
    self._gate_embedding_dim = mtf.Dimension("gate_embedding",
                                             gate_embedding_size)

    self._embedding_weights = mtf.layers.embedding_weights(
        mesh=mesh,
        vocab_dim=vocab_dim,
        output_dim=output_dim,
        variable_dtype=variable_dtype,
        name="{}_embedding_weights".format(name),
        ensemble_dim=ensemble_dim)
    ensemble_dims = [ensemble_dim] if ensemble_dim else []
    self._context_weights = mtf.layers.embedding_weights(
        mesh=mesh,
        vocab_dim=self._copy_output_dim,
        output_dim=output_dim,
        variable_dtype=variable_dtype,
        name="{}_context_weights".format(name),
        ensemble_dim=ensemble_dims + [self._gates_dim])
    self._context_weights_bias = mtf.get_variable(
        mesh,
        name="{}_context_weights_bias".format(name),
        shape=mtf.Shape(ensemble_dims + [self._gates_dim, output_dim]),
        dtype=variable_dtype,
        initializer=tf.zeros_initializer())

    self._prior_weights = mtf.layers.embedding_weights(
        mesh=mesh,
        vocab_dim=self._gate_embedding_dim,
        output_dim=output_dim,
        variable_dtype=variable_dtype,
        name="{}_prior_weights".format(name),
        ensemble_dim=ensemble_dims + [self._pre_gates_dim])
    self._prior_weights_bias = mtf.get_variable(
        mesh,
        name="{}_prior_weights_bias".format(name),
        shape=mtf.Shape(ensemble_dims +
                        [self._pre_gates_dim, self._gate_embedding_dim]),
        dtype=variable_dtype,
        initializer=tf.zeros_initializer())
    self._prior_vocab_vector = mtf.get_variable(
        mesh,
        name="{}_prior_vocab_vector".format(name),
        shape=mtf.Shape(ensemble_dims +
                        [self._frequent_vocab_dim, self._gate_embedding_dim]),
        dtype=variable_dtype,
        initializer=tf.random_normal_initializer())
    self._prior_gates_vector = mtf.get_variable(
        mesh,
        name="{}_prior_gates_vector".format(name),
        shape=mtf.Shape(ensemble_dims + [self._pre_gates_dim, output_dim]),
        dtype=variable_dtype,
        initializer=tf.random_normal_initializer())
    self._prior_bias = mtf.get_variable(
        mesh,
        name="{}_prior_bias".format(name),
        shape=mtf.Shape(ensemble_dims +
                        [self._frequent_vocab_dim, self._pre_gates_dim]),
        dtype=variable_dtype,
        initializer=tf.random_normal_initializer())

  def ids_to_embedding(self, ids: mtf.Tensor, context) -> mtf.Tensor:
    del context
    return mtf.gather(self._embedding_weights, ids, self._vocab_dim)

  def _sigmoid_tree(self, tensor):
    """Create probability distribution along gates dim using a sigmoid tree."""
    gamma = mtf.split(
        mtf.sigmoid(tensor), self._pre_gates_dim, self._pre_gates_dim.size)
    return mtf.concat([
        gamma[0] * gamma[1],
        gamma[0] * (1 - gamma[1]),
        (1 - gamma[0]) * gamma[2],
        (1 - gamma[0]) * (1 - gamma[2]),
    ], self._gates_dim.name)

  def _dropout(self, tensor, context):
    if context.train and self._dropout_rate != 0.0:
      return mtf.dropout(
          tensor,
          1.0 - self._dropout_rate,
          noise_shape=tensor.shape - context.length_dim)
    return tensor

  def _rearrange_sentinels(self, logits):
    """Reorder along the vocab dim so the last few tokens don't share gates."""
    if not self._extra_ids:
      return logits
    sentinels, nonsentinels = mtf.split(
        logits, self._vocab_dim,
        [self._extra_ids, self._vocab_dim.size - self._extra_ids])
    return mtf.concat([nonsentinels, sentinels], self._vocab_dim.name)

  def hidden_to_logits(self, hidden: mtf.Tensor,
                       context: transformer.Context) -> mtf.Tensor:
    """Function called by mtf transformer to get the logits.

    Args:
      hidden: an mtf.Tensor, hidden model states of the final decoder layer.
      context: a transformer.Context, the context used for the call to the
        transformer.

    Returns:
      An mtf.Tensor, the logits.
    """
    hidden *= self._output_dim.size**-0.5

    component_contexts = mtf.einsum([
        mtf.rename_dimension(hidden, self._output_dim.name,
                             self._copy_output_dim.name),
        self._context_weights,
    ],
                                    reduced_dims=[self._copy_output_dim])
    component_contexts = mtf.tanh(component_contexts +
                                  self._context_weights_bias)
    component_logits = mtf.einsum([component_contexts, self._embedding_weights],
                                  reduced_dims=[self._output_dim])
    component_logits = self._dropout(component_logits, context)

    prior_tanh = mtf.tanh(
        mtf.einsum([self._prior_weights, hidden],
                   reduced_dims=[self._output_dim]) + self._prior_weights_bias)
    prior_tanh = self._dropout(prior_tanh, context)
    prior_shared_logits = mtf.einsum([self._prior_gates_vector, hidden],
                                     reduced_dims=[self._output_dim])
    prior_frequent_vocab_logits = (
        mtf.einsum([self._prior_vocab_vector, prior_tanh]) +
        prior_shared_logits + self._prior_bias)
    prior_logits = mtf.concat([
        prior_frequent_vocab_logits,
        mtf.ones(
            self._mesh,
            mtf.Shape([self._rare_vocab_dim]),
            dtype=prior_shared_logits.dtype) * prior_shared_logits
    ], self._vocab_dim.name)
    if context.train and self._noise_std_dev != 0.0:
      prior_logits += mtf.random_normal(
          self._mesh, prior_logits.shape, stddev=self._noise_std_dev)
    prior_proportions = self._sigmoid_tree(prior_logits)

    logits = mtf.einsum([component_logits, prior_proportions],
                        reduced_dims=[self._gates_dim])
    return self._rearrange_sentinels(logits)
