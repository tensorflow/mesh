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
"""Implementation of adaptive softmax.

See the papers https://arxiv.org/abs/1609.04309 and
https://arxiv.org/abs/1809.10853 for more details.
"""

import math
from typing import Dict, Sequence, Union

import gin
import mesh_tensorflow as mtf
from mesh_tensorflow.transformer import transformer
from mesh_tensorflow.transformer import vocab_embeddings

import tensorflow.compat.v1 as tf


class _Cluster(object):
  """Helper class for adaptive embeddings specifying a cluster of tokens.

  Essentially a wrapper around a vocab embedding for the cluster with additional
  metadata so that we can apply the embedding to the actual ids and hidden
  states.
  """

  def __init__(self, embedding, start_token_id, end_token_id,
               length_projection_factor, vocab_dim):
    """Cluster constructor.

    Args:
      embedding: a FactorizedVocabEmbedding or transformer.VocabEmbedding, the
        vocab embedding to use for the cluster.
      start_token_id: an integer, the inclusive id of the first token in the
        cluster.
      end_token_id: an integer, the exclusive id of the last token in the
        cluster.
      length_projection_factor: a float between 0 and 1, the sequence length
        dimension will be projected down to this number times the sequence
        length dimension to contain the elements in this cluster. If the input
        contains too many tokens in the cluster, tokens later in the input will
        be ignored.
      vocab_dim: an mtf.Dimension, the dimension the embedding uses as its
        vocab.
    """
    self._embedding = embedding
    self._start_token_id = start_token_id
    self._end_token_id = end_token_id
    self._length_projection_factor = length_projection_factor
    self._vocab_dim = vocab_dim

  @property
  def end_token_id(self):
    return self._end_token_id

  @property
  def length_projection_factor(self):
    return self._length_projection_factor

  def ids_to_embedding(self, ids, context):
    """Ids to embeddings with ids not in cluster mapped to the zero vector."""
    ids -= self._start_token_id
    # The mtf.gather in the embedding's ids_to_embedding implementation will
    # cause the one hot representations of tokens greater than cluster vocab
    # dimension size to be the zero vector. Thus the embeddings for those tokens
    # will be the zero vector.
    ids = mtf.where(mtf.greater_equal(ids, 0), ids, self._vocab_dim.size)
    # Handle the case of the head cluster where we will have entries at the end
    # corresponding to the tail clusters.
    ids = mtf.where(
        mtf.less(ids, self._end_token_id - self._start_token_id),
        ids,
        self._vocab_dim.size,
    )
    return self._embedding.ids_to_embedding(ids, context)

  def get_cluster_mask(self, targets):
    """Computes mask over the targets masking out tokens not in the cluster."""
    return mtf.logical_and(
        mtf.greater_equal(targets, self._start_token_id),
        mtf.less(targets, self._end_token_id))

  def get_cluster_length_dim(self, length_dim):
    """Returns dimension used instead of sequence length for the cluster."""
    cluster_length = math.ceil(self._length_projection_factor * length_dim.size)
    return mtf.Dimension(length_dim.name, int(cluster_length))

  def get_project_to_cluster_length(self, cluster_mask, dtype):
    """Returns projection from length dim to the shorter cluster length dim."""
    seq_length_dim = cluster_mask.shape.get_dim_by_name("length")
    cluster_length_dim = self.get_cluster_length_dim(seq_length_dim)
    return mtf.cast(cluster_mask, dtype) * mtf.one_hot(
        mtf.cumsum(mtf.cast(cluster_mask, tf.int32), seq_length_dim) - 1,
        output_dim=cluster_length_dim,
        dtype=dtype)

  def compute_loss(self, decoder, hidden, targets, context):
    """Computes the loss during training."""
    logits = self._embedding.hidden_to_logits(hidden, context=context)
    soft_targets = mtf.one_hot(
        targets - self._start_token_id,
        self._vocab_dim,
        dtype=context.activation_dtype)
    loss = mtf.layers.softmax_cross_entropy_with_logits(
        logits, soft_targets, self._vocab_dim, z_loss=decoder.z_loss)

    padding_mask = mtf.layers.weights_nonzero(
        targets, dtype=context.activation_dtype)

    return (mtf.reduce_sum(loss * padding_mask) /
            decoder.loss_denominator(targets, context.num_microbatches))

  def compute_log_softmax(self, hidden, context):
    """Returns the log softmax of logits computed from the hidden state."""
    logits = self._embedding.hidden_to_logits(hidden, context=context)
    return mtf.log_softmax(logits, reduced_dim=self._vocab_dim)

  def get_log_softmax_prefix(self, log_softmax, end_index):
    """Returns first end_index entries in log_softmax along the vocab dim."""
    prefix_dim = mtf.Dimension(self._vocab_dim.name, end_index)

    indices = mtf.mtf_range(
        log_softmax.mesh, dim=self._vocab_dim, dtype=tf.int32)
    prefix_indices = mtf.where(mtf.less(indices, end_index), indices, -1)
    projection = mtf.one_hot(
        prefix_indices, prefix_dim, dtype=log_softmax.dtype)

    return mtf.einsum([log_softmax, projection], reduced_dims=[self._vocab_dim])

  def get_log_softmax_value(self, log_softmax, index):
    """Returns the entry at index of the log_softmax along the vocab dim."""
    return mtf.gather(log_softmax, index, dim=self._vocab_dim)


@gin.configurable
class AdaptiveSoftmaxVocabEmbedding(object):
  """Vocab embedding implementing the adaptive softmax.

  The adaptive softmax was first introduced in this paper
  (https://arxiv.org/abs/1609.04309). Note that this implementation is actually
  most similar to the adaptive vocab embeddings in
  https://arxiv.org/abs/1809.10853 as it supports having different embedding
  sizes for different clusters.

  The adaptive softmax works by factorizing the traditional softmax over
  multiple clusters:
    p(v|h) = p(v|c,h) p(c|h),
  where both probability distributions take the form of a softmax.

  Further speed up is achieved by putting the class containing the most
  frequently occurring tokens in the "head" cluster. Essentially, those tokens
  are included as "classes" in the p(c|h) softmax. Thus computing their
  probabilities requires only single softmax evaluation.

  This implementation differs from vocab_embeddings.AdaptiveVocabEmbedding. That
  implementation only supports variable embeddings sizes across clusters. This
  implementation also supports the adaptive softmax.

  A few conditions must be met in order to use this vocab:
    - Unitransformer.shared_embedding_and_softmax_weights = True.
    - If training, then
      Unitranformer.loss_fn = adaptive_softmax.adaptive_softmax_loss_fn.
    - Label smoothing is not supported and will be ignored silently.
    - loss_on_targets_only is not supported and will be ignored silently.
  """

  def __init__(self,
               mesh: mtf.Mesh,
               vocab_dim: mtf.Dimension,
               output_dim: mtf.Dimension,
               variable_dtype: mtf.VariableDType,
               name: str,
               ensemble_dim: mtf.Dimension,
               clusters: Sequence[Dict[str, Union[int, float]]] = gin.REQUIRED):
    """Configurable embedding for the vocabulary.

    Most of the arguments get passed to `mtf.layers.embedding_weights`.

    The clustering parameters are specified by the `clusters` argument. It is a
    list of dicts with keys:
      - token_count: The number of tokens in the cluster.
      - embedding_size: (optional) The hidden dimension size of the cluster's
        embedding. Defaults to the model dimension size.
      - length_projection_factor: (optional) Since MTF can't handle variable
        length dimensions, we project from the sequence length dimension to a
        dimension of size length_projection_factor * sequence_length during
        training. This can save compute time and resources if the cluster has
        many tokens that appear infrequently. If all of the tokens belonging to
        the cluster cannot fit within this reduced dimension, some will be
        discarded and ignored for the purposes of computing loss. Defaults 1.
        Ignored for the head (first) cluster and not during training.

    The first cluster will become the head cluster.

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
      mesh: a mtf.Mesh, the mesh used to layout the tensors.
      vocab_dim: a mtf.Dimension, the dimension corresponding to vocabulary.
      output_dim: a mtf.Dimension, the dimension corresponding to the model
        hidden states.
      variable_dtype: a mtf.VariableDType, the datatype information for the
        variables used in the embedding tensors.
      name: a string, a name to base variable names off of.
      ensemble_dim: a mtf.Dimension, the dimension used for ensembling.
        Absolutely no guarantees that this code will work with ensembling.
      clusters: a list(dict), specification of the clusters. See above for more
        information.

    Raises:
      ValueError: The sum of the token counts across the clusters does not equal
        the vocabulary size or a length_projection_factor is not in the range
        (0, 1].
    """
    self._mesh = mesh
    self._variable_dtype = variable_dtype
    self._name = name
    self._ensemble_dim = ensemble_dim
    self._vocab_dim = vocab_dim
    self._output_dim = output_dim
    self._num_clusters = len(clusters)

    token_counts = [cluster["token_count"] for cluster in clusters]
    if sum(token_counts) != vocab_dim.size:
      raise ValueError(
          "The cluster token counts {} do not sum to the vocab size {}.".format(
              token_counts, vocab_dim.size))

    self._tail_clusters = []
    start_token_id = 0
    for i, cluster_spec in enumerate(clusters):
      cluster = self._create_cluster(cluster_spec, i, start_token_id)
      if i == 0:
        self._head_cluster = cluster
      else:
        self._tail_clusters.append(cluster)
      start_token_id += cluster_spec["token_count"]

  def _create_cluster(self, cluster_spec, index, start_token_id):
    """Creates a cluster given its spec."""
    token_count = cluster_spec["token_count"]
    embedding_size = cluster_spec.get("embedding_size", self._output_dim.size)
    length_projection_factor = cluster_spec.get("length_projection_factor", 1)
    if length_projection_factor <= 0 or length_projection_factor > 1:
      raise ValueError(
          "Invalid length_projection_factor of {}. Must be in range (0, 1]"
          .format(length_projection_factor))

    if index == 0:
      # Include the entries for the tail clusters in the head cluster "vocab".
      cluster_vocab_dim = mtf.Dimension(self._vocab_dim.name,
                                        token_count + self._num_clusters - 1)
    else:
      cluster_vocab_dim = mtf.Dimension(self._vocab_dim.name, token_count)

    if embedding_size == self._output_dim.size:
      # In this case we don't need to up project from the embedding space to
      # the model state space.
      cluster_embedding = transformer.VocabEmbedding(
          mesh=self._mesh,
          vocab_dim=cluster_vocab_dim,
          output_dim=self._output_dim,
          variable_dtype=self._variable_dtype,
          name="{}_{}".format(self._name, index),
          ensemble_dim=self._ensemble_dim)
    else:
      cluster_embedding = vocab_embeddings.FactorizedVocabEmbedding(
          mesh=self._mesh,
          vocab_dim=cluster_vocab_dim,
          output_dim=self._output_dim,
          variable_dtype=self._variable_dtype,
          name="{}_{}".format(self._name, index),
          ensemble_dim=self._ensemble_dim,
          inner_dimension_size=embedding_size)
    return _Cluster(
        embedding=cluster_embedding,
        start_token_id=start_token_id,
        end_token_id=start_token_id + token_count,
        length_projection_factor=length_projection_factor,
        vocab_dim=cluster_vocab_dim)

  def ids_to_embedding(self, ids: mtf.Tensor, context) -> mtf.Tensor:
    all_clusters = self._tail_clusters + [self._head_cluster]
    # Ids not in each cluster will be mapped to the zero vector. Since clusters
    # are disjoint, this sum is correct.
    return sum(
        cluster.ids_to_embedding(ids, context) for cluster in all_clusters)

  def hidden_to_logits(self, hidden: mtf.Tensor,
                       context: transformer.Context) -> mtf.Tensor:
    """Function called by mtf transformer to get the logits.

    The benefit from the adaptive softmax comes from not having to compute the
    logits over all of the vocab during training. Thus, we use the somewhat
    hacky solution of returning the hidden states during training and then using
    them to compute the loss in a custom loss function.

    When not training, this method will be true to its name as return the
    logits corresponding to the hidden state.

    Args:
      hidden: an mtf.Tensor, hidden model states of the final decoder layer.
      context: a transformer.Context, the context used for the call to the
        transformer.

    Returns:
      an mtf.Tensor
    """
    if context.mode == tf.estimator.ModeKeys.TRAIN:
      return hidden
    else:
      return self._hidden_to_logits(hidden, context)

  def _hidden_to_logits(self, hidden, context):
    """Actually compute the logits over the entire vocab."""
    head_size = self._head_cluster.end_token_id
    # Note that computing the log softmax is equivalent to computing the logits.
    head_log_softmax = self._head_cluster.compute_log_softmax(hidden, context)
    logits = [
        self._head_cluster.get_log_softmax_prefix(head_log_softmax, head_size)
    ]

    for i, cluster in enumerate(self._tail_clusters):
      tail_log_softmax = cluster.compute_log_softmax(hidden, context)
      cluster_softmax = self._head_cluster.get_log_softmax_value(
          head_log_softmax, head_size + i)
      logits.append(cluster_softmax + tail_log_softmax)
    return mtf.concat(logits, concat_dim_name=self._vocab_dim.name)

  def compute_loss(self, decoder: transformer.Unitransformer,
                   hidden: mtf.Tensor, targets: mtf.Tensor,
                   context: transformer.Context) -> mtf.Tensor:
    """Returns the loss without computing a softmax over the entire vocab."""
    loss = 0
    tail_cluster_masks = []
    for cluster in self._tail_clusters:
      cluster_mask = cluster.get_cluster_mask(targets)
      tail_cluster_masks.append(cluster_mask)

      if cluster.length_projection_factor == 1:
        targets_in_cluster = mtf.where(cluster_mask, targets, 0)
        hidden_in_cluster = mtf.where(cluster_mask, hidden, 0)
      else:
        # TODO(mmatena): Unfold the batch dim to get a super long sequence dim
        # to reduce the risk of overflowing the projection.
        proj_to_cluster_len = cluster.get_project_to_cluster_length(
            cluster_mask, dtype=targets.dtype)
        targets_in_cluster = mtf.einsum(
            [proj_to_cluster_len, targets],
            reduced_dims=[targets.shape.get_dim_by_name("length")])
        hidden_in_cluster = mtf.einsum(
            [mtf.cast(proj_to_cluster_len, hidden.dtype), hidden],
            reduced_dims=[hidden.shape.get_dim_by_name("length")])

      loss += cluster.compute_loss(decoder, hidden_in_cluster,
                                   targets_in_cluster, context)

    tail_clusters_dim = mtf.Dimension("tail_clusters", len(tail_cluster_masks))
    tail_node_targets = mtf.reduce_sum(
        mtf.stack([(self._head_cluster.end_token_id + i) *
                   mtf.cast(mask, targets.dtype)
                   for i, mask in enumerate(tail_cluster_masks)],
                  tail_clusters_dim.name),
        reduced_dim=tail_clusters_dim)
    head_targets = mtf.where(
        mtf.cast(tail_node_targets, tf.bool), tail_node_targets, targets)
    loss += self._head_cluster.compute_loss(decoder, hidden, head_targets,
                                            context)

    return loss


@gin.configurable
def adaptive_softmax_loss_fn(decoder: transformer.Unitransformer,
                             context: transformer.Context, logits: mtf.Tensor,
                             targets: mtf.Tensor,
                             output_vocab_dim: mtf.Dimension) -> mtf.Tensor:
  """Custom loss to use when training with an adaptive softmax.

  Embedding and softmax weights must be shared in order for this function to
  work. Note that label smoothing and loss_on_targets_only is not supported and
  will be silently ignored.

  Args:
    decoder: a transformer.Unitransformer
    context: a transformer.Context
    logits: an mtf.Tensor, note that this will actually be the hidden state of
      the final decoder layer
    targets: an mtf.Tensor
    output_vocab_dim: an mtf.Dimension

  Returns:
    the loss
  """
  del output_vocab_dim
  hidden = logits
  vocab_embedding = context.shared_params["embedding"]
  return vocab_embedding.compute_loss(
      decoder, hidden=hidden, targets=targets, context=context)
