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
"""Tests for mesh_tensorflow.transformer.adaptive_softmax."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mesh_tensorflow as mtf
from mesh_tensorflow.transformer import adaptive_softmax
import mock
import numpy as np
import scipy.special
import tensorflow.compat.v1 as tf


def initialize_by_shape(shape_to_value):
  """Create an initializer with values specified by tensor shape."""

  def initialize(shape, dtype):
    shape = tuple(shape)
    if shape not in shape_to_value:
      raise ValueError(
          'Shape {} not found in shape to value map.'.format(shape))
    return tf.reshape(
        tf.constant(shape_to_value[tuple(shape)], dtype=dtype), shape)

  return initialize


def _log_softmax(logits):
  log_z = scipy.special.logsumexp(logits)
  return logits - log_z


def _softmax_cross_entropy_with_logits(logits, target):
  soft_target = np.zeros(len(logits))
  soft_target[target] = 1
  return -np.sum(_log_softmax(logits) * soft_target)


class AdaptiveSoftmaxTest(tf.test.TestCase):

  def setUp(self):
    super(AdaptiveSoftmaxTest, self).setUp()
    self.graph = mtf.Graph()
    self.mesh = mtf.Mesh(self.graph, 'mtf_mesh')
    self.variable_dtype = mtf.VariableDType(activation_dtype=tf.float32)

    self.addCleanup(mock.patch.stopall)
    self.initializer_mock = mock.MagicMock()
    random_normal_initializer_mock = mock.patch.object(
        tf, 'random_normal_initializer').start()
    random_normal_initializer_mock.return_value = self.initializer_mock

  def _export_to_tf_tensor(self, mtf_tensor):
    mesh_impl = mtf.placement_mesh_impl.PlacementMeshImpl(
        shape=[], layout={}, devices=[''])
    lowering = mtf.Lowering(self.graph, {self.mesh: mesh_impl})
    return lowering, lowering.export_to_tf_tensor(mtf_tensor)

  def test_adaptive_softmax_loss_fn_tailClustersAllProject_correctlyComputesTheLoss(
      self):
    # Arrange.
    seq_len = 16
    vocab_size = 8
    model_size = 2

    vocab_dim = mtf.Dimension('vocab', vocab_size)
    model_dim = mtf.Dimension('model', model_size)
    length_dim = mtf.Dimension('length', seq_len)

    decoder = mock.MagicMock()
    decoder.z_loss = 0.0
    decoder.loss_denominator = mock.MagicMock()
    decoder.loss_denominator.return_value = 1.0

    # 7 tokens in head cluster
    # 5 tokens in tail cluster 1
    # 4 tokens in tail cluster 2
    targets_array = [2, 4, 4, 6, 2, 5, 7, 5, 2, 1, 6, 7, 0, 0, 3, 2]
    targets = tf.constant(targets_array, dtype=tf.int32)
    hidden = tf.constant(
        [[0, -10], [1, -11], [2, -12], [3, -13], [4, -14], [5, -15], [6, -16],
         [7, -17], [8, -18], [9, -19], [10, -20], [11, -21], [12, -22],
         [13, -23], [14, -24], [15, -25]],
        dtype=tf.float32)

    mtf_targets = mtf.import_tf_tensor(
        self.mesh, targets, shape=mtf.Shape([length_dim]))
    mtf_hidden = mtf.import_tf_tensor(
        self.mesh, hidden, shape=mtf.Shape([length_dim, model_dim]))

    self.initializer_mock.side_effect = initialize_by_shape({
        (5, 2): [[1, 6], [2, 7], [3, 8], [4, 9], [5, 10]],
        (3, 2): [[11, 14], [12, 15], [13, 16]],
        (2, 1): [[17, 18]],
        (1, 2): [[19], [20]],
    })

    vocab_embedding = adaptive_softmax.AdaptiveSoftmaxVocabEmbedding(
        self.mesh,
        vocab_dim,
        output_dim=model_dim,
        variable_dtype=self.variable_dtype,
        name='embedding',
        ensemble_dim=None,
        clusters=[{
            'token_count': 3,
            'embedding_size': 2
        }, {
            'token_count': 3,
            'embedding_size': 2,
            'length_projection_factor': 0.5,
        }, {
            'token_count': 2,
            'embedding_size': 1,
            'length_projection_factor': 0.125,
        }])

    context = mock.MagicMock()
    context.activation_dtype = tf.float32
    context.shared_params = {'embedding': vocab_embedding}

    # Act.
    mtf_loss = adaptive_softmax.adaptive_softmax_loss_fn(
        decoder, context, mtf_hidden, mtf_targets, output_vocab_dim=None)
    lowering, loss = self._export_to_tf_tensor(mtf_loss)

    self.evaluate(tf.global_variables_initializer())
    self.evaluate(lowering.copy_masters_to_slices())
    actual_loss, = self.evaluate([loss])

    # Assert.
    def expected_head_loss(position, label):
      factor = model_dim.size**-0.5
      logits = [
          factor * (1 * position - 6 * (10 + position)),
          factor * (2 * position - 7 * (10 + position)),
          factor * (3 * position - 8 * (10 + position)),
          factor * (4 * position - 9 * (10 + position)),
          factor * (5 * position - 10 * (10 + position)),
      ]
      return _softmax_cross_entropy_with_logits(logits, label)

    expected_head_labels = [2, 3, 3, 4, 2, 3, 4, 3, 2, 1, 4, 4, 0, 0, 3, 2]
    expected_head_loss = sum(
        expected_head_loss(position, expected_label)
        for position, expected_label in enumerate(expected_head_labels)
        if expected_label)

    def expected_tail_cluster_1_loss(position):
      factor = model_dim.size**-0.5
      logits = [
          factor * (11 * position - 14 * (10 + position)),
          factor * (12 * position - 15 * (10 + position)),
          factor * (13 * position - 16 * (10 + position)),
      ]
      first_token_in_cluster_id = 3
      return _softmax_cross_entropy_with_logits(
          logits, targets_array[position] - first_token_in_cluster_id)

    expected_tail_cluster_1_loss = sum([
        expected_tail_cluster_1_loss(position=1),
        expected_tail_cluster_1_loss(position=2),
        expected_tail_cluster_1_loss(position=5),
        expected_tail_cluster_1_loss(position=7),
        expected_tail_cluster_1_loss(position=14),
    ])

    def expected_tail_cluster_2_loss(position):
      factor = model_dim.size**-0.5
      logits = [
          factor * (17 * 19 * position - 17 * 20 * (10 + position)),
          factor * (18 * 19 * position - 18 * 20 * (10 + position)),
      ]
      first_token_in_cluster_id = 6
      return _softmax_cross_entropy_with_logits(
          logits, targets_array[position] - first_token_in_cluster_id)

    # Due to the length_projection_factor of 1/8, only 2 tokens will be counted
    # despite there being 4 tokens in this cluster.
    expected_tail_cluster_2_loss = sum([
        expected_tail_cluster_2_loss(position=3),
        expected_tail_cluster_2_loss(position=6),
    ])

    expected_loss = (
        expected_head_loss + expected_tail_cluster_1_loss +
        expected_tail_cluster_2_loss)

    self.assertAllClose(actual_loss, expected_loss)

  def test_adaptive_softmax_loss_fn_tailClusterDoesNotProject_correctlyComputesTheLoss(
      self):
    # Arrange.
    seq_len = 16
    vocab_size = 8
    model_size = 2

    vocab_dim = mtf.Dimension('vocab', vocab_size)
    model_dim = mtf.Dimension('model', model_size)
    length_dim = mtf.Dimension('length', seq_len)

    decoder = mock.MagicMock()
    decoder.z_loss = 0.0
    decoder.loss_denominator = mock.MagicMock()
    decoder.loss_denominator.return_value = 1.0

    # 7 tokens in head cluster
    # 5 tokens in tail cluster 1
    # 4 tokens in tail cluster 2
    targets_array = [2, 4, 4, 6, 2, 5, 7, 5, 2, 1, 6, 7, 0, 0, 3, 2]
    targets = tf.constant(targets_array, dtype=tf.int32)
    hidden = tf.constant(
        [[0, -10], [1, -11], [2, -12], [3, -13], [4, -14], [5, -15], [6, -16],
         [7, -17], [8, -18], [9, -19], [10, -20], [11, -21], [12, -22],
         [13, -23], [14, -24], [15, -25]],
        dtype=tf.float32)

    mtf_targets = mtf.import_tf_tensor(
        self.mesh, targets, shape=mtf.Shape([length_dim]))
    mtf_hidden = mtf.import_tf_tensor(
        self.mesh, hidden, shape=mtf.Shape([length_dim, model_dim]))

    self.initializer_mock.side_effect = initialize_by_shape({
        (5, 2): [[1, 6], [2, 7], [3, 8], [4, 9], [5, 10]],
        (3, 2): [[11, 14], [12, 15], [13, 16]],
        (2, 1): [[17, 18]],
        (1, 2): [[19], [20]],
    })

    vocab_embedding = adaptive_softmax.AdaptiveSoftmaxVocabEmbedding(
        self.mesh,
        vocab_dim,
        output_dim=model_dim,
        variable_dtype=self.variable_dtype,
        name='embedding',
        ensemble_dim=None,
        clusters=[{
            'token_count': 3,
            'embedding_size': 2
        }, {
            'token_count': 3,
            'embedding_size': 2,
            'length_projection_factor': 0.5,
        }, {
            'token_count': 2,
            'embedding_size': 1,
            'length_projection_factor': 1,
        }])

    context = mock.MagicMock()
    context.activation_dtype = tf.float32
    context.shared_params = {'embedding': vocab_embedding}

    # Act.
    mtf_loss = adaptive_softmax.adaptive_softmax_loss_fn(
        decoder, context, mtf_hidden, mtf_targets, output_vocab_dim=None)
    lowering, loss = self._export_to_tf_tensor(mtf_loss)

    self.evaluate(tf.global_variables_initializer())
    self.evaluate(lowering.copy_masters_to_slices())
    actual_loss, = self.evaluate([loss])

    # Assert.
    def expected_head_loss(position, label):
      factor = model_dim.size**-0.5
      logits = [
          factor * (1 * position - 6 * (10 + position)),
          factor * (2 * position - 7 * (10 + position)),
          factor * (3 * position - 8 * (10 + position)),
          factor * (4 * position - 9 * (10 + position)),
          factor * (5 * position - 10 * (10 + position)),
      ]
      return _softmax_cross_entropy_with_logits(logits, label)

    expected_head_labels = [2, 3, 3, 4, 2, 3, 4, 3, 2, 1, 4, 4, 0, 0, 3, 2]
    expected_head_loss = sum(
        expected_head_loss(position, expected_label)
        for position, expected_label in enumerate(expected_head_labels)
        if expected_label)

    def expected_tail_cluster_1_loss(position):
      factor = model_dim.size**-0.5
      logits = [
          factor * (11 * position - 14 * (10 + position)),
          factor * (12 * position - 15 * (10 + position)),
          factor * (13 * position - 16 * (10 + position)),
      ]
      first_token_in_cluster_id = 3
      return _softmax_cross_entropy_with_logits(
          logits, targets_array[position] - first_token_in_cluster_id)

    expected_tail_cluster_1_loss = sum([
        expected_tail_cluster_1_loss(position=1),
        expected_tail_cluster_1_loss(position=2),
        expected_tail_cluster_1_loss(position=5),
        expected_tail_cluster_1_loss(position=7),
        expected_tail_cluster_1_loss(position=14),
    ])

    def expected_tail_cluster_2_loss(position):
      factor = model_dim.size**-0.5
      logits = [
          factor * (17 * 19 * position - 17 * 20 * (10 + position)),
          factor * (18 * 19 * position - 18 * 20 * (10 + position)),
      ]
      first_token_in_cluster_id = 6
      return _softmax_cross_entropy_with_logits(
          logits, targets_array[position] - first_token_in_cluster_id)

    expected_tail_cluster_2_loss = sum([
        expected_tail_cluster_2_loss(position=3),
        expected_tail_cluster_2_loss(position=6),
        expected_tail_cluster_2_loss(position=10),
        expected_tail_cluster_2_loss(position=11),
    ])

    expected_loss = (
        expected_head_loss + expected_tail_cluster_1_loss +
        expected_tail_cluster_2_loss)

    self.assertAllClose(actual_loss, expected_loss)

  def test_hidden_to_logits_returnsHiddenDuringTraining(self):
    # Arrange.
    seq_len = 2
    vocab_size = 3
    model_size = 2

    vocab_dim = mtf.Dimension('vocab', vocab_size)
    model_dim = mtf.Dimension('model', model_size)
    length_dim = mtf.Dimension('length', seq_len)

    context = mock.MagicMock()
    context.activation_dtype = tf.float32
    context.mode = tf.estimator.ModeKeys.TRAIN

    embeddings = tf.constant([[1, 0], [0, 2]], dtype=tf.float32)
    mtf_embeddings = mtf.import_tf_tensor(
        self.mesh, embeddings, shape=mtf.Shape([length_dim, model_dim]))

    self.initializer_mock.side_effect = initialize_by_shape({
        (3, 2): [[1, 6], [2, 7], [3, 8]],
    })

    vocab_embedding = adaptive_softmax.AdaptiveSoftmaxVocabEmbedding(
        self.mesh,
        vocab_dim,
        output_dim=model_dim,
        variable_dtype=self.variable_dtype,
        name='embedding',
        ensemble_dim=None,
        clusters=[{
            'token_count': 3,
            'embedding_size': 2
        }])
    mtf_logits = vocab_embedding.hidden_to_logits(
        mtf_embeddings, context=context)

    self.assertEqual(mtf_logits, mtf_embeddings)

  def test_hidden_to_logits_returnsCorrectLogitsDuringEval(self):
    # Arrange.
    seq_len = 2
    vocab_size = 8
    model_size = 2

    vocab_dim = mtf.Dimension('vocab', vocab_size)
    model_dim = mtf.Dimension('model', model_size)
    length_dim = mtf.Dimension('length', seq_len)

    context = mock.MagicMock()
    context.activation_dtype = tf.float32
    context.mode = tf.estimator.ModeKeys.EVAL

    embeddings = tf.constant([[1, 0], [0, 2]], dtype=tf.float32)
    mtf_embeddings = mtf.import_tf_tensor(
        self.mesh, embeddings, shape=mtf.Shape([length_dim, model_dim]))

    self.initializer_mock.side_effect = initialize_by_shape({
        (5, 2): [[1, 6], [2, 7], [3, 8], [4, 9], [5, 10]],
        (3, 2): [[11, 14], [12, 15], [13, 16]],
        (2, 1): [[17, 18]],
        (1, 2): [[19], [20]],
    })

    vocab_embedding = adaptive_softmax.AdaptiveSoftmaxVocabEmbedding(
        self.mesh,
        vocab_dim,
        output_dim=model_dim,
        variable_dtype=self.variable_dtype,
        name='embedding',
        ensemble_dim=None,
        clusters=[{
            'token_count': 3,
            'embedding_size': 2
        }, {
            'token_count': 3,
            'embedding_size': 2,
            'length_projection_factor': 0.5,
        }, {
            'token_count': 2,
            'embedding_size': 1,
            'length_projection_factor': 0.125,
        }])

    # Act.
    mtf_logits = vocab_embedding.hidden_to_logits(
        mtf_embeddings, context=context)
    lowering, logits = self._export_to_tf_tensor(mtf_logits)

    self.evaluate(tf.global_variables_initializer())
    self.evaluate(lowering.copy_masters_to_slices())
    actual_logits, = self.evaluate([logits])

    # Assert.
    def scaled_log_softmax(a):
      a = np.array(a, dtype=float) * model_dim.size**-0.5
      return _log_softmax(a)

    head_log_softmax1 = scaled_log_softmax([1, 2, 3, 4, 5])
    head_log_softmax2 = scaled_log_softmax([2 * 6, 2 * 7, 2 * 8, 2 * 9, 2 * 10])

    expected_logits = [
        np.concatenate([
            head_log_softmax1[:3],
            head_log_softmax1[3] + scaled_log_softmax([11, 12, 13]),
            head_log_softmax1[4] + scaled_log_softmax([17 * 19, 18 * 19]),
        ]),
        np.concatenate([
            head_log_softmax2[:3],
            head_log_softmax2[3] + scaled_log_softmax([2 * 14, 2 * 15, 2 * 16]),
            head_log_softmax2[4] +
            scaled_log_softmax([2 * 17 * 20, 2 * 18 * 20]),
        ]),
    ]

    self.assertAllClose(actual_logits, expected_logits, atol=5e-5)

  def test_ids_to_embedding_correctlyEmbeds(self):
    seq_len = 6
    vocab_size = 5
    model_size = 2

    vocab_dim = mtf.Dimension('vocab', vocab_size)
    model_dim = mtf.Dimension('model', model_size)
    length_dim = mtf.Dimension('length', seq_len)

    ids = tf.constant([0, 1, 2, 3, 4, 0], dtype=tf.int32)
    mtf_ids = mtf.import_tf_tensor(
        self.mesh, ids, shape=mtf.Shape([length_dim]))

    self.initializer_mock.side_effect = initialize_by_shape({
        (3, 2): [[0, 1], [2, 0], [-1000, -4000]],
        (3, 1): [[1], [2], [3]],
        (1, 2): [[1], [2]],
    })

    vocab_embedding = adaptive_softmax.AdaptiveSoftmaxVocabEmbedding(
        self.mesh,
        vocab_dim,
        output_dim=model_dim,
        variable_dtype=self.variable_dtype,
        name='embedding',
        ensemble_dim=None,
        clusters=[{
            'token_count': 2,
            'embedding_size': 2
        }, {
            'token_count': 3,
            'embedding_size': 1
        }])

    mtf_embedding = vocab_embedding.ids_to_embedding(mtf_ids, context=None)

    mesh_impl = mtf.placement_mesh_impl.PlacementMeshImpl(
        shape=[], layout={}, devices=[''])
    lowering = mtf.Lowering(self.graph, {self.mesh: mesh_impl})
    actual_embedding = lowering.export_to_tf_tensor(mtf_embedding)

    self.evaluate(tf.global_variables_initializer())
    self.evaluate(lowering.copy_masters_to_slices())
    actual, = self.evaluate([actual_embedding])

    self.assertAllClose(actual,
                        [[0, 1], [2, 0], [1, 2], [2, 4], [3, 6], [0, 1]])

  def test_constructor_tokenCountsDontSumToVocabSize_raisesValueError(self):
    vocab_dim = mtf.Dimension('vocab', 5)
    model_dim = mtf.Dimension('model', 2)

    with self.assertRaises(ValueError):
      adaptive_softmax.AdaptiveSoftmaxVocabEmbedding(
          self.mesh,
          vocab_dim,
          output_dim=model_dim,
          variable_dtype=self.variable_dtype,
          name='embedding',
          ensemble_dim=None,
          clusters=[{
              'token_count': 3,
              'embedding_size': 2
          }, {
              'token_count': 3,
              'embedding_size': 1
          }])

  def test_constructor_projectFactorNotWithinZeroAndOne_raisesValueError(self):
    vocab_dim = mtf.Dimension('vocab', 3)
    model_dim = mtf.Dimension('model', 2)

    with self.assertRaises(ValueError):
      adaptive_softmax.AdaptiveSoftmaxVocabEmbedding(
          self.mesh,
          vocab_dim,
          output_dim=model_dim,
          variable_dtype=self.variable_dtype,
          name='embedding',
          ensemble_dim=None,
          clusters=[{
              'token_count': 3,
              'embedding_size': 2,
              'length_projection_factor': 1.1,
          }])


if __name__ == '__main__':
  tf.test.main()
