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
"""Tests for mesh_tensorflow.transformer.vocab_embeddings."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mesh_tensorflow as mtf
from mesh_tensorflow.transformer import vocab_embeddings
import mock
import numpy as np
import scipy.misc
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


class FactorizedVocabEmbeddingTest(tf.test.TestCase):

  def setUp(self):
    super(FactorizedVocabEmbeddingTest, self).setUp()
    self.graph = mtf.Graph()
    self.mesh = mtf.Mesh(self.graph, 'mtf_mesh')
    self.variable_dtype = mtf.VariableDType(activation_dtype=tf.float32)

    self.addCleanup(mock.patch.stopall)
    self.initializer_mock = mock.MagicMock()
    random_normal_initializer_mock = mock.patch.object(
        tf, 'random_normal_initializer').start()
    random_normal_initializer_mock.return_value = self.initializer_mock

  def test_ids_to_embedding_correctlyEmbeds(self):
    seq_len = 4
    vocab_size = 3
    model_size = 2
    inner_dimension_size = 1

    vocab_dim = mtf.Dimension('vocab', vocab_size)
    model_dim = mtf.Dimension('model', model_size)
    length_dim = mtf.Dimension('length', seq_len)

    ids = tf.constant([0, 1, 2, 1], dtype=tf.int32)
    mtf_ids = mtf.import_tf_tensor(
        self.mesh, ids, shape=mtf.Shape([length_dim]))

    def initialize(shape, dtype):
      return tf.reshape(1 + tf.range(np.prod(shape), dtype=dtype), shape)

    self.initializer_mock.side_effect = initialize

    vocab_embedding = vocab_embeddings.FactorizedVocabEmbedding(
        self.mesh,
        vocab_dim,
        output_dim=model_dim,
        variable_dtype=self.variable_dtype,
        name='embedding',
        ensemble_dim=None,
        inner_dimension_size=inner_dimension_size)

    mtf_embedding = vocab_embedding.ids_to_embedding(mtf_ids, context=None)

    mesh_impl = mtf.placement_mesh_impl.PlacementMeshImpl(
        shape=[], layout={}, devices=[''])
    lowering = mtf.Lowering(self.graph, {self.mesh: mesh_impl})
    actual_embedding = lowering.export_to_tf_tensor(mtf_embedding)

    self.evaluate(tf.global_variables_initializer())
    self.evaluate(lowering.copy_masters_to_slices())
    actual = self.evaluate([actual_embedding])[0]

    self.assertAllClose(actual, [[1, 2], [2, 4], [3, 6], [2, 4]])

  def test_hidden_to_logits_computesLogitsCorrectly(self):
    seq_len = 4
    vocab_size = 3
    model_size = 2
    inner_dimension_size = 1

    vocab_dim = mtf.Dimension('vocab', vocab_size)
    model_dim = mtf.Dimension('model', model_size)
    length_dim = mtf.Dimension('length', seq_len)

    embeddings = tf.constant([[1, 0], [0, 1], [1, 1], [2, 1]], dtype=tf.float32)
    mtf_embeddings = mtf.import_tf_tensor(
        self.mesh, embeddings, shape=mtf.Shape([length_dim, model_dim]))

    def initialize(shape, dtype):
      return tf.reshape(1 + tf.range(np.prod(shape), dtype=dtype), shape)

    self.initializer_mock.side_effect = initialize

    vocab_embedding = vocab_embeddings.FactorizedVocabEmbedding(
        self.mesh,
        vocab_dim,
        output_dim=model_dim,
        variable_dtype=self.variable_dtype,
        name='embedding',
        ensemble_dim=None,
        inner_dimension_size=inner_dimension_size)

    mtf_logits = vocab_embedding.hidden_to_logits(mtf_embeddings, context=None)

    mesh_impl = mtf.placement_mesh_impl.PlacementMeshImpl(
        shape=[], layout={}, devices=[''])
    lowering = mtf.Lowering(self.graph, {self.mesh: mesh_impl})
    actual_logits = lowering.export_to_tf_tensor(mtf_logits)

    self.evaluate(tf.global_variables_initializer())
    self.evaluate(lowering.copy_masters_to_slices())
    actual = self.evaluate([actual_logits])[0]

    self.assertAllClose(
        actual, model_size**-0.5 *
        np.array([[1, 2, 3], [2, 4, 6], [3, 6, 9], [4, 8, 12]]))


class AdaptiveVocabEmbeddingTest(tf.test.TestCase):

  def setUp(self):
    super(AdaptiveVocabEmbeddingTest, self).setUp()
    self.graph = mtf.Graph()
    self.mesh = mtf.Mesh(self.graph, 'mtf_mesh')
    self.variable_dtype = mtf.VariableDType(activation_dtype=tf.float32)

    self.addCleanup(mock.patch.stopall)
    self.initializer_mock = mock.MagicMock()
    random_normal_initializer_mock = mock.patch.object(
        tf, 'random_normal_initializer').start()
    random_normal_initializer_mock.return_value = self.initializer_mock

  def test_constructor_tokenCountsDontSumToVocabSize_raisesValueError(self):
    vocab_dim = mtf.Dimension('vocab', 5)
    model_dim = mtf.Dimension('model', 2)

    with self.assertRaises(ValueError):
      vocab_embeddings.AdaptiveVocabEmbedding(
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
        (2, 2): [[0, 1], [2, 0]],
        (3, 1): [[1], [2], [3]],
        (1, 2): [[1], [2]],
    })

    vocab_embedding = vocab_embeddings.AdaptiveVocabEmbedding(
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
    actual = self.evaluate([actual_embedding])[0]

    self.assertAllClose(actual,
                        [[0, 1], [2, 0], [1, 2], [2, 4], [3, 6], [0, 1]])

  def test_hidden_to_logits_computesLogitsCorrectly(self):
    seq_len = 4
    vocab_size = 5
    model_size = 2

    vocab_dim = mtf.Dimension('vocab', vocab_size)
    model_dim = mtf.Dimension('model', model_size)
    length_dim = mtf.Dimension('length', seq_len)

    embeddings = tf.constant([[1, 0], [0, 1], [1, 1], [2, 1]], dtype=tf.float32)
    mtf_embeddings = mtf.import_tf_tensor(
        self.mesh, embeddings, shape=mtf.Shape([length_dim, model_dim]))

    self.initializer_mock.side_effect = initialize_by_shape({
        (2, 2): [[0, 1], [2, 0]],
        (3, 1): [[1], [2], [3]],
        (1, 2): [[1], [2]],
    })

    vocab_embedding = vocab_embeddings.AdaptiveVocabEmbedding(
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

    mtf_logits = vocab_embedding.hidden_to_logits(mtf_embeddings, context=None)

    mesh_impl = mtf.placement_mesh_impl.PlacementMeshImpl(
        shape=[], layout={}, devices=[''])
    lowering = mtf.Lowering(self.graph, {self.mesh: mesh_impl})
    actual_logits = lowering.export_to_tf_tensor(mtf_logits)

    self.evaluate(tf.global_variables_initializer())
    self.evaluate(lowering.copy_masters_to_slices())
    actual = self.evaluate([actual_logits])[0]

    self.assertAllClose(
        actual,
        model_size**-0.5 * np.array([[0, 2, 1, 2, 3], [1, 0, 2, 4, 6],
                                     [1, 2, 3, 6, 9], [1, 4, 4, 8, 12]]))


class MixtureOfSoftmaxesTest(tf.test.TestCase):

  def setUp(self):
    super(MixtureOfSoftmaxesTest, self).setUp()
    self.graph = mtf.Graph()
    self.mesh = mtf.Mesh(self.graph, 'mtf_mesh')
    self.variable_dtype = mtf.VariableDType(activation_dtype=tf.float32)

    self.addCleanup(mock.patch.stopall)
    self.initializer_mock = mock.MagicMock()
    random_normal_initializer_mock = mock.patch.object(
        tf, 'random_normal_initializer').start()
    random_normal_initializer_mock.return_value = self.initializer_mock

  def test_ids_to_embedding_correctlyEmbeds(self):
    seq_len = 4
    vocab_size = 4
    model_size = 3
    num_softmaxes = 1

    vocab_dim = mtf.Dimension('vocab', vocab_size)
    model_dim = mtf.Dimension('model', model_size)
    length_dim = mtf.Dimension('length', seq_len)

    ids = tf.constant([0, 1, 2, 3], dtype=tf.int32)
    mtf_ids = mtf.import_tf_tensor(
        self.mesh, ids, shape=mtf.Shape([length_dim]))

    self.initializer_mock.side_effect = initialize_by_shape({
        # Embedding weights.
        (4, 3): [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 2]],
        # Mixture weights.
        (1, 3): [[1, 0, 0]],
        # Context weights
        (1, 3, 3): [[[1, 0, 0], [0, 1, 0], [0, 0, 1]],],
    })

    vocab_embedding = vocab_embeddings.MixtureOfSoftmaxes(
        self.mesh,
        vocab_dim,
        output_dim=model_dim,
        variable_dtype=self.variable_dtype,
        name='embedding',
        ensemble_dim=None,
        num_softmaxes=num_softmaxes)

    mtf_embedding = vocab_embedding.ids_to_embedding(mtf_ids, context=None)

    mesh_impl = mtf.placement_mesh_impl.PlacementMeshImpl(
        shape=[], layout={}, devices=[''])
    lowering = mtf.Lowering(self.graph, {self.mesh: mesh_impl})
    actual_embedding = lowering.export_to_tf_tensor(mtf_embedding)

    self.evaluate(tf.global_variables_initializer())
    self.evaluate(lowering.copy_masters_to_slices())
    actual = self.evaluate([actual_embedding])[0]

    self.assertAllClose(actual, [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 2]])

  def test_hidden_to_logits_computesLogitsCorrectly(self):
    seq_len = 1
    vocab_size = 4
    model_size = 3
    num_softmaxes = 2

    vocab_dim = mtf.Dimension('vocab', vocab_size)
    model_dim = mtf.Dimension('model', model_size)
    length_dim = mtf.Dimension('length', seq_len)

    embeddings = tf.constant(
        np.array([[1.0, 1.0, 2.0]]) / model_size**-0.5, dtype=tf.float32)
    mtf_embeddings = mtf.import_tf_tensor(
        self.mesh, embeddings, shape=mtf.Shape([length_dim, model_dim]))

    self.initializer_mock.side_effect = initialize_by_shape({
        # Embedding weights.
        (4, 3): [[1, 0, 0], [0, 1, 0], [0, 0, 1], [0, 0, 1]],
        # Mixture weights.
        (2, 3): [[1, 0, 0], [0, 1, 1]],
        # Context weights
        (2, 3, 3): [
            [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            [[0, 0, 1], [0, 1, 0], [1, 0, 0]],
        ],
    })

    vocab_embedding = vocab_embeddings.MixtureOfSoftmaxes(
        self.mesh,
        vocab_dim,
        output_dim=model_dim,
        variable_dtype=self.variable_dtype,
        name='embedding',
        ensemble_dim=None,
        num_softmaxes=num_softmaxes)

    mtf_logits = vocab_embedding.hidden_to_logits(mtf_embeddings, context=None)

    mesh_impl = mtf.placement_mesh_impl.PlacementMeshImpl(
        shape=[], layout={}, devices=[''])
    lowering = mtf.Lowering(self.graph, {self.mesh: mesh_impl})
    actual_logits = lowering.export_to_tf_tensor(mtf_logits)

    self.evaluate(tf.global_variables_initializer())
    self.evaluate(lowering.copy_masters_to_slices())
    actual, = self.evaluate([actual_logits])

    expected_priors = scipy.special.softmax([1, 3])
    expected_probs_1 = scipy.special.softmax(np.tanh([1, 1, 2, 2]))
    expected_probs_2 = scipy.special.softmax(np.tanh([2, 1, 1, 1]))
    expected_probs = (
        expected_priors[0] * expected_probs_1 +
        expected_priors[1] * expected_probs_2)
    expected_logits = np.log(expected_probs)

    self.assertAllClose(actual, [expected_logits])


class MixtapeTest(tf.test.TestCase):

  def setUp(self):
    super(MixtapeTest, self).setUp()
    self.graph = mtf.Graph()
    self.mesh = mtf.Mesh(self.graph, 'mtf_mesh')
    self.variable_dtype = mtf.VariableDType(activation_dtype=tf.float32)

    self.addCleanup(mock.patch.stopall)
    self.initializer_mock = mock.MagicMock()
    random_normal_initializer_mock = mock.patch.object(
        tf, 'random_normal_initializer').start()
    random_normal_initializer_mock.return_value = self.initializer_mock

  def test_ids_to_embedding_correctlyEmbeds(self):
    seq_len = 5
    vocab_size = 5
    model_size = 2
    gate_embedding_size = 1
    frequent_token_fraction = 0.4

    vocab_dim = mtf.Dimension('vocab', vocab_size)
    model_dim = mtf.Dimension('model', model_size)
    length_dim = mtf.Dimension('length', seq_len)

    context = mock.MagicMock()
    context.train = False

    ids = tf.constant([0, 1, 2, 3, 4], dtype=tf.int32)
    mtf_ids = mtf.import_tf_tensor(
        self.mesh, ids, shape=mtf.Shape([length_dim]))

    self.initializer_mock.side_effect = initialize_by_shape({
        # Embedding weights.
        (5, 2): list(range(10)),
        # Context weights.
        (4, 2, 2): list(range(16)),
        # Prior weights.
        (3, 1, 2): list(range(6)),
        # Prior vocab vector.
        (2, 1): list(range(2)),
        # Prior gates vector.
        (3, 2): list(range(6)),
        # Prior bias.
        (2, 3): list(range(6)),
    })

    vocab_embedding = vocab_embeddings.Mixtape(
        self.mesh,
        vocab_dim,
        output_dim=model_dim,
        variable_dtype=self.variable_dtype,
        name='embedding',
        ensemble_dim=None,
        gate_embedding_size=gate_embedding_size,
        frequent_token_fraction=frequent_token_fraction)

    mtf_embedding = vocab_embedding.ids_to_embedding(mtf_ids, context=None)

    mesh_impl = mtf.placement_mesh_impl.PlacementMeshImpl(
        shape=[], layout={}, devices=[''])
    lowering = mtf.Lowering(self.graph, {self.mesh: mesh_impl})
    actual_embedding = lowering.export_to_tf_tensor(mtf_embedding)

    self.evaluate(tf.global_variables_initializer())
    self.evaluate(lowering.copy_masters_to_slices())
    actual = self.evaluate([actual_embedding])[0]

    self.assertAllClose(actual, np.reshape(list(range(10)), (5, 2)))

  def test_hidden_to_logits_computesLogitsCorrectly(self):
    seq_len = 1
    vocab_size = 5
    model_size = 2
    gate_embedding_size = 1
    frequent_token_fraction = 0.4

    vocab_dim = mtf.Dimension('vocab', vocab_size)
    model_dim = mtf.Dimension('model', model_size)
    length_dim = mtf.Dimension('length', seq_len)

    context = mock.MagicMock()
    context.train = False

    embeddings = tf.constant(
        np.array([[1.0, 2.0]]) / model_size**-0.5, dtype=tf.float32)
    mtf_embeddings = mtf.import_tf_tensor(
        self.mesh, embeddings, shape=mtf.Shape([length_dim, model_dim]))

    self.initializer_mock.side_effect = initialize_by_shape({
        # Embedding weights.
        (5, 2): list(range(10)),
        # Context weights.
        (4, 2, 2): [
            [[1, 0], [0, 1]],
            [[0, 1], [1, 0]],
            [[1, 0], [0, 0]],
            [[0, 0], [0, 1]],
        ],
        # Prior weights.
        (3, 1, 2): [
            [[1, 0]],
            [[0, 1]],
            [[1, 1]],
        ],
        # Prior vocab vector.
        (2, 1): [[1], [1]],
        # Prior gates vector.
        (3, 2): [
            [1, 0],
            [0, 1],
            [1, 1],
        ],
        # Prior bias.
        (2, 3): [[1, 2, 3], [3, 4, 5]],
    })

    vocab_embedding = vocab_embeddings.Mixtape(
        self.mesh,
        vocab_dim,
        output_dim=model_dim,
        variable_dtype=self.variable_dtype,
        name='embedding',
        ensemble_dim=None,
        gate_embedding_size=gate_embedding_size,
        frequent_token_fraction=frequent_token_fraction,
        noise_std_dev=0.0)

    mtf_logits = vocab_embedding.hidden_to_logits(
        mtf_embeddings, context=context)

    mesh_impl = mtf.placement_mesh_impl.PlacementMeshImpl(
        shape=[], layout={}, devices=[''])
    lowering = mtf.Lowering(self.graph, {self.mesh: mesh_impl})
    actual_logits = lowering.export_to_tf_tensor(mtf_logits)

    self.evaluate(tf.global_variables_initializer())
    self.evaluate(lowering.copy_masters_to_slices())
    actual, = self.evaluate([actual_logits])

    self.assertAllClose(actual,
                        [[0.905462, 4.390559, 6.575162, 9.513036, 12.450909]])


if __name__ == '__main__':
  tf.test.main()
