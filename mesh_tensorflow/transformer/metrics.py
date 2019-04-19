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

"""Functions for computing metrics.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensor2tensor.utils import bleu_hook
import tensorflow as tf


# TODO(katherinelee): Look at other bleu implementations.
def bleu(labels, outputs):
  bleu_score = tf.py_function(
      bleu_hook.compute_bleu, (labels, outputs), tf.float32)
  return bleu_score, tf.constant(1.0)


def token_accuracy(labels, outputs):
  """Compute tokenwise (elementwise) accuracy.

  Args:
    labels: ground-truth labels, shape=(batch, seq_length)
    outputs: predicted tokens, shape=(batch, seq_length)
  Returns:
    Two ops, one for getting the current average accuracy and another for
    updating the running average estimate.
  """
  weights = tf.to_float(tf.not_equal(labels, 0))
  return tf.metrics.accuracy(labels, outputs, weights=weights)


def sequence_accuracy(labels, outputs):
  """Compute the sequence-level accuracy.

  A sequence is only considered correct if all of its entries were predicted
  correctly.

  Args:
    labels: ground-truth labels, shape=(batch, packed_seq_length)
    outputs: predicted tokens, shape=(batch, seq_length)
  Returns:
    Two ops, one for getting the current average accuracy and another for
    updating the running average estimate.
  """
  # A sequence is correct if all of the non-padded entries are correct
  all_correct = tf.reduce_all(
      tf.logical_or(tf.equal(labels, outputs), tf.equal(labels, 0)), axis=-1
  )
  return tf.metrics.mean(all_correct)
