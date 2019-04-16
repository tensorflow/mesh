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


def padded_neg_log_perplexity(labels, logits):
  weights = tf.to_float(tf.not_equal(labels, 0))
  xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
      labels=labels, logits=logits)
  return tf.metrics.mean(-xent, weights)


# TODO(katherinelee): Look at other bleu implementations.
def bleu(labels, logits):
  outputs = tf.to_int32(tf.argmax(logits, axis=-1))
  # Convert the outputs and labels to a [batch_size, input_length] tensor.
  bleu_score = tf.py_function(
      bleu_hook.compute_bleu, (labels, outputs), tf.float32)
  return bleu_score, tf.constant(1.0)
