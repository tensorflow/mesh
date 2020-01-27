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
"""Functions and classes related to optimization (weight updates)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mesh_tensorflow as mtf
import mesh_tensorflow.optimize as mtf_optimize

import tensorflow.compat.v1 as tf


def clip_by_global_norm(grads, clip_norm):
  """Clip the grads by global norm."""
  global_norm = mtf.sqrt(
      mtf.add_n([mtf.reduce_sum(mtf.square(t)) for t in grads if t is not None
                ]))
  multiplier = clip_norm / mtf.maximum(global_norm, clip_norm)
  clipped_grads = [None if t is None else t * multiplier for t in grads]
  return clipped_grads, global_norm


def create_optimizer(loss, init_lr, num_train_steps, num_warmup_steps,
                     max_optimized_variable_size=None,
                     optimizer="adam",
                     clip_gradients=True):
  """Creates an optimizer training op."""
  global_step = tf.train.get_or_create_global_step()
  mesh = loss.mesh

  if init_lr:
    # Implements linear decay of the learning rate.
    learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)
    learning_rate = tf.train.polynomial_decay(
        learning_rate,
        global_step,
        num_train_steps,
        end_learning_rate=0.0,
        power=1.0,
        cycle=False)
    # Implements linear warmup. I.e., if global_step < num_warmup_steps, the
    # learning rate will be `global_step/num_warmup_steps * init_lr`.
    if num_warmup_steps:
      global_steps_int = tf.cast(global_step, tf.int32)
      warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

      global_steps_float = tf.cast(global_steps_int, tf.float32)
      warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

      warmup_percent_done = global_steps_float / warmup_steps_float
      warmup_learning_rate = init_lr * warmup_percent_done

      is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
      learning_rate = ((1.0 - is_warmup) * learning_rate +
                       is_warmup * warmup_learning_rate)

    mtf_learning_rate = mtf.import_tf_tensor(mesh, learning_rate, [])
  else:
    if optimizer == "adam":
      raise ValueError("Adam does not have a default learning rate")
    learning_rate = None
    mtf_learning_rate = None

  # It is recommended that you use this optimizer for fine tuning, since this
  # is how the model was trained (note that the Adam m/v variables are NOT
  # loaded from init_checkpoint.)
  if optimizer == "adam":
    optimizer = mtf_optimize.AdamWeightDecayOptimizer(
        learning_rate=mtf_learning_rate,
        weight_decay_rate=0.01,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-6,
        exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])
  elif optimizer == "adafactor":
    optimizer = mtf_optimize.AdafactorOptimizer(
        learning_rate=learning_rate,
        min_dim_size_to_factor=32)
  else:
    raise ValueError("unknown optimizer")

  trainable_variables = mesh.graph.trainable_variables
  if max_optimized_variable_size:
    trainable_variables = [t for t in trainable_variables
                           if t.shape.size <= max_optimized_variable_size]

  var_grads = mtf.gradients(
      [loss], [v.outputs[0] for v in trainable_variables])

  # This is how the model was pre-trained.
  if clip_gradients:
    (var_grads, _) = clip_by_global_norm(
        var_grads, clip_norm=mtf.constant(mesh, 1.0, dtype=tf.float32))

  update_ops = optimizer.apply_grads(var_grads, trainable_variables)

  return learning_rate, update_ops
