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

"""Mesh Tensorflow Optimizers."""


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import gin
from mesh_tensorflow import layers
from mesh_tensorflow import ops_with_redefined_builtins as mtf
import tensorflow.compat.v1 as tf


def make_optimizer(hparams, lr):
  if hparams.optimizer == "SGD":
    return SgdOptimizer(lr)
  elif hparams.optimizer == "Adafactor":
    return adafactor_optimizer_from_hparams(hparams, lr)
  else:
    raise ValueError("Unknown Optimizer")


class Optimizer(object):
  """Base optimizer class.

  Constructor of subclasses must take `learning_rate` as an argument.
  """

  def apply_grads(self, grads, variables):
    """Apply gradients to variables.

    Call this function externally instead of apply_grad().  This causes the
    operations to be combined, which is necessary for stacking variables
    see mtf.rewrite_stack_variables().

    Args:
      grads: a list of Tensor
      variables: a list of Variables
    Returns:
      a list of Operations
    """
    ops = []
    for grad, var in zip(grads, variables):
      ops.extend(self.apply_grad(grad, var))
    if not ops:
      return ops
    return variables[0].graph.combine_assignments(ops)

  def apply_grad(self, grad, var):
    """Update variable and accumulators.

    Args:
      grad: a Tensor
      var: a Variablle
    Returns:
      a list of Operations
    """
    raise ValueError("apply_grad not implemented %s %s" % (grad, var))


@gin.configurable
class SgdOptimizer(Optimizer):
  """Optimizer implementing SGD."""

  def __init__(self, learning_rate):
    self._lr = learning_rate

  @property
  def lr(self):
    return self._lr

  def apply_grad(self, grad, var):
    if grad is None:
      tf.logging.warning("Gradient is None for variable %s" % var.name)
      return []
    # It is critical to use assign_sub instead of mtf.assign(var - ...)
    #  for the case of bfloat16 activations, so as to avoid repeatedly rounding
    #  the slice value, which results in poor quality.
    return [mtf.assign_sub(var, grad * self.lr)]


@gin.configurable
class MomentumOptimizer(Optimizer):
  """SGD with momentum."""

  def __init__(self, learning_rate, momentum):
    self._lr = learning_rate
    self._momentum = momentum

  @property
  def lr(self):
    return self._lr

  @property
  def momentum(self):
    return self._momentum

  def apply_grad(self, grad, var):
    if grad is None:
      tf.logging.warning("Gradient is None for variable %s" % var.name)
      return []

    updates = []
    v = mtf.get_variable(
        var.mesh, var.name + "_momentum_v", var.shape,
        dtype=var.dtype, initializer=tf.zeros_initializer(), trainable=False)

    with tf.variable_scope(var.name + "/sgd_momentum"):
      updates.append(mtf.assign(v, grad * self.lr + v * self.momentum))
      updates.append(mtf.assign_sub(var, v))

    return updates


@gin.configurable
class AdamWeightDecayOptimizer(Optimizer):
  """A basic Adam optimizer that includes "correct" L2 weight decay."""

  def __init__(self,
               learning_rate,
               weight_decay_rate=0.0,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-6,
               exclude_from_weight_decay=None):
    """Constructs a AdamWeightDecayOptimizer."""

    self.learning_rate = learning_rate
    self.weight_decay_rate = weight_decay_rate
    self.beta_1 = beta_1
    self.beta_2 = beta_2
    self.epsilon = epsilon
    self.exclude_from_weight_decay = exclude_from_weight_decay

  def apply_grad(self, grad, var):
    """See base class."""
    if grad is None:
      tf.logging.warning("Gradient is None for variable %s" % var.name)
      return []
    grad = mtf.to_float(grad)

    assignments = []

    m = mtf.get_variable(
        var.mesh, var.name + "/adam_m", var.shape,
        initializer=tf.zeros_initializer(), trainable=False)

    v = mtf.get_variable(
        var.mesh, var.name + "/adam_v", var.shape,
        initializer=tf.zeros_initializer(), trainable=False)

    # Standard Adam update.
    next_m = self.beta_1 * m + (1.0 - self.beta_1) * grad
    next_v = self.beta_2 * v + (1.0 - self.beta_2) * mtf.square(grad)

    update = next_m / (mtf.sqrt(next_v) + self.epsilon)

    # Just adding the square of the weights to the loss function is *not*
    # the correct way of using L2 regularization/weight decay with Adam,
    # since that will interact with the m and v parameters in strange ways.
    #
    # Instead we want ot decay the weights in a manner that doesn't interact
    # with the m/v parameters. This is equivalent to adding the square
    # of the weights to the loss with plain (non-momentum) SGD.
    if self._do_use_weight_decay(var.name):
      update += self.weight_decay_rate * var.value

    update_with_lr = self.learning_rate * update

    var_update = mtf.assign_sub(var, update_with_lr)

    assignments.extend(
        [var_update,
         mtf.assign(m, next_m),
         mtf.assign(v, next_v)])
    return assignments

  def _do_use_weight_decay(self, param_name):
    """Whether to use L2 weight decay for `param_name`."""
    if not self.weight_decay_rate:
      return False
    if self.exclude_from_weight_decay:
      for r in self.exclude_from_weight_decay:
        if re.search(r, param_name) is not None:
          return False
    return True


@gin.configurable
class AdafactorOptimizer(Optimizer):
  """Adafactor."""

  def __init__(self,
               multiply_by_parameter_scale=True,
               learning_rate=None,
               decay_rate=None,
               beta1=0.0,
               clipping_threshold=1.0,
               factored=True,
               epsilon1=1e-30,
               epsilon2=1e-3,
               min_dim_size_to_factor=128):
    """Construct a new Adafactor optimizer.

    See class comment.

    Args:
      multiply_by_parameter_scale: a boolean
      learning_rate: an optional Scalar.
      decay_rate: an optional Scalar.
      beta1: a float value between 0 and 1
      clipping_threshold: an optional float >= 1
      factored: a boolean - whether to use factored second-moment estimator
        for 2d variables
      epsilon1: Regularization constant for squared gradient.
      epsilon2: Regularization constant for parameter scale.
      min_dim_size_to_factor: only factor accumulator if two tensor dimensions
        are at least this size.

    Raises:
      ValueError: if absolute_update_scale and relative_update_scale_fn are both
        present or both absent.
    """
    self._multiply_by_parameter_scale = multiply_by_parameter_scale
    if learning_rate is None:
      learning_rate = self._learning_rate_default(multiply_by_parameter_scale)
    self._learning_rate = learning_rate
    if decay_rate is None:
      decay_rate = self._decay_rate_default()
    self._decay_rate = decay_rate
    self._beta1 = beta1
    self._clipping_threshold = clipping_threshold
    self._factored = factored
    self._epsilon1 = epsilon1
    self._epsilon2 = epsilon2
    self._min_dim_size_to_factor = min_dim_size_to_factor

  def _factored_dims(self, shape):
    """Should we use a factored second moment estimator.

    Based on the shape of the variable.
    If we factor the accumulator, then this function returns a list of two
    mtf.Dimensions to reduce over.  We always pick the two largest dimensions.
    If there are not two dimensions of size >= min_dim_size_to_factor, then we
    do not factor.

    Args:
      shape: a Shape
    Returns:
      either a list of 2 Dimensions or None
    """
    if not self._factored or shape.ndims < 2:
      return None
    sorted_dims = sorted(shape.dims, key=lambda d: -d.size)
    if sorted_dims[1].size < self._min_dim_size_to_factor:
      return None
    return sorted_dims[:2]

  def _parameter_scale(self, var):
    """Estimate the scale of the parameters from the current values.

    We include a minimum value of 0.001 to give it a chance to escape 0
    if it was zero-initialized.

    Instead of using the value, we could impute the scale from the shape,
    as initializers do.

    Args:
      var: a variable or Tensor.
    Returns:
      a Scalar
    """
    return mtf.maximum(reduce_rms(var), self._epsilon2)

  def apply_grad(self, grad, var):
    if grad is None:
      tf.logging.warning("Gradient is None for variable %s" % var.name)
      return []
    # create slots
    grad = mtf.to_float(grad)
    factored_dims = self._factored_dims(var.shape)
    if factored_dims:
      d0, d1 = factored_dims
      vr_shape = var.shape - d0
      vc_shape = var.shape - d1
      vr = mtf.get_variable(
          var.mesh, var.name + "_slot_vr", vr_shape,
          initializer=tf.zeros_initializer(), trainable=False)
      vc = mtf.get_variable(
          var.mesh, var.name + "_slot_vc", vc_shape,
          initializer=tf.zeros_initializer(), trainable=False)
    else:
      v = mtf.get_variable(
          var.mesh, var.name + "_slot_v", var.shape,
          initializer=tf.zeros_initializer(), trainable=False)
    if self._beta1:
      m = mtf.get_variable(
          var.mesh, var.name + "_slot_m", var.shape,
          initializer=tf.zeros_initializer(), trainable=False)

    with tf.variable_scope(var.name + "/adafactor"):
      grad_squared = mtf.square(grad) + self._epsilon1
      decay_rate = self._decay_rate
      old_val = mtf.to_float(var.value)
      if self._multiply_by_parameter_scale:
        update_scale = self._parameter_scale(old_val) * self._learning_rate
      else:
        update_scale = self._learning_rate
      mixing_rate = 1.0 - decay_rate
      updates = []
      if factored_dims:
        grad_squared_row_mean = mtf.reduce_mean(
            grad_squared, output_shape=vr_shape)
        grad_squared_col_mean = mtf.reduce_mean(
            grad_squared, output_shape=vc_shape)
        new_vr = vr * decay_rate + grad_squared_row_mean * mixing_rate
        new_vc = vc * decay_rate + grad_squared_col_mean * mixing_rate
        vr_update = mtf.assign(vr, new_vr)
        vc_update = mtf.assign(vc, new_vc)
        updates.extend([vr_update, vc_update])
        long_term_mean = mtf.reduce_mean(new_vr, reduced_dim=d1)
        r_factor = mtf.rsqrt(new_vr / long_term_mean)
        c_factor = mtf.rsqrt(new_vc)
        x = grad * r_factor * c_factor
      else:
        new_v = v * decay_rate + grad_squared * mixing_rate
        v_update = mtf.assign(v, new_v)
        updates.append(v_update)
        x = grad * mtf.rsqrt(new_v)
      if self._clipping_threshold is not None:
        clipping_denom = mtf.maximum(
            1.0, reduce_rms(x) / self._clipping_threshold)
        x /= clipping_denom
      subtrahend = x * update_scale
      if self._beta1:
        new_m = (m * tf.constant(self._beta1)
                 + subtrahend * tf.constant(1.0 - self._beta1))
        subtrahend = new_m
        updates.append(mtf.assign(m, new_m))
      # It is critical to use assign_sub instead of mtf.assign(var - subtrahend)
      #  for the case of bfloat16 activations, so as to avoid repeatedly
      #  rounding the slice value, which results in poor quality.
      var_update = mtf.assign_sub(var, subtrahend)
      updates.append(var_update)
      return updates

  def _decay_rate_default(self):
    return adafactor_decay_rate_pow(0.8)

  def _learning_rate_default(self, multiply_by_parameter_scale):
    learning_rate = tf.minimum(tf.math.rsqrt(step_num() + 1.0), 0.01)
    if (not multiply_by_parameter_scale
        and not layers.unit_scaling_convention()):
      learning_rate *= 0.05
    return learning_rate


def adafactor_decay_rate_adam(beta2):
  """Second-moment decay rate like Adam, subsuming the correction factor.

  Args:
    beta2: a float between 0 and 1
  Returns:
    a scalar
  """
  t = tf.cast(tf.train.get_or_create_global_step(), tf.float32) + 1.0
  decay = beta2 * (1.0 - tf.pow(beta2, t - 1.0)) / (1.0 - tf.pow(beta2, t))
  return decay


def adafactor_decay_rate_pow(exponent):
  """Second moment decay rate where memory-length grows as step_num^exponent.

  Args:
    exponent: a float between 0 and 1
  Returns:
    a scalar
  """
  return 1.0 - tf.pow((step_num() + 1.0), -exponent)


def step_num():
  return tf.cast(tf.train.get_or_create_global_step(), tf.float32)


def adafactor_optimizer_from_hparams(hparams, lr):
  """Create an Adafactor optimizer based on model hparams.

  Args:
    hparams: model hyperparameters
    lr: learning rate scalar.
  Returns:
    an AdafactorOptimizer
  Raises:
    ValueError: on illegal values
  """
  if hparams.optimizer_adafactor_decay_type == "Adam":
    decay_rate = adafactor_decay_rate_adam(
        hparams.optimizer_adafactor_beta2)
  elif hparams.optimizer_adafactor_decay_type == "pow":
    decay_rate = adafactor_decay_rate_pow(
        hparams.optimizer_adafactor_memory_exponent)
  else:
    raise ValueError("unknown optimizer_adafactor_decay_type")
  return AdafactorOptimizer(
      multiply_by_parameter_scale=(
          hparams.optimizer_adafactor_multiply_by_parameter_scale),
      learning_rate=lr,
      decay_rate=decay_rate,
      beta1=hparams.optimizer_adafactor_beta1,
      clipping_threshold=hparams.optimizer_adafactor_clipping_threshold,
      factored=hparams.optimizer_adafactor_factored)


def reduce_rms(x):
  return mtf.sqrt(mtf.reduce_mean(mtf.square(x)))
