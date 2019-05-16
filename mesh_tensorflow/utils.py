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

"""Common utilities for Mesh TensorFlow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import contextlib
import heapq
import six

import tensorflow as tf
from tensorflow.python.framework import ops


@contextlib.contextmanager
def outside_all_rewrites():
  with ops.control_dependencies(None):
    yield


class BalancedVariablePlacer(object):
  """Place the variable on different device and balance the memory usage."""

  def __init__(self, devices, init_usage=None):
    init_usage = init_usage if init_usage else [0] * len(devices)
    assert len(devices) == len(init_usage)
    self._mem_device_heap = list(zip(init_usage, devices))
    heapq.heapify(self._mem_device_heap)
    self._last_device = devices[0]

  def device_function(self, var):
    """Choose a device for the input variable.

    Args:
      var: an Variable.

    Returns:
      The device for placing the var.
    """
    if var.type not in ('Variable', 'VariableV2', 'VarHandleOp'):
      tf.logging.debug('Place {} on last device: {}.'.format(
          var.name, self._last_device))
      return self._last_device

    shape = tf.TensorShape(var.get_attr('shape'))
    assert shape.num_elements() is not None

    size = var.get_attr('dtype').size
    mem, device = heapq.heappop(self._mem_device_heap)
    mem += shape.num_elements() * size
    heapq.heappush(self._mem_device_heap, (mem, device))
    tf.logging.debug('Place variable {} on {} and consumes {} Bytes.'.format(
        var.name, device, mem))
    self._last_device = device

    return device


def create_host_call(model_dir):
  """Construct a host_call writing scalar summaries.

  Borrowed from t2t.
  TODO(noam): remove this code once there is a better way to get summaries on
  TPU.

  Args:
    model_dir: String containing path to train

  Returns:
    (fn, args) Pair to be called by TPUEstimator as the host_call.
  """
  graph = tf.get_default_graph()
  summaries = graph.get_collection(tf.GraphKeys.SUMMARIES)
  gs_t = tf.reshape(tf.to_int32(tf.train.get_global_step()), [1])
  summary_kwargs = collections.OrderedDict()
  for t in summaries:
    # TODO(aidangomez): enable ImageSummary support when we have a faster method
    # see @shibow's comment in cl/202344570
    if t.op.type not in ['ScalarSummary']:
      tf.logging.warn('Ignoring unsupported tf.Summary type %s' % t.op.type)
      continue

    name = t.op.name
    tensor = t.op.inputs[1]
    if t.op.type == 'ScalarSummary':
      assert tensor.shape.is_compatible_with([])
      if tensor.dtype == tf.int64:
        tensor = tf.to_int32(tensor)
      summary_kwargs['ScalarSummary' + name] = tf.reshape(tensor, [1])
  # When no supported summaries are found, don't create host_call. Otherwise,
  # TPU outfeed queue would enqueue global_step while host_call doesn't dequeue
  # it, eventually causing hang.
  if not summary_kwargs:
    return None
  summary_kwargs['global_step'] = gs_t
  tf.logging.info('summary_kwargs %s' % str(summary_kwargs))

  def host_call_fn(**kwargs):
    """Training host call. Creates summaries for training metrics.

    Args:
      **kwargs: Dict of {str: Tensor} , with `Tensor` of shape `[batch]`. Must
        contain key 'global_step' with value of current global_step Tensor.

    Returns:
      List of summary ops to run on the CPU host.
    """
    gs = tf.to_int64(kwargs.pop('global_step')[0])
    with tf.contrib.summary.create_file_writer(model_dir).as_default():
      with tf.contrib.summary.always_record_summaries():
        # We need to use tf.contrib.summary in order to feed the `step`.
        for name, value in sorted(six.iteritems(kwargs)):
          if name.startswith('ScalarSummary'):
            name = name[len('ScalarSummary'):]
            tf.contrib.summary.scalar(
                name, tf.reduce_mean(tf.to_float(value)), step=gs)
        return tf.contrib.summary.all_summary_ops()

  return (host_call_fn, summary_kwargs)


def remove_summaries():
  """Remove summaries from the default graph."""
  g = tf.get_default_graph()
  key = tf.GraphKeys.SUMMARIES
  tf.logging.debug('Remove summaries %s' % str(g.get_collection(key)))
  del g.get_collection_ref(key)[:]
  assert not g.get_collection(key)
