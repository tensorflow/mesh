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

"""Common utilities for Mesh TensorFlow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import contextlib
import heapq

import tensorflow.compat.v1 as tf
import tensorflow.compat.v2 as tf2
# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.framework import ops  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.tpu.topology import Topology


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


SCALAR_SUMMARIES_COLLECTION_KEY = 'mtf_scalar_summaries'


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
  # a list of (name, lowered tensor) tuples
  summaries = graph.get_collection(SCALAR_SUMMARIES_COLLECTION_KEY)

  def maybe_cast(tensor):
    assert tensor.shape.is_compatible_with([]), tensor.name
    if tensor.dtype == tf.int64:
      return tf.to_int32(tensor)
    if tensor.dtype == tf.bfloat16:
      return tf.cast(tensor, tf.float32)
    return tensor

  reshaped_tensors = [tf.reshape(maybe_cast(t), [1]) for _, t in summaries]

  # When no supported summaries are found, don't create host_call. Otherwise,
  # TPU outfeed queue would enqueue global_step while host_call doesn't dequeue
  # it, eventually causing hang.
  if not reshaped_tensors:
    return None

  def host_call_fn(global_step, *args):
    """Training host call. Creates scalar summaries for training metrics."""
    # This function is executed on the CPU and should not directly reference
    # any Tensors in the rest of the `model_fn`. To pass Tensors from the
    # model to the `model_fn`, provide as part of the `host_call`.
    global_step = tf.cast(global_step[0], tf.int64)
    with tf2.summary.create_file_writer(model_dir).as_default():
      # We cannot directly use any tensor from summaries, because each
      # tensor here must be a concat of multiple tensors from all shards.
      # Therefore, we rely on the assumption that args wil have the same
      # length as summaries, and all tensors in args will have the same
      # order of self._tup_summaries.
      assert len(args) == len(summaries)
      for i, tensor in enumerate(args):
        name = summaries[i][0]
        tf.summary.scalar(
            name, tf.reduce_mean(tensor), step=global_step)
      return tf.summary.all_v2_summary_ops()

  global_step_t = tf.reshape(tf.to_int32(tf.train.get_global_step()), [1])
  return host_call_fn, [global_step_t] + reshaped_tensors


def topology_rank(topology):
  # Deserialize the Topology proto, if it is a string.
  if isinstance(topology, bytes):
    topology = Topology(serialized=topology)

  if not isinstance(topology, Topology):
    raise ValueError('`topology` is not a Topology object; got {}'.format(
        type(topology)))

  return len(topology.mesh_shape)


def remove_summaries():
  """Remove summaries from the default graph."""
  g = tf.get_default_graph()
  key = 'mtf_scalar_summaries'
  tf.logging.debug('Remove summaries %s' % str(g.get_collection(key)))
  del g.get_collection_ref(key)[:]
  assert not g.get_collection(key)
