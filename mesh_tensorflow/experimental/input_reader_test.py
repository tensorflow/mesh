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

"""Tests for third_party.py.mesh_tensorflow.experimental.input_reader."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import mesh_tensorflow as mtf
import mesh_tensorflow.experimental.input_reader as input_reader
import numpy as np
import tensorflow.compat.v1 as tf

# pylint: disable=g-direct-tensorflow-import
from tensorflow.core.protobuf.tpu import topology_pb2
from tensorflow.python.tpu import device_assignment
from tensorflow.python.tpu import tpu


class MtfInputReaderTest(parameterized.TestCase, tf.test.TestCase):

  def initialize_system(self, sess):
    """Run tpu.initialize_system and return the number of TPU devices."""
    topology_object = topology_pb2.TopologyProto()
    topology = sess.run(tf.tpu.initialize_system())
    topology_object.ParseFromString(topology)
    num_cores = topology_object.num_tasks * (
        topology_object.num_tpu_devices_per_task)
    return topology, num_cores

  @parameterized.parameters((True,), (False,))
  def test_get_laidout_tensors(self, is_eval_mode):
    mesh_shape = "mesh_x:2, mesh_y:1"
    layout = "batch:mesh_x, io:mesh_y"
    batch_io_dim = 4

    with tf.Session() as sess:
      topology, num_cores = self.initialize_system(sess)

      # Get a device_assignment object for mtf.
      d_assignment = device_assignment.device_assignment(
          topology,
          computation_shape=[1,] * mtf.utils.topology_rank(topology),
          num_replicas=num_cores)

      # Hacked dataset creator: creates different datasets for the first and
      # second call, in order to test SimdMeshImplInputReader.
      self.sub_batch_created_times = 0
      def stateful_ds_creator():
        whole_batch = tf.eye(batch_io_dim, dtype=tf.float32)
        sub_batch = tf.slice(whole_batch,
                             [self.sub_batch_created_times * 2, 0],
                             [2, 4])
        self.sub_batch_created_times += 1
        return tf.data.Dataset.from_tensors(sub_batch).repeat().unbatch()

      batch_dim = mtf.Dimension("batch", batch_io_dim)
      io_dim = mtf.Dimension("io", batch_io_dim)
      mtf_input_shapes = [mtf.Shape([batch_dim, io_dim])]

      # Get mesh_impl.
      mesh_shape = mtf.convert_to_shape(mesh_shape)
      layout_rules = mtf.convert_to_layout_rules(layout)
      mesh_impl = mtf.simd_mesh_impl.SimdMeshImpl(
          mesh_shape, layout_rules, None, d_assignment)

      simd_input_reader = input_reader.SimdMeshImplInputReader(
          mesh_impl, stateful_ds_creator, mtf_input_shapes,
          external_worker=False,
          is_eval_mode=is_eval_mode)

      def model_fn(features):
        return features

      replicated_computation = tpu.replicate(
          computation=model_fn,
          inputs=[[]] * num_cores,
          infeed_queue=simd_input_reader.infeed_queue,
          device_assignment=d_assignment)

      simd_input_reader.start_infeed_thread(sess, 1)
      results = sess.run(replicated_computation)
      print("results: {}".format(results))

      core_0_data = results[0][0]
      core_1_data = results[1][0]
      print("core_0_data: {}".format(core_0_data))
      print("core_1_data: {}".format(core_1_data))

      if is_eval_mode:
        # If there is only one dataset object, then the stateful_ds_creator()
        # should be called only once.
        self.assertAllClose(
            np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32),
            core_0_data)
        self.assertAllClose(
            np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32),
            core_1_data)
      else:
        # If there are two dataset objects, then the stateful_ds_creator()
        # should be called twice.
        self.assertAllClose(
            np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32),
            core_0_data)
        self.assertAllClose(
            np.array([[0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32),
            core_1_data)

      sess.run(tf.tpu.shutdown_system())


if __name__ == "__main__":
  tf.test.main()
