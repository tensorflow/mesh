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

# Lint as: python3
"""Tests for MeshTensorFlow BERT optimization."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mesh_tensorflow as mtf
import mesh_tensorflow.bert.optimization as optimization_lib
import mesh_tensorflow.optimize as mtf_optimize

import tensorflow.compat.v1 as tf
# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.tpu import device_assignment as tpu_device_assignment


class OptimizationTest(tf.test.TestCase):

  def test_adam(self):
    self.lowering = None

    def create_computation_fn(device_assignment):

      def computation_fn():
        graph = mtf.Graph()
        mesh = mtf.Mesh(graph, 'my_mesh')
        mesh_shape = mtf.convert_to_shape('all:2')
        layout = 'none:all'
        mesh_devices = [''] * mesh_shape.size
        mesh_impl = mtf.simd_mesh_impl.SimdMeshImpl(
            mesh_shape, mtf.convert_to_layout_rules(layout), mesh_devices,
            device_assignment)
        hidden_dim = mtf.Dimension('hidden', 3)
        w = mtf.get_variable(
            mesh,
            'w',
            shape=[hidden_dim],
            initializer=tf.constant_initializer([0.1, -0.2, -0.1]))
        x = mtf.constant(mesh, [0.4, 0.2, -0.5], [hidden_dim], dtype=tf.float32)
        loss = mtf.reduce_mean(mtf.square(x - w))
        var_grads = mtf.gradients(
            [loss], [v.outputs[0] for v in graph.trainable_variables])
        optimizer = mtf_optimize.AdamWeightDecayOptimizer(learning_rate=0.2)
        update_ops = optimizer.apply_grads(var_grads, graph.trainable_variables)
        self.lowering = mtf.Lowering(graph, {mesh: mesh_impl})
        tf_update_ops = [
            self.lowering.lowered_operation(op) for op in update_ops
        ]
        return tf.group(tf_update_ops)

      return computation_fn

    with self.test_session() as sess:
      topology = sess.run(tf.tpu.initialize_system())
      device_assignment = tpu_device_assignment.device_assignment(
          topology, computation_shape=[1, 1, 1], num_replicas=2)
      tpu_computation_fn = tf.tpu.batch_parallel(
          create_computation_fn(device_assignment),
          inputs=None,
          num_shards=2,
          infeed_queue=None,
          device_assignment=device_assignment)

      sess.run(tf.global_variables_initializer())
      sess.run(self.lowering.copy_masters_to_slices())

      for _ in range(100):
        sess.run(tpu_computation_fn)

      sess.run(self.lowering.copy_slices_to_masters())
      w_np = sess.run(tf.global_variables()[0])
      self.assertAllClose([0.4, 0.2, -0.5], w_np.flat, rtol=1e-2, atol=1e-2)
      sess.run(tf.tpu.shutdown_system())

  def test_optimizer(self):
    self.lowering = None

    def create_computation_fn(device_assignment):

      def computation_fn():
        graph = mtf.Graph()
        mesh = mtf.Mesh(graph, 'my_mesh')
        mesh_shape = mtf.convert_to_shape('all:2')
        layout = 'none:all'
        mesh_devices = [''] * mesh_shape.size
        mesh_impl = mtf.simd_mesh_impl.SimdMeshImpl(
            mesh_shape, mtf.convert_to_layout_rules(layout), mesh_devices,
            device_assignment)
        hidden_dim = mtf.Dimension('hidden', 3)
        w = mtf.get_variable(
            mesh,
            'w',
            shape=[hidden_dim],
            initializer=tf.constant_initializer([0.1, -0.2, -0.1]))
        x = mtf.constant(mesh, [0.4, 0.2, -0.5], [hidden_dim], dtype=tf.float32)
        loss = mtf.reduce_mean(mtf.square(x - w))

        lr, update_ops = optimization_lib.create_optimizer(loss, 0.2, 100, 10)
        self.lowering = mtf.Lowering(graph, {mesh: mesh_impl})

        tf_update_ops = [
            self.lowering.lowered_operation(op) for op in update_ops
        ]
        tf_update_ops.append(
            tf.assign_add(tf.train.get_or_create_global_step(), 1))
        train_op = tf.group(tf_update_ops)

        return lr, train_op

      return computation_fn

    with self.test_session() as sess:
      topology = sess.run(tf.tpu.initialize_system())
      device_assignment = tpu_device_assignment.device_assignment(
          topology, computation_shape=[1, 1, 1], num_replicas=2)
      tpu_computation_fn = tf.tpu.batch_parallel(
          create_computation_fn(device_assignment),
          inputs=None,
          num_shards=2,
          infeed_queue=None,
          device_assignment=device_assignment)

      sess.run(tf.global_variables_initializer())
      sess.run(self.lowering.copy_masters_to_slices())
      lrs = []
      for _ in range(100):
        lr = sess.run(tpu_computation_fn)
        lrs.append(lr[0][0])
      self.assertAllClose(0.02, lrs[0])
      self.assertAllClose(0.18, lrs[8])
      self.assertAllClose(0., lrs[99])
      sess.run(tf.tpu.shutdown_system())


if __name__ == '__main__':
  tf.disable_v2_behavior()
  tf.test.main()
