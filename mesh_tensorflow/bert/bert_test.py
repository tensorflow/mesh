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
"""Tests for mtf.bert."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mesh_tensorflow as mtf
import mesh_tensorflow.bert.bert as bert_lib

import tensorflow.compat.v1 as tf
# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.tpu import device_assignment as tpu_device_assignment


class BertTest(tf.test.TestCase):

  def test_bert_forward(self):

    def create_computation_fn(device_assignment):
      d_model = 128
      num_blocks = 2
      seq_length = 128
      batch_size = 2
      vocab_size = 30522

      bert_config = bert_lib.BertConfig(
          vocab_size=vocab_size,
          d_model=int(d_model),
          num_blocks=int(num_blocks),
          attention_num_heads=int(d_model / 64),
          feedforward_intermediate_size=int(d_model * 4),
          feedforward_intermediate_act='relu',
          feedforward_intermediate_dropout_prob=0.1,
          attention_probs_dropout_prob=0.1,
          max_position_embeddings=seq_length,
          type_vocab_size=2,
          initializer_range=0.02)

      def computation_fn():
        graph = mtf.Graph()
        mesh = mtf.Mesh(graph, 'my_mesh')
        mesh_shape = mtf.convert_to_shape('all:2')
        layout = 'num_heads:all'
        mesh_devices = [''] * mesh_shape.size
        mesh_impl = mtf.simd_mesh_impl.SimdMeshImpl(
            mesh_shape, mtf.convert_to_layout_rules(layout), mesh_devices,
            device_assignment)
        batch_dim = mtf.Dimension('batch', batch_size)
        seq_dim = mtf.Dimension('seq', seq_length)

        input_ids = tf.random.uniform((batch_size, seq_length),
                                      minval=0,
                                      maxval=vocab_size,
                                      dtype=tf.int32)
        mtf_input_ids = mtf.import_tf_tensor(mesh, input_ids,
                                             [batch_dim, seq_dim])

        model = bert_lib.BertModel(
            config=bert_config,
            is_training=True,
            input_ids=mtf_input_ids,
            input_mask=None,
            token_type_ids=None)
        pooled = model.get_pooled_output()
        lowering = mtf.Lowering(graph, {mesh: mesh_impl})
        return lowering.export_to_tf_tensor(pooled)

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
      sess.run(tf.variables_initializer(tf.get_collection('TPU_VAR')))

      print('TPU', sess.run(tpu_computation_fn))
      sess.run(tf.tpu.shutdown_system())


if __name__ == '__main__':
  tf.disable_v2_behavior()
  tf.test.main()
