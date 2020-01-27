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

"""Tests for mesh_tensorflow.auto_mtf.scheduler."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import mesh_tensorflow as mtf
from mesh_tensorflow.auto_mtf import graph_interface
from mesh_tensorflow.auto_mtf import scheduler
import tensorflow.compat.v1 as tf


class SchedulerTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters('NAIVE', 'LIST')
  def testReturnsTopoSort(self, scheduler_alg):
    mtf_graph = mtf.Graph()
    mesh = mtf.Mesh(mtf_graph, 'my_mesh')
    x = mtf.Constant(mesh, 0,
                     shape=mtf.convert_to_shape('a:3,b:4'),
                     dtype=tf.int32,
                     name='X').outputs[0]
    y = mtf.Constant(mesh, 0,
                     shape=mtf.convert_to_shape('b:4,c:5'),
                     dtype=tf.int32,
                     name='Y').outputs[0]
    mtf.EinsumOperation([x, y], mtf.convert_to_shape('a:3,c:5'), name='Z1')
    mtf.EinsumOperation([x, y], mtf.convert_to_shape('a:3,c:5'), name='Z2')

    graph = graph_interface.GraphInterface(mtf_graph)
    graph.set_tensor_final('Z1:0')
    graph.set_tensor_final('Z2:0')
    schedule = list(scheduler.minimize_peak_memory(graph, scheduler_alg))

    self.assertCountEqual(schedule[0:2], [0, 1])
    self.assertCountEqual(schedule[2:4], [2, 3])

  def testMinimizePeakMemoryList(self):
    mtf_graph = mtf.Graph()
    mesh = mtf.Mesh(mtf_graph, 'my_mesh')
    x = mtf.Constant(mesh, 0,
                     shape=mtf.convert_to_shape('a:3,b:4'),
                     dtype=tf.int32,
                     name='X').outputs[0]
    y = mtf.Constant(mesh, 0,
                     shape=mtf.convert_to_shape('b:4,c:5'),
                     dtype=tf.int32,
                     name='Y').outputs[0]
    mtf.EinsumOperation([x, y], mtf.convert_to_shape('a:3,b:4,c:5'), name='Z')
    w = mtf.EinsumOperation([x, y], mtf.convert_to_shape('a:3,c:5'),
                            name='W').outputs[0]
    mtf.BroadcastOperation(w, mtf.convert_to_shape('a:3,b:4,c:5'), name='V')

    graph = graph_interface.GraphInterface(mtf_graph)
    graph.set_tensor_final('Z:0')
    graph.set_tensor_final('V:0')
    schedule = list(scheduler.minimize_peak_memory(graph, 'LIST'))

    # List Scheduler prefers to schedule things that free the most memory.
    # When nothing is scheduled:
    #   X frees -12 entries.
    #   Y frees -20 entries.
    # After [X] scheduled:
    #   Y frees -20 entries.
    # After [X, Y] scheduled:
    #   Z frees -60 entries.
    #   W frees -15 entries.
    # After [X, Y, W] scheduled:
    #   Z frees -28 entries.
    #   V frees -45 entries.
    # Hence the schedule should be [X, Y, W, Z, V].
    self.assertEqual(schedule, [0, 1, 3, 2, 4])

  def testMinimizePeakMemoryList_SingleUseTensor(self):
    mtf_graph = mtf.Graph()
    mesh = mtf.Mesh(mtf_graph, 'my_mesh')
    mtf.Constant(mesh, 0, shape=mtf.convert_to_shape('a:4'), dtype=tf.int32,
                 name='X')
    y = mtf.Constant(mesh, 0, shape=mtf.convert_to_shape('b:3'), dtype=tf.int32,
                     name='Y').outputs[0]
    mtf.BroadcastOperation(y, mtf.convert_to_shape('b:3,c:2'), name='Z')

    graph = graph_interface.GraphInterface(mtf_graph)
    graph.set_tensor_final('X:0')
    graph.set_tensor_final('Z:0')
    schedule = list(scheduler.minimize_peak_memory(graph, 'LIST'))
    # When nothing is scheduled:
    #   X frees -4 entries
    #   Y frees -3 entries
    # After [Y] scheduled:
    #   X frees -4 entries
    #   Z frees -3 entries
    # Hence the schedule should be [Y, Z, X].
    self.assertEqual(schedule, [1, 2, 0])

  def testMinimizePeakMemoryList_ZeroUseTensor(self):
    mtf_graph = mtf.Graph()
    mesh = mtf.Mesh(mtf_graph, 'my_mesh')
    mtf.Constant(mesh, 0, shape=mtf.convert_to_shape('a:4'), dtype=tf.int32,
                 name='X')
    y = mtf.Constant(mesh, 0, shape=mtf.convert_to_shape('b:3'), dtype=tf.int32,
                     name='Y').outputs[0]
    mtf.BroadcastOperation(y, mtf.convert_to_shape('b:3,c:2'), name='Z')

    graph = graph_interface.GraphInterface(mtf_graph)
    schedule = list(scheduler.minimize_peak_memory(graph, 'LIST'))
    # When nothing is scheduled:
    #   X frees 0 entries
    #   Y frees -3 entries
    # Hence the schedule should be [X, Y, Z].
    self.assertEqual(schedule, [0, 1, 2])


if __name__ == '__main__':
  tf.disable_v2_behavior()
  tf.test.main()
