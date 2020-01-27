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

"""Tests for mesh_tensorflow.auto_mtf.graph_interface."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mesh_tensorflow as mtf
from mesh_tensorflow.auto_mtf import graph_interface
import tensorflow.compat.v1 as tf
from tensorflow.core.framework import cost_graph_pb2
from tensorflow.core.framework import tensor_shape_pb2
from tensorflow.core.framework import types_pb2


class GraphInterfaceTest(tf.test.TestCase):

  def setUp(self):
    super(GraphInterfaceTest, self).setUp()
    self._cost_graph = cost_graph_pb2.CostGraphDef(
        node=[
            cost_graph_pb2.CostGraphDef.Node(
                name="X",
                device="/device:CPU:0",
                id=0,
                output_info=[
                    cost_graph_pb2.CostGraphDef.Node.OutputInfo(
                        size=48,
                        alias_input_port=-1,
                        dtype=types_pb2.DT_INT32,
                        shape=tensor_shape_pb2.TensorShapeProto(
                            dim=[
                                tensor_shape_pb2.TensorShapeProto.Dim(size=3),
                                tensor_shape_pb2.TensorShapeProto.Dim(size=4),
                            ]
                        )
                    ),
                ],
            ),
            cost_graph_pb2.CostGraphDef.Node(
                name="Y",
                device="/device:CPU:0",
                id=1,
                output_info=[
                    cost_graph_pb2.CostGraphDef.Node.OutputInfo(
                        size=80,
                        alias_input_port=-1,
                        dtype=types_pb2.DT_INT32,
                        shape=tensor_shape_pb2.TensorShapeProto(
                            dim=[
                                tensor_shape_pb2.TensorShapeProto.Dim(size=4),
                                tensor_shape_pb2.TensorShapeProto.Dim(size=5),
                            ]
                        )
                    ),
                ],
            ),
            cost_graph_pb2.CostGraphDef.Node(
                name="Z1",
                device="/device:CPU:0",
                id=2,
                input_info=[
                    cost_graph_pb2.CostGraphDef.Node.InputInfo(
                        preceding_node=0,
                        preceding_port=0,
                    ),
                    cost_graph_pb2.CostGraphDef.Node.InputInfo(
                        preceding_node=1,
                        preceding_port=0,
                    ),
                ],
                output_info=[
                    cost_graph_pb2.CostGraphDef.Node.OutputInfo(
                        size=60,
                        alias_input_port=-1,
                        dtype=types_pb2.DT_INT32,
                        shape=tensor_shape_pb2.TensorShapeProto(
                            dim=[
                                tensor_shape_pb2.TensorShapeProto.Dim(size=3),
                                tensor_shape_pb2.TensorShapeProto.Dim(size=5),
                            ]
                        )
                    ),
                ],
                is_final=True,
            ),
            cost_graph_pb2.CostGraphDef.Node(
                name="Z2",
                device="/device:CPU:0",
                id=3,
                input_info=[
                    cost_graph_pb2.CostGraphDef.Node.InputInfo(
                        preceding_node=0,
                        preceding_port=0,
                    ),
                    cost_graph_pb2.CostGraphDef.Node.InputInfo(
                        preceding_node=1,
                        preceding_port=0,
                    ),
                ],
                output_info=[
                    cost_graph_pb2.CostGraphDef.Node.OutputInfo(
                        size=60,
                        alias_input_port=-1,
                        dtype=types_pb2.DT_INT32,
                        shape=tensor_shape_pb2.TensorShapeProto(
                            dim=[
                                tensor_shape_pb2.TensorShapeProto.Dim(size=3),
                                tensor_shape_pb2.TensorShapeProto.Dim(size=5),
                            ]
                        )
                    ),
                ],
            ),
        ]
    )
    self._sizeless_cost_graph = self.StripCostGraphDef(
        self._cost_graph, "SIZES")
    self._deviceless_cost_graph = self.StripCostGraphDef(
        self._cost_graph, "DEVICES")

    self._cost_graph_string = self._cost_graph.SerializeToString()
    self._sizeless_cost_graph_string = (
        self._sizeless_cost_graph.SerializeToString())
    self._deviceless_cost_graph_string = (
        self._deviceless_cost_graph.SerializeToString())

  def StripCostGraphDef(self, cost_graph, to_strip):
    """Removes fields from a CostGraphDef protobuf.

    Helper method to reduce the initialization of CostGraphDef(s).

    Args:
      cost_graph: a CostGraphDef to strip.
      to_strip: a string, either "SIZES" or "DEVICES".

    Returns:
      a new CostGraphDef with either size information or device information
          stripped, as appropriate.
    """
    new_cost_graph = cost_graph_pb2.CostGraphDef()
    new_cost_graph.CopyFrom(cost_graph)
    for node in new_cost_graph.node:
      if to_strip == "SIZES":
        for output_info in node.output_info:
          output_info.size = 0
          output_info.ClearField("shape")
      if to_strip == "DEVICES":
        node.ClearField("device")
    return new_cost_graph

  def VerifyGraphInterface(self, graph):
    self.assertEqual(list(graph.get_all_operation_names()),
                     ["X", "Y", "Z1", "Z2"])

    self.assertEqual(list(graph.get_operation_input_names("X")), [])
    self.assertEqual(list(graph.get_operation_input_names("Y")), [])
    self.assertEqual(list(graph.get_operation_input_names("Z1")),
                     ["X:0", "Y:0"])
    self.assertEqual(list(graph.get_operation_input_names("Z2")),
                     ["X:0", "Y:0"])

    self.assertEqual(list(graph.get_operation_output_names("X")), ["X:0"])
    self.assertEqual(list(graph.get_operation_output_names("Y")), ["Y:0"])
    self.assertEqual(list(graph.get_operation_output_names("Z1")), ["Z1:0"])
    self.assertEqual(list(graph.get_operation_output_names("Z2")), ["Z2:0"])

    self.assertEqual(list(graph.get_all_tensor_names()),
                     ["X:0", "Y:0", "Z1:0", "Z2:0"])

    self.assertEqual(graph.get_tensor_dtype("X:0"), tf.int32)
    self.assertEqual(graph.get_tensor_dtype("Y:0"), tf.int32)
    self.assertEqual(graph.get_tensor_dtype("Z1:0"), tf.int32)
    self.assertEqual(graph.get_tensor_dtype("Z2:0"), tf.int32)

    self.assertEqual(graph.get_tensor_shape("X:0"), tf.TensorShape([3, 4]))
    self.assertEqual(graph.get_tensor_shape("Y:0"), tf.TensorShape([4, 5]))
    self.assertEqual(graph.get_tensor_shape("Z1:0"), tf.TensorShape([3, 5]))
    self.assertEqual(graph.get_tensor_shape("Z2:0"), tf.TensorShape([3, 5]))

    self.assertEqual(graph.get_tensor_num_entries("X:0"), 12)
    self.assertEqual(graph.get_tensor_num_entries("Y:0"), 20)
    self.assertEqual(graph.get_tensor_num_entries("Z1:0"), 15)
    self.assertEqual(graph.get_tensor_num_entries("Z2:0"), 15)

    graph.set_tensor_final("Z1:0")

    self.assertEqual(graph.compute_memory_contents_under_schedule([0, 1, 2, 3]),
                     [frozenset(["X:0"]), frozenset(["X:0", "Y:0"]),
                      frozenset(["X:0", "Y:0", "Z1:0"]),
                      frozenset(["X:0", "Y:0", "Z1:0", "Z2:0"])])
    self.assertEqual(graph.compute_memory_contents_under_schedule([0, 1, 3, 2]),
                     [frozenset(["X:0"]), frozenset(["X:0", "Y:0"]),
                      frozenset(["X:0", "Y:0", "Z2:0"]),
                      frozenset(["X:0", "Y:0", "Z1:0"])])

  def testTensorFlowGraph(self):
    tf_graph = tf.Graph()
    with tf_graph.as_default():
      with tf.device("/device:CPU:0"):
        x = tf.zeros([3, 4], dtype=tf.int32, name="X")
        y = tf.zeros([4, 5], dtype=tf.int32, name="Y")
        tf.matmul(x, y, name="Z1")
        tf.matmul(x, y, name="Z2")

    graph = graph_interface.GraphInterface(tf_graph,
                                           canonical_device="/device:CPU:0")
    self.VerifyGraphInterface(graph)

    self.assertCountEqual(graph.get_operation_mtf_dimension_names("X"), [])
    self.assertCountEqual(graph.get_operation_mtf_dimension_names("Y"), [])
    self.assertCountEqual(graph.get_operation_mtf_dimension_names("Z1"), [])
    self.assertCountEqual(graph.get_operation_mtf_dimension_names("Z2"), [])

    self.assertCountEqual(graph.get_tensor_mtf_dimension_names("X:0"), [])
    self.assertCountEqual(graph.get_tensor_mtf_dimension_names("Y:0"), [])
    self.assertCountEqual(graph.get_tensor_mtf_dimension_names("Z1:0"), [])
    self.assertCountEqual(graph.get_tensor_mtf_dimension_names("Z2:0"), [])

    self.assertEqual(graph.get_tensor_device("X:0"), "/device:CPU:0")
    self.assertEqual(graph.get_tensor_device("Y:0"), "/device:CPU:0")
    self.assertEqual(graph.get_tensor_device("Z1:0"), "/device:CPU:0")
    self.assertEqual(graph.get_tensor_device("Z2:0"), "/device:CPU:0")

    self.assertTrue(graph.is_tensor_on_canonical_device("X:0"))
    self.assertTrue(graph.is_tensor_on_canonical_device("Y:0"))
    self.assertTrue(graph.is_tensor_on_canonical_device("Z1:0"))
    self.assertTrue(graph.is_tensor_on_canonical_device("Z2:0"))

    self.assertEqual(graph.compute_cost_graph().SerializeToString(),
                     self._cost_graph_string)
    self.assertEqual(graph.compute_cost_graph(devices=["/device:CPU:0"])
                     .SerializeToString(),
                     self._cost_graph_string)
    self.assertEqual(graph.compute_cost_graph(devices=[]).SerializeToString(),
                     self._sizeless_cost_graph_string)

  def testMeshTensorFlowGraph(self):
    mtf_graph = mtf.Graph()
    mesh = mtf.Mesh(mtf_graph, "my_mesh")
    x = mtf.Constant(mesh, 0,
                     shape=mtf.convert_to_shape("a:3,b:4"),
                     dtype=tf.int32,
                     name="X").outputs[0]
    y = mtf.Constant(mesh, 0,
                     shape=mtf.convert_to_shape("b:4,c:5"),
                     dtype=tf.int32,
                     name="Y").outputs[0]
    mtf.EinsumOperation([x, y], mtf.convert_to_shape("a:3,c:5"), name="Z1")
    mtf.EinsumOperation([x, y], mtf.convert_to_shape("a:3,c:5"), name="Z2")

    graph = graph_interface.GraphInterface(mtf_graph)
    self.VerifyGraphInterface(graph)

    self.assertCountEqual(graph.get_operation_mtf_dimension_names("X"),
                          ["a", "b"])
    self.assertCountEqual(graph.get_operation_mtf_dimension_names("Y"),
                          ["b", "c"])
    self.assertCountEqual(graph.get_operation_mtf_dimension_names("Z1"),
                          ["a", "b", "c"])
    self.assertCountEqual(graph.get_operation_mtf_dimension_names("Z2"),
                          ["a", "b", "c"])

    self.assertCountEqual(graph.get_tensor_mtf_dimension_names("X:0"),
                          ["a", "b"])
    self.assertCountEqual(graph.get_tensor_mtf_dimension_names("Y:0"),
                          ["b", "c"])
    self.assertCountEqual(graph.get_tensor_mtf_dimension_names("Z1:0"),
                          ["a", "c"])
    self.assertCountEqual(graph.get_tensor_mtf_dimension_names("Z1:0"),
                          ["a", "c"])

    self.assertIsNone(graph.get_tensor_device("X:0"))
    self.assertIsNone(graph.get_tensor_device("Y:0"))
    self.assertIsNone(graph.get_tensor_device("Z1:0"))
    self.assertIsNone(graph.get_tensor_device("Z2:0"))

    self.assertTrue(graph.is_tensor_on_canonical_device("X:0"))
    self.assertTrue(graph.is_tensor_on_canonical_device("Y:0"))
    self.assertTrue(graph.is_tensor_on_canonical_device("Z1:0"))
    self.assertTrue(graph.is_tensor_on_canonical_device("Z2:0"))

    self.assertEqual(graph.compute_cost_graph().SerializeToString(),
                     self._deviceless_cost_graph_string)
    self.assertEqual(graph.compute_cost_graph(devices=[]).SerializeToString(),
                     self._deviceless_cost_graph_string)

  def testNotAGraph(self):
    self.assertRaises(TypeError, graph_interface.GraphInterface, "hello")


if __name__ == "__main__":
  tf.disable_v2_behavior()
  tf.test.main()
