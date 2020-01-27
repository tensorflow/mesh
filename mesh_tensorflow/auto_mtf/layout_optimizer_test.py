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

"""Tests for mesh_tensorflow.layout_optimizer."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mesh_tensorflow as mtf
from mesh_tensorflow.auto_mtf import layout_optimizer
from mesh_tensorflow.auto_mtf import memory_estimator
import six
import tensorflow.compat.v1 as tf


class VariableNamesTest(tf.test.TestCase):

  def testGlobalVarName(self):
    self.assertEqual("x_(cake:lie)",
                     layout_optimizer._global_var_name("cake", "lie"))

  def testLocalVarName(self):
    self.assertEqual("y_()", layout_optimizer._local_var_name(frozenset(), {}))
    self.assertEqual(
        "y_(channel:y,hidden:x)",
        layout_optimizer._local_var_name(frozenset(["channel", "hidden"]),
                                         {"hidden": "x", "channel": "y"}))
    self.assertEqual(
        "y_(channel:y,hidden)",
        layout_optimizer._local_var_name(frozenset(["channel", "hidden"]),
                                         {"channel": "y"}))


class AssignmentsTest(tf.test.TestCase):

  def testGenerateAssignments(self):
    splittable_dims = {"s1", "s2", "s3"}
    mesh_dims = {"m1": 4, "m2": 8}

    assignments = layout_optimizer._generate_assignments(splittable_dims,
                                                         mesh_dims)
    # Check that some valid assignments of various sizes are included
    self.assertIn({}, assignments)
    self.assertIn({"s3": "m2"}, assignments)
    self.assertIn({"s1": "m2", "s2": "m1"}, assignments)
    # Not allowed to map two splittable dimensions to the same mesh dimension.
    self.assertNotIn({"s1": "m2", "s3": "m2"}, assignments)
    # Check the total number of assignments returned. We are looking for
    # thirteen because one assignment has no entries, six assignments have one
    # entry, and six assignments have two entries.
    self.assertLen(assignments, 13)


class OptimizeLayoutTest(tf.test.TestCase):

  def setUp(self):
    super(OptimizeLayoutTest, self).setUp()
    self.mtf_graph = mtf.Graph()
    self.mesh = mtf.Mesh(self.mtf_graph, "my_mesh")
    self.mesh_shape = mtf.convert_to_shape("m1:4,m2:2")

  def get_layout_optimizer(self):
    return layout_optimizer.LayoutOptimizer(memory_estimator.MemoryEstimator(
        self.mtf_graph, self.mesh_shape))

  def testOptimizeLayout(self):
    x1 = mtf.zeros(self.mesh, "a:10,b:5")
    x2 = mtf.zeros(self.mesh, "b:5,c:20")
    mtf.einsum([x1, x2], "a:10,c:20")
    optimizer = self.get_layout_optimizer()

    # Cut dimensions to make them equally sized.
    layout = optimizer.solve()
    self.assertEqual(layout, "a:m2;c:m1")

    # This optimal layout should have the lowest value.
    layout_value = optimizer.evaluate_layout(layout)
    self.assertLessEqual(layout_value, optimizer.evaluate_layout("a:m1;b:m2"))
    self.assertLessEqual(layout_value, optimizer.evaluate_layout("a:m1;c:m2"))
    self.assertLessEqual(layout_value, optimizer.evaluate_layout("b:m1;a:m2"))
    self.assertLessEqual(layout_value, optimizer.evaluate_layout("b:m1;c:m2"))
    self.assertLessEqual(layout_value, optimizer.evaluate_layout("c:m1;b:m2"))
    self.assertEqual(layout_value, optimizer.evaluate_layout("c:m1;a:m2"))

  def testOptimizeLayoutRepetition(self):
    x1 = mtf.zeros(self.mesh, "a:10,b:5")
    x2 = mtf.zeros(self.mesh, "b:5,c:20")
    for _ in six.moves.xrange(100):
      mtf.einsum([x1, x2], "a:10,c:20")
    optimizer = self.get_layout_optimizer()

    self.assertGreaterEqual(len(list(
        optimizer._graph.get_all_operation_names())), 50)
    self.assertLessEqual(len(optimizer._model.Proto().variables), 50)

    # Same checks.
    layout = optimizer.solve()
    self.assertEqual(layout, "a:m2;c:m1")
    layout_value = optimizer.evaluate_layout(layout)
    self.assertLessEqual(layout_value, optimizer.evaluate_layout("a:m1;b:m2"))
    self.assertLessEqual(layout_value, optimizer.evaluate_layout("a:m1;c:m2"))
    self.assertLessEqual(layout_value, optimizer.evaluate_layout("b:m1;a:m2"))
    self.assertLessEqual(layout_value, optimizer.evaluate_layout("b:m1;c:m2"))
    self.assertLessEqual(layout_value, optimizer.evaluate_layout("c:m1;b:m2"))
    self.assertEqual(layout_value, optimizer.evaluate_layout("c:m1;a:m2"))

  def testOptimizeLayoutUnsplittable(self):
    x1 = mtf.zeros(self.mesh, "a:10,b:5")
    x2 = mtf.zeros(self.mesh, "b:5,c:20")
    mtf.UnstackOperation(x1, mtf.Dimension("a", 10))
    mtf.UnstackOperation(x2, mtf.Dimension("c", 20))
    optimizer = self.get_layout_optimizer()

    # No dimensions can be split, because a and c are unstack dimensions and
    # b has size 5 (so there are divisiblity issues).
    self.assertEqual(optimizer.solve(), "")

  def testOptimizeLayoutTiebreak(self):
    x1 = mtf.zeros(self.mesh, "a:10,b:5")
    x2 = mtf.zeros(self.mesh, "b:5,c:20")
    mtf.einsum([x1, x2], "a:10,c:20")
    # Rewrite mesh_shape to have a dummy dimension.
    self.mesh_shape = mtf.convert_to_shape("m1:4,m2:2,m3:1")
    optimizer = self.get_layout_optimizer()
    layout = optimizer.solve()
    self.assertEqual(layout, "a:m2;b:m3;c:m1")

# TODO(joshuawang): Add test to ensure only a single device"s worth of memory is
# being measured.


if __name__ == "__main__":
  tf.disable_v2_behavior()
  tf.test.main()
