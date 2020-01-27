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

"""Tests for mesh_tensorflow.auto_mtf.valid_layouts."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mesh_tensorflow as mtf
from mesh_tensorflow.auto_mtf import valid_layouts
import tensorflow.compat.v1 as tf


class LayoutValidatorTest(tf.test.TestCase):

  def setUp(self):
    super(LayoutValidatorTest, self).setUp()
    graph = mtf.Graph()
    mesh = mtf.Mesh(graph, "my_mesh")

    a_dim = mtf.Dimension("a", 5)
    b_dim = mtf.Dimension("b", 10)
    concat_dim1 = mtf.Dimension("concat", 15)
    concat_dim2 = mtf.Dimension("concat", 20)

    x1 = mtf.zeros(mesh, mtf.Shape([a_dim, b_dim, concat_dim1]))
    x2 = mtf.zeros(mesh, mtf.Shape([a_dim, b_dim, concat_dim2]))
    mtf.ConcatOperation([x1, x2], "concat")

    # We add a tensor with anonymous shape, which is supposed to be
    # unsplittable (i.e. none of its dimensions show up during
    # test_SplittableMtfDimensionNames).
    _ = mtf.zeros(mesh, mtf.anonymous_shape(mtf.Shape([a_dim, b_dim])))

    mesh_shape = mtf.Shape([("m1", 4), ("m2", 2)])
    self.valid_layouts = valid_layouts.LayoutValidator(graph, mesh_shape)

  def test_SplittableMtfDimensionNames(self):
    self.assertEqual(self.valid_layouts.splittable_mtf_dimension_names,
                     set(["a", "b"]))

  def test_MeshDimensionNameToSize(self):
    self.assertEqual(self.valid_layouts.mesh_dimension_name_to_size,
                     {"m1": 4, "m2": 2})

  def test_is_valid_assignment(self):
    # Due to divisibility, the a dimension cannot be assigned to m1 or m2.
    self.assertFalse(self.valid_layouts.is_valid_assignment("a", "m1"))
    self.assertFalse(self.valid_layouts.is_valid_assignment("a", "m2"))
    # The b dimension can only be assigned to m2.
    self.assertFalse(self.valid_layouts.is_valid_assignment("b", "m1"))
    self.assertTrue(self.valid_layouts.is_valid_assignment("b", "m2"))
    # Due to ConcatOperation, the concat dimension may not be assigned.
    self.assertFalse(self.valid_layouts.is_valid_assignment("concat", "m1"))
    self.assertFalse(self.valid_layouts.is_valid_assignment("concat", "m2"))


if __name__ == "__main__":
  tf.disable_v2_behavior()
  tf.test.main()
