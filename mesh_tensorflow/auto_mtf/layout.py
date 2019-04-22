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

"""Wrapper function to automatically compute a layout.

Sample Usage:
  import mesh_tensorflow as mtf
  import mesh_tensorflow.auto_mtf

  # Construct a Mesh TensorFlow graph and mesh.
  mtf_graph = mtf.Graph()
  mesh = mtf.Mesh(mtf_graph, "my_mesh")
  x = mtf.zeros(self.mesh, "a:10,b:5")
  y = mtf.zeros(self.mesh, "b:5,c:20")
  z = mtf.einsum([x, y], "a:10,c:20")

  # Decide on a mesh shape.
  mesh_shape = mtf.convert_to_shape("m1:4,m2:2")

  # Compute a layout based on the graph and mesh.
  # Note that knowing the identity of the outputs is important to the
  # optimization since they cannot be freed.
  layout = mtf.auto_mtf.layout(mtf_graph, mesh_shape, [z])
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mesh_tensorflow as mtf
from mesh_tensorflow.auto_mtf import layout_optimizer
from mesh_tensorflow.auto_mtf import memory_estimator


def layout(mtf_graph, mesh_shape, mtf_outputs=()):
  """Compute layout rules based on a computational graph and mesh shape.

  Args:
    mtf_graph: a mtf.Graph.
    mesh_shape: an mtf.Shape, str, or listlike of mtf.Dimension.
    mtf_outputs: an optional iterable of mtf.Tensor, representing the outputs
        of the computation.

  Returns:
    a mtf.LayoutRules
  """
  mesh_shape = mtf.convert_to_shape(mesh_shape)
  estimator = memory_estimator.MemoryEstimator(mtf_graph, mesh_shape,
                                               mtf_outputs)
  optimizer = layout_optimizer.LayoutOptimizer(estimator)
  return mtf.convert_to_layout_rules(optimizer.solve())
