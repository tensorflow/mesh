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

  # Compute a layout and mesh shape based on graph and 8 machines.
  # Note that knowing the identity of the outputs is important to the
  # optimization since they cannot be freed.
  layout, mesh_shape = mtf.auto_mtf.layout_and_mesh_Shape(mtf_graph, 8, [z])
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mesh_tensorflow as mtf
from mesh_tensorflow.auto_mtf import layout_optimizer
from mesh_tensorflow.auto_mtf import memory_estimator
import tensorflow.compat.v1 as tf


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


def layout_and_mesh_shape(mtf_graph, num_machines, mtf_outputs=(),
                          max_mesh_shape_dimensions=2):
  """Compute layout rules and mesh shape based on computational graph.

  Brute-forces over all possible mesh shapes to find a (layout, mesh_shape)
  pair. Note that the layout optimizer is more efficient when the mesh_shape has
  fewer dimensions, so a smaller max_mesh_shape_dimensions makes this call
  faster.

  Args:
    mtf_graph: a mtf.Graph.
    num_machines: integer, a power of two, the number of machines available.
    mtf_outputs: an optional iterable of mtf.Tensor, representing the outputs
        of the computation.
    max_mesh_shape_dimensions: optional integer, the maximum number of
        dimensions to consider in any layout. For example, num_machines=1024 and
        max_mesh_shape_dimensions=2 results in testing the mesh shapes
        "mesh_0:1024", "mesh_0:512;mesh_1:2", "mesh_0:256;mesh_1:4",
        "mesh_0:128;mesh_1:8", "mesh_0:64;mesh_1:16", and "mesh_0:32;mesh_1:32".
        If set to None, there is no maximum.

  Returns:
    a (mtf.LayoutRules, mtf.Shape) tuple.
  """
  best_layout_and_mesh_shape = (None, None)
  best_value = None
  for mesh_shape_list in _mesh_shape_iterator(num_machines,
                                              max_mesh_shape_dimensions):
    mesh_shape = mtf.Shape([mtf.Dimension("mesh_{}".format(i), size)
                            for i, size in enumerate(mesh_shape_list)])
    tf.logging.info("Computing layout for mesh shape: {}".format(mesh_shape))
    estimator = memory_estimator.MemoryEstimator(mtf_graph, mesh_shape,
                                                 mtf_outputs)
    optimizer = layout_optimizer.LayoutOptimizer(estimator)
    layout_string = optimizer.solve()
    value = optimizer.evaluate_layout(layout_string)
    if best_value is None or value < best_value:
      best_value = value
      best_layout_and_mesh_shape = (mtf.convert_to_layout_rules(layout_string),
                                    mesh_shape)
  return best_layout_and_mesh_shape


def _mesh_shape_iterator(num_machines, max_mesh_shape_dimensions=None):
  """Iterable of mesh shapes that use a certain number of machines.

  Args:
    num_machines: integer, a power of two, the number of machines available.
    max_mesh_shape_dimensions: optional integer, the maximum number of
        dimensions to consider in any layout.

  Yields:
    [int], the dimension sizes of a mesh shape.
  """
  if num_machines == 1:
    yield [1]
    return

  current_product = num_machines
  mesh_shape = [num_machines]
  while True:
    if (max_mesh_shape_dimensions is None
        or len(mesh_shape) <= max_mesh_shape_dimensions):
      yield list(mesh_shape)
    while mesh_shape[-1] == 2:
      current_product //= mesh_shape.pop()
      if not mesh_shape:
        return
    mesh_shape[-1] //= 2
    current_product //= 2
    while current_product < num_machines:
      mesh_shape.append(min(mesh_shape[-1], num_machines // current_product))
      current_product *= mesh_shape[-1]
