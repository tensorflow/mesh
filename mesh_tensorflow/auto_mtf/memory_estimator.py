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

"""Base class to estimate the memory cost of a MTF computation.

We would like to estimate the footprint of computing a Mesh TensorFlow model.
Unfortunately, the size of the Mesh TensorFlow tensors isn't the full story;
a single Mesh TensorFlow operation with a single output tensor might lower into
multiple TensorFlow operations and multiple (temporary and output) TensorFlow
tensors.

However, the Mesh TensorFlow tensors serve as a quick, rough approximation to
the final memory usage. The base MemoryEstimator class uses these tensors to
compute a GraphInterface, without needing to lower the graph.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from mesh_tensorflow.auto_mtf import graph_interface
from mesh_tensorflow.auto_mtf import valid_layouts


class MemoryEstimator(object):
  """Estimates memory cost of a MTF graph based on the size of MTF tensors.

  Usage Example:
    estimator = memory_estimator.MemoryEstimator(mtf_graph, mesh_shape)
    layout_validator = estimator.get_layout_validator()
    graph = estimator.get_graph_interface()

  Attributes:
    mtf_graph: an mtf.Graph, see argument in __init__.
    mesh_shape: an mtf.Shape, see argument in __init__.
    mtf_outputs: an iterable of mtf.Tensor, see argument in __init__.
  """

  def __init__(self, mtf_graph, mesh_shape, mtf_outputs=()):
    """Initializer.

    Args:
      mtf_graph: a mtf.Graph.
      mesh_shape: an mtf.Shape.
      mtf_outputs: an optional iterable of mtf.Tensor, representing the outputs
          of the computation.
    """
    self.mtf_graph = mtf_graph
    self.mesh_shape = mesh_shape
    self.mtf_outputs = mtf_outputs

    self._layout_validator = None  # valid_layouts.LayoutValidator
    self._graph_interface = None  # graph_interface.GraphInterface

  def get_layout_validator(self):
    """LayoutValidator for the model and mesh_shape.

    Returns:
      a valid_layouts.LayoutValidator
    """
    if self._layout_validator is None:
      self._compute_layout_validator()
    return self._layout_validator

  def get_graph_interface(self):
    """GraphInterface representation of the model's computation graph.

    Returns:
      a graph_interface.GraphInterface
    """
    if self._graph_interface is None:
      self._compute_graph_interface()
    return self._graph_interface

  def _compute_layout_validator(self):
    """Computes self._layout_validator."""
    self._layout_validator = valid_layouts.LayoutValidator(self.mtf_graph,
                                                           self.mesh_shape)

  def _compute_graph_interface(self):
    """Computes self._graph_interface."""
    self._graph_interface = graph_interface.GraphInterface(self.mtf_graph)
    for mtf_output in self.mtf_outputs:
      self._graph_interface.set_tensor_final(mtf_output.name)
