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

"""Check whether a layout is valid under Mesh TensorFlow.

Not all layouts can be used to lower a Mesh TensorFlow graph. Some Mesh
TensorFlow operations error when a certain Mesh TensorFlow dimension is assigned
to a mesh dimension (e.g. mtf.ConcatOperation with its concatenation dimension).
A Mesh TensorFlow dimension can only be assigned to a mesh dimension if the
former's size is evenly divisible by the latter's size. This module provides
methods to check these conditions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import fractions
import re


class LayoutValidator(object):
  """Validates potential Mesh TensorFlow layouts.

  Usage Example:
    mtf_graph = mtf.Graph()
    # Add operations to mtf_graph using Mesh TensorFlow.
    mesh_shape = mtf.Shape([("m1", 4), ("m2", 2)])
    layout_validator = valid_layouts.LayoutValidator(mtf_graph, mesh_shape)

    print(layout_validator.splittable_mtf_dimension_names)
    # Set of names of Mesh TensorFlow dimensions that may be assigned to mesh
    # dimensions.

    print(layout_validator.is_valid_assignment("batch", "m1"))
    # Whether the 'batch' Mesh TensorFlow dimension may be assigned to the 'm1'
    # mesh dimension. Unlike the previous method, this ensures that every
    # occurrence of the 'batch' dimension has a size that is evenly divisible by
    # the size of 'm1'.

  Attributes:
    splittable_mtf_dimension_names: a set(string) of the names of MTF dimensions
        that may be assigned in a layout.
    mesh_dimension_name_to_size: a {string: int}, mapping names of mesh
        dimensions to their size.
  """

  def __init__(self, mtf_graph, mesh_shape):
    """Initializer.

    Args:
      mtf_graph: an mtf.Graph, representing the Mesh TensorFlow computation of
          interest.
      mesh_shape: an mtf.Shape, representing the mesh of interest.
    """
    self._splittable_mtf_dimension_names = self._initialize_splittable_dimensions(
        mtf_graph)
    self._mtf_dimension_name_to_size_gcd = (
        self._initialize_mtf_dimension_name_to_size_gcd(mtf_graph))
    self._mesh_dimension_name_to_size = self._initialize_mesh_dimension_name_to_size(
        mesh_shape)

  @property
  def splittable_mtf_dimension_names(self):
    return self._splittable_mtf_dimension_names

  @property
  def mesh_dimension_name_to_size(self):
    return self._mesh_dimension_name_to_size

  def is_valid_assignment(self, mtf_dimension_name, mesh_dimension_name):
    """Whether this MTF dimension may be assigned to this mesh dimension.

    Args:
      mtf_dimension_name: string, the name of a Mesh TensorFlow dimension.
      mesh_dimension_name: string, the name of a mesh dimension.

    Returns:
      A boolean indicating whether the assignment is valid.
    """
    return ((mtf_dimension_name in self._splittable_mtf_dimension_names) and
            (self._mtf_dimension_name_to_size_gcd[mtf_dimension_name] %
             self._mesh_dimension_name_to_size[mesh_dimension_name] == 0))

  def _initialize_splittable_dimensions(self, mtf_graph):
    """Initializer for self._splittable_mtf_dimension_names.

    Args:
      mtf_graph: an mtf.Graph.

    Returns:
      A set(string) of the names of Mesh TensorFlow dimensions that may be
      assigned in a layout.
    """
    all_mtf_dimension_names = set()  # set(string)
    for mtf_operation in mtf_graph.operations:
      for mtf_tensor in mtf_operation.outputs:
        for mtf_dimension in mtf_tensor.shape.dims:
          if not re.match(r"_anonymous_\d*", mtf_dimension.name):
            all_mtf_dimension_names.add(mtf_dimension.name)

    unsplittable_mtf_dimension_names = set()  # set(string)
    for mtf_operation in mtf_graph.operations:
      unsplittable_mtf_dimension_names.update(mtf_operation.unsplittable_dims)

    return all_mtf_dimension_names - unsplittable_mtf_dimension_names

  def _initialize_mtf_dimension_name_to_size_gcd(self, mtf_graph):
    """Initializer for self._mtf_dimension_name_to_size_gcd.

    Args:
      mtf_graph: an mtf.Graph.

    Returns:
      A {string: int}, mapping the name of an MTF dimension to the greatest
      common divisor of all the sizes it has. All these sizes being evenly
      divisible by some x is equivalent to the GCD being divisible by x.
    """
    mtf_dimension_name_to_size_gcd = {}
    for mtf_operation in mtf_graph.operations:
      for mtf_tensor in mtf_operation.outputs:
        for mtf_dimension in mtf_tensor.shape.dims:
          mtf_dimension_name_to_size_gcd[mtf_dimension.name] = fractions.gcd(
              mtf_dimension_name_to_size_gcd.get(mtf_dimension.name,
                                                 mtf_dimension.size),
              mtf_dimension.size)

    return mtf_dimension_name_to_size_gcd

  def _initialize_mesh_dimension_name_to_size(self, mesh_shape):
    """Initializer for self._mesh_dimension_name_to_size.

    Args:
      mesh_shape: an mtf.Shape.

    Returns:
      A {string: int} mapping mesh dimension names to their sizes.
    """
    mesh_dimension_name_to_size = {}  # {string: int}
    for mesh_dimension in mesh_shape.dims:
      mesh_dimension_name_to_size[mesh_dimension.name] = mesh_dimension.size
    return mesh_dimension_name_to_size
