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

"""Graph representation returned by cost estimators.

Cost estimators need to return a notion of computational graph, but it can
be complicated and expensive to work with tf.Graph and mtf.Graph. The
GraphInterface class serves as this return value. The base class returns
information corresponding to a tf.Graph or mtf.Graph, but subclasses may
return information corresponding to a mix of graphs.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math
import mesh_tensorflow as mtf
import tensorflow.compat.v1 as tf
from tensorflow.core.framework import cost_graph_pb2


class GraphInterface(object):
  """tf.Graph & mtf.Graph common representation which produces a CostGraphDef.

  Attributes:
    canonical_device: string or None, the name of the canonical device for
        is_tensor_on_canonical_device

  Usage Example:
    mtf_graph = mtf.Graph()
    # Add operations to mtf_graph using Mesh TensorFlow.
    graph = graph_interface.GraphInterface(mtf_graph)
    for operation_name in graph.get_all_operation_names():
      print("Operation: {}".format(operation_name))
      for input_name in graph.get_operation_input_names(operation_name):
        print("  Input: {}".format(input_name))
      for output_name in graph.get_operation_output_names(operation_name):
        print("  Output: {}".format(output_name))
        # Tensor names can also be used to retrieve data type, shape, and Mesh
        # TensorFlow dimension names.
      # Operation names can also be used to get Mesh TensorFlow dimension names.
    cost_graph = graph.compute_cost_graph()
    # Give cost_graph to a scheduler to compute schedule.
    memory_contents = graph.compute_memory_contents_under_schedule(schedule)
  """

  def __init__(self, graph, canonical_device=None):
    """Initializer.

    Args:
      graph: either a tf.Graph or mtf.Graph.
      canonical_device: optional string, the name of the canonical device for
          IsTensoronCanonicalDevice.
    """
    self._graph = graph
    self.canonical_device = canonical_device
    self._operations = self._initialize_operations()
    self._operation_name_to_id = self._initialize_operation_name_to_id()
    self._tensor_name_to_ids = self._initialize_tensor_name_to_ids()
    self._final_tensors = set()  # set(tf.Tensor or mtf.Tensor)

  def get_num_operations(self):
    """The number of operations in the graph.

    Returns:
      an integer, the number of operations.
    """
    return len(self._operations)

  def get_all_operation_names(self):
    """Generates the names of all operations in the graph.

    Yields:
      a string, the name of an operation.
    """
    for operation in self._operations:
      yield operation.name

  def get_operation_input_names(self, operation_name):
    """Generates the names of all input tensors of an operation.

    Args:
      operation_name: a string, the name of an operation in the graph.

    Yields:
      a string, the name of an input tensor.
    """
    for input_tensor in self._name_to_operation(operation_name).inputs:
      yield input_tensor.name

  def get_operation_output_names(self, operation_name):
    """Generates the names of all output tensors of an operation.

    Args:
      operation_name: a string, the name of an operation in the graph.

    Yields:
      a string, the name of an output tensor.
    """
    for output_tensor in self._name_to_operation(operation_name).outputs:
      yield output_tensor.name

  def get_all_tensor_names(self):
    """Generates the names of all tensors in the graph.

    Yields:
      a string, the name of a tensor.
    """
    for tensor in self._get_tensors():
      yield tensor.name

  def get_tensor_dtype(self, tensor_name):
    """The tf.Dtype of a tensor.

    Args:
      tensor_name: string, the name of a tensor in the graph.

    Returns:
      a tf.DType
    """
    return self._name_to_tensor(tensor_name).dtype

  def get_tensor_shape(self, tensor_name):
    """The tf.TensorShape of a tensor.

    Args:
      tensor_name: string, the name of a tensor in the graph.

    Returns:
      a tf.TensorShape
    """
    tensor = self._name_to_tensor(tensor_name)

    if isinstance(tensor, mtf.Tensor):
      return tf.TensorShape(tensor.shape.to_integer_list)
    else:  # tf.Tensor
      return tensor.shape

  def get_tensor_num_entries(self, tensor_name, partial_layout=None,
                             mesh_dimension_to_size=None):
    """The number of entries in a tensor.

    If partial_layout is specified, then mesh_dimension_to_size must also be. In
    this case, the number of entries on a single device is returned.

    Args:
      tensor_name: a string, name of a tensor in the graph.
      partial_layout: an optional {string: string}, from MTF dimension name to
          mesh dimension name.
      mesh_dimension_to_size: an optional {string: int}, from mesh dimension
          name to size.

    Returns:
      an integer
    """
    shape = self.get_tensor_shape(tensor_name)
    # We don't have to worry about divisiblity issues because Mesh TensorFlow
    # only allows evenly divisible assignments.
    num_entries = 1
    for dim in shape.dims:
      num_entries = num_entries * dim.value

    if not partial_layout:
      return num_entries

    for mtf_dimension_name in self.get_tensor_mtf_dimension_names(tensor_name):
      if mtf_dimension_name not in partial_layout:
        continue
      mesh_dimension_name = partial_layout[mtf_dimension_name]
      mesh_dimension_size = mesh_dimension_to_size[mesh_dimension_name]
      num_entries = int(math.ceil(num_entries / mesh_dimension_size))

    return num_entries

  def get_tensor_size(self, tensor_name, partial_layout=None,
                      mesh_dimension_to_size=None):
    """The size of a tensor in bytes.

    If partial_layout is specified, then mesh_dimension_to_size must also be. In
    this case, the size on a single device is returned.

    Args:
      tensor_name: a string, name of a tensor in the graph.
      partial_layout: an optional {string: string}, from MTF dimension name to
          mesh dimension name.
      mesh_dimension_to_size: an optional {string: int}, from mesh dimension
          name to size.

    Returns:
      an integer
    """
    return (self.get_tensor_dtype(tensor_name).size *
            self.get_tensor_num_entries(tensor_name, partial_layout,
                                        mesh_dimension_to_size))

  def get_tensor_device(self, tensor_name):
    """The device of a tensor.

    Note that only tf tensors have device assignments.

    Args:
      tensor_name: a string, name of a tensor in the graph.

    Returns:
      a string or None, representing the device name.
    """
    tensor = self._name_to_tensor(tensor_name)
    if isinstance(tensor, tf.Tensor):
      return tensor.device
    else:  # mtf.Tensor
      return None

  def is_tensor_on_canonical_device(self, tensor_name):
    """Whether the tensor is on the first (canonical) device.

    Tensors not assigned to a device are assumed to be on all devices, including
    the canonical device.

    Args:
      tensor_name: a string, name of a tensor in the graph.

    Returns:
      a boolean indicating whether the tensor is on the first device.
    """
    device = self.get_tensor_device(tensor_name)
    return not device or device == self.canonical_device

  def get_operation_device(self, operation_name):
    """The device of an operation.

    Note that only tf operations have device assignments.

    Args:
      operation_name: a string, name of an operation in the graph.

    Returns:
      a string or None, representing the device name.
    """
    operation = self._name_to_operation(operation_name)
    if isinstance(operation, tf.Operation):
      return operation.device
    else:  # mtf.Operation
      return None

  def get_tensor_mtf_dimension_names(self, tensor_name):
    """The Mesh TensorFlow dimensions associated with a tensor.

    Args:
      tensor_name: a string, name of a tensor in the graph.

    Returns:
      a [string], the names of Mesh TensorFlow dimensions.
    """
    tensor = self._name_to_tensor(tensor_name)
    if isinstance(tensor, mtf.Tensor):
      return tensor.shape.dimension_names
    else:  # tf.Tensor
      return []

  def get_operation_mtf_dimension_names(self, operation_name):
    """The Mesh TensorFlow dimensions associated with an operation.

    Args:
      operation_name: a string, name of an operation in the graph.

    Returns:
      a set(string), the names of Mesh TensorFlow dimensions.
    """
    mtf_dimension_names = set()
    for tensor_name in self.get_operation_input_names(operation_name):
      mtf_dimension_names.update(self.get_tensor_mtf_dimension_names(
          tensor_name))
    for tensor_name in self.get_operation_output_names(operation_name):
      mtf_dimension_names.update(self.get_tensor_mtf_dimension_names(
          tensor_name))
    return mtf_dimension_names

  def set_tensor_final(self, tensor_name):
    """Denotes a tensor as a final output of the computation.

    Args:
      tensor_name: a string, name of a tensor in the graph.
    """
    tensor = self._name_to_tensor(tensor_name)
    self._final_tensors.add(tensor)

  def is_tensor_final(self, tensor_name):
    """Whether a tensor is a final output of the computation.

    Args:
      tensor_name: a string, name of a tensor in the graph.

    Returns:
      a boolean indicating whether the tensor was a final output.
    """
    tensor = self._name_to_tensor(tensor_name)
    return tensor in self._final_tensors

  def compute_cost_graph(self, devices=None):
    """Computes a CostGraphDef protobuf based on this graph.

    Defined in tensorflow/core/framework/cost_graph.proto.

    Args:
      devices: optional [string], the names of devices to consider. If
          specified, any tensor on a device not listed is given a size of zero.
          Any device-less tensor (e.g. Mesh TensorFlow tensor) is not affected.

    Returns:
      a CostGraphDef protobuf with a Node for every operation in the graph, each
      of which is populated with size/dtype information for its inputs and
      outputs (which match the input/output order of the operation).
    """
    cost_graph_def = cost_graph_pb2.CostGraphDef()

    for i, operation_name in enumerate(self.get_all_operation_names()):
      node = cost_graph_def.node.add(
          name=operation_name,
          device=self.get_operation_device(operation_name),
          id=i)
      for input_name in self.get_operation_input_names(operation_name):
        id1, id2 = self._tensor_name_to_ids[input_name]
        node.input_info.add(preceding_node=id1, preceding_port=id2)

      for output_name in self.get_operation_output_names(operation_name):
        tensor_device = self.get_tensor_device(output_name)
        # devices = [] is not the same as None, and tensor_device = '' is also
        # not the same as None.
        if devices is None or tensor_device is None or tensor_device in devices:
          node.output_info.add(
              size=self.get_tensor_size(output_name),
              alias_input_port=-1,
              dtype=self.get_tensor_dtype(output_name).as_datatype_enum,
              shape=self.get_tensor_shape(output_name).as_proto(),
          )
        else:
          node.output_info.add(
              size=0,
              alias_input_port=-1,
              dtype=self.get_tensor_dtype(output_name).as_datatype_enum,
          )

        # NOTE(joshuawang): Unfortunately, the CostGraphDef protobuf has final
        # operations, not tensors. As a result, we have to declare any operation
        # that outputs a final tensor as final, which may expand the final set
        # of tensors to keep in memory. This issue also arises in the scheduler
        # code we will interface with.
        if self.is_tensor_final(output_name):
          node.is_final = True

    return cost_graph_def

  def compute_memory_contents_under_schedule(self, schedule):
    """The in-memory tensors present when executing each operation in schedule.

    Simulates running operations in the order given by a schedule. Keeps track
    of the tensors in memory at every point in time, and outputs a list (one
    entry for each point in time) of all sets of all memory contents (i.e. a
    frozenset of strings) ever seen in this execution.

    It is assumed (but not checked) that schedule is a valid topological sort of
    the operations in this graph.

    Args:
      schedule: A list of integer ids; the order to run operations in.

    Returns:
      a list of frozenset of strings, where the ith entry describes the tensors
      in memory when executing operation i (where schedule[i] is an index into
      get_all_operation_names()).
    """
    out_degree = self._compute_initial_out_degree()

    curr_memory_contents = set()
    memory_contents_for_each_operation = []

    for operation_id in schedule:
      operation_name = self._operations[operation_id].name
      # Allocate new memory to perform the computation at this node.
      for output_name in self.get_operation_output_names(operation_name):
        curr_memory_contents.add(output_name)
      memory_contents_for_each_operation.append(frozenset(curr_memory_contents))

      # Free any tensors which are no longer needed.
      for output_name in self.get_operation_output_names(operation_name):
        if out_degree[output_name] == 0:
          curr_memory_contents.remove(output_name)
      for input_name in self.get_operation_input_names(operation_name):
        out_degree[input_name] -= 1
        if out_degree[input_name] == 0:
          curr_memory_contents.remove(input_name)

    return memory_contents_for_each_operation

  def _initialize_operations(self):
    """Initializer for _operations.

    Raises:
      TypeError: _graph is not a tf.Graph or mtf.Graph.

    Returns:
      a list of (tf.Operation or mtf.Operation)
    """
    if isinstance(self._graph, tf.Graph):
      return self._graph.get_operations()
    elif isinstance(self._graph, mtf.Graph):
      return self._graph.operations
    else:
      raise TypeError('Graph is not tf.Graph or mtf.Graph: {}'
                      .format(type(self._graph)))

  def _initialize_operation_name_to_id(self):
    """Initializer for _operation_name_to_id.

    Returns:
      a {string: int}, mapping operation names to their index in _operations.
    """
    operation_name_to_id = {}
    for i, operation in enumerate(self._operations):
      operation_name_to_id[operation.name] = i
    return operation_name_to_id

  def _initialize_tensor_name_to_ids(self):
    """Initializer for _tensor_name_to_ids.

    Returns:
      a {string: (int, int)}, mapping the name of tensor T to the index of T's
          operation in _operations and T's index in T's operation's outputs.
    """
    tensor_name_to_ids = {}
    for i, operation in enumerate(self._operations):
      for j, tensor in enumerate(operation.outputs):
        tensor_name_to_ids[tensor.name] = (i, j)
    return tensor_name_to_ids

  def _get_tensors(self):
    """Generator for all tensors.

    Yields:
      a tf.Tensor or mtf.Tensor
    """
    for operation in self._operations:
      for tensor in operation.outputs:
        yield tensor

  def _name_to_operation(self, operation_name):
    """The operation with the given name.

    Args:
      operation_name: a string, name of a operation in the graph.

    Returns:
      a tf.Operation or mtf.Operation
    """
    return self._operations[self._operation_name_to_id[operation_name]]

  def _name_to_tensor(self, tensor_name):
    """The tensor with the given name.

    Args:
      tensor_name: a string, name of a tensor in the graph.

    Returns:
      a tf.Tensor or mtf.Tensor
    """
    id1, id2 = self._tensor_name_to_ids[tensor_name]
    return self._operations[id1].outputs[id2]

  def _compute_initial_out_degree(self):
    """The number of operations which use each tensor as input.

    Returns:
      a {string, int} mapping tensor name to the number of operations which use
      it as input, or one plus that quantity if the tensor is final.
    """
    out_degree = collections.defaultdict(int)

    # Pretend that final tensors have an additional degree so they are not
    # freed.
    for tensor_name in self.get_all_tensor_names():
      if self.is_tensor_final(tensor_name):
        out_degree[tensor_name] = 1

    for operation_name in self.get_all_operation_names():
      for input_name in self.get_operation_input_names(operation_name):
        out_degree[input_name] += 1

    return out_degree
