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

"""Mesh TensorFlow ops."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import copy
import functools
import itertools
import operator
import os
import re

from mesh_tensorflow import utils
import numpy as np
import six
from six.moves import xrange  # pylint: disable=redefined-builtin

import tensorflow.compat.v1 as tf

# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.ops.gen_nn_ops import conv3d_backprop_input_v2
from tensorflow.python.ops.nn_ops import conv3d_backprop_filter_v2

Dimension = collections.namedtuple("Dimension", ["name", "size"])


def convert_to_dimension(d):
  """Converts input to a Dimension.

  Args:
    d: Dimension, tuple (string, int), or None.

  Returns:
    Dimension or None.

  Raises:
    ValueError: If d cannot be converted to a Dimension.
  """
  if d is None:
    return None
  if isinstance(d, Dimension):
    if not isinstance(d.name, str) or not isinstance(d.size, int):
      raise ValueError("Bad dimension %s" % (d,))
    return d
  name, size = d
  if isinstance(name, str) and isinstance(size, int):
    return Dimension(name, size)
  else:
    raise ValueError("could not convert %s to Dimension" % (d,))


class Shape(object):
  """Shape of a Tensor or Mesh.

  #### Examples

  ```python
  # Create shape [4, 8] with names "x" and "y" respectively.
  shape = mtf.Shape([mtf.Dimension("x", 4), mtf.Dimension("y", 8)])
  ```
  """

  def __init__(self, dims):
    """Constructs a shape for a Tensor or Mesh.

    Args:
      dims: List-like of Dimensions.

    Raises:
      ValueError: If Dimensions are repeated.
    """
    self._dims = [convert_to_dimension(d) for d in tuple(dims)]
    if len(set(dims)) != len(dims):
      raise ValueError("Shape must not have repeated dimensions %s" % dims)

  @property
  def dims(self):
    return list(self._dims)

  @property
  def ndims(self):
    return len(self._dims)

  def __repr__(self):
    return self.to_string

  def __eq__(self, other):
    return self.dims == other.dims

  def __ne__(self, other):
    return self.dims != other.dims

  def __add__(self, other):
    if isinstance(other, Shape):
      other = other.dims
    if isinstance(other, Dimension):
      other = [other]
    return Shape(self.dims + other)

  def __sub__(self, other):
    if other is None:
      return self
    if isinstance(other, Shape):
      other = other.dims
    if isinstance(other, Dimension):
      if other not in self.dims:
        raise ValueError(
            "Subtracting a dimension from a shape requires that the shape"
            " contain that dimension.  Use shape - [dimension] for the case"
            " where the dimension may not be in the shape.")
      other = [other]
    return Shape([d for d in self.dims if d not in other])

  def __len__(self):
    return len(self._dims)

  def __getitem__(self, key):
    return self._dims[key]

  def __iter__(self):
    return iter(self._dims)

  @property
  def to_integer_list(self):
    return [d.size for d in self.dims]

  @property
  def size(self):
    return list_product(self.to_integer_list)

  @property
  def to_string(self):
    return "Shape[%s]" % ", ".join(
        ["%s=%d" % (d.name, d.size) for d in self.dims])

  @property
  def cumprod(self):
    """Cumulative product (exclusive) of Dimension sizes."""
    return _cumprod(self.to_integer_list)[:-1]

  def cumprod_to_tensor_axis(self, cumprod):
    """Maximum tensor axis i such that self.cumprod[i] == cumprod, or None."""
    try:
      return len(self) - 1 - self.cumprod[::-1].index(cumprod)
    except ValueError:
      return None

  @property
  def dimension_names(self):
    return [d.name for d in self.dims]

  def rename_dimension(self, old_name, new_name):
    """Returns a copy where one dimension is renamed."""
    if old_name not in self.dimension_names:
      raise ValueError("Shape %s does not have dimension named %s"
                       % (self, old_name))
    return Shape(
        [Dimension(new_name, d.size) if d.name == old_name else d
         for d in self.dims])

  def resize_dimension(self, name, new_size):
    """Returns a copy where one dimension has a different size."""
    if name not in self.dimension_names:
      raise ValueError("Shape %s does not have dimension named %s"
                       % (self, name))
    return Shape(
        [Dimension(name, new_size) if d.name == name else d
         for d in self.dims])

  def get_dim_by_name(self, name):
    """Get the Dimension with `name` from this shape.

    Args:
      name: a string, the name of the dimension we wish to get

    Returns:
      Dimension with `name`
    Raises:
      ValueError: if the shape does not contain a dimension with `name`
    """
    for d in self.dims:
      if d.name == name:
        return d
    raise ValueError("Dimension {} not found in {}.".format(
        name, self.to_string))


def convert_to_shape(x):
  """Converts input to a Shape.

  Args:
    x: Shape, str, or None.

  Returns:
    Shape or None.

  Raises:
    ValueError: If x cannot be converted to a Shape.
  """
  if x is None:
    return None
  if isinstance(x, Shape):
    return x
  if isinstance(x, str):
    x = _parse_string_to_list_of_pairs(x, seconds_to_int=True)
  return Shape(x)


class LayoutRules(object):
  """Represents layout of a computation.

  #### Examples

  ```python
  # Map "d_ff" and "heads" Tensor Dimensions to the "model" Mesh Dimension.
  layout_rules = mtf.LayoutRules([("d_ff", "model"), ("heads", "model")])
  ```
  """

  def __init__(self, pairs):
    """Constructs a layout.

    Args:
      pairs: Set-like of string pairs (tensor_dim_name, mesh_dim_name).
    """
    self._pairs = set(pairs)

  def __repr__(self):
    return "LayoutRules%s" % self._pairs

  def tensor_dimension_to_mesh_axis(self, tensor_dimension, mesh_shape):
    """Mesh axis associated with tensor dimension (or None).

    Args:
      tensor_dimension: Dimension.
      mesh_shape: Shape.

    Returns:
      Integer or None.

    Raises:
      ValueError: If one Tensor dimension maps to two mesh dimensions.
    """
    val = [i for i, mesh_dimension in enumerate(mesh_shape)
           if (tensor_dimension.name, mesh_dimension.name) in self._pairs]
    if len(val) > 1:
      raise ValueError(
          "Tensor dimension maps to multiple mesh dimensions"
          " tensor_dimension=%s mesh_shape=%s layout=%s"
          % (tensor_dimension, mesh_shape, self._pairs))
    return val[0] if val else None

  def tensor_layout(self, tensor_shape, mesh_shape):
    """Computes TensorLayout given a Tensor Shape and a Mesh Shape.

    Args:
      tensor_shape: Shape.
      mesh_shape: Shape.

    Returns:
      TensorLayout.

    Raises:
      ValueError: If two Tensor Dimensions map to the same Mesh Dimensions.
    """
    ret = [self.tensor_dimension_to_mesh_axis(d, mesh_shape)
           for d in tensor_shape]
    not_nones = [a for a in ret if a is not None]
    if len(not_nones) != len(set(not_nones)):
      raise ValueError(
          "Two Tensor Dimensions may not map to the same Mesh Dimension:"
          " layout=%s tensor_shape=%s mesh_shape=%s " %
          (self, tensor_shape, mesh_shape))
    return TensorLayout(ret)

  def mesh_dimension_name_to_tensor_dimension_names(self, mesh_dimension_name):
    return [tdn for tdn, mdn in self._pairs if mdn == mesh_dimension_name]


def convert_to_layout_rules(x):
  """Converts input to a LayoutRules.

  Args:
    x: LayoutRules, str, or set-like of string pairs.

  Returns:
    LayoutRules.
  """
  if isinstance(x, LayoutRules):
    return x
  if isinstance(x, str):
    x = _parse_string_to_list_of_pairs(x)
  return LayoutRules(x)


class TensorLayout(object):
  """Injective partial map between Tensor axes and Mesh axes.

  TensorLayout is a tuple of optional integers with length tensor.ndims. Each
  item is either a unique integer indicating the mesh axis over which that
  tensor dimension is split or None, indicating that this tensor dimension is
  not split.

  #### Examples

  ```python
  # Split first and last Tensor dimensions according to mesh axes 0 and 1.
  tensor_layout = mtf.TensorLayout([0, None, 1])
  ```
  """

  def __init__(self, tensor_axis_to_mesh_axis):
    """Creates a TensorLayout.

    Args:
      tensor_axis_to_mesh_axis: List-like where each element is an int or None.
    """
    self._tensor_axis_to_mesh_axis = tuple(tensor_axis_to_mesh_axis)

  def __eq__(self, other):
    return self.tensor_axis_to_mesh_axis == other.tensor_axis_to_mesh_axis

  def __ne__(self, other):
    return self.tensor_axis_to_mesh_axis != other.tensor_axis_to_mesh_axis

  def __repr__(self):
    return "TensorLayout%s" % (self.tensor_axis_to_mesh_axis,)

  def __len__(self):
    return len(self._tensor_axis_to_mesh_axis)

  def __getitem__(self, key):
    return self._tensor_axis_to_mesh_axis[key]

  def __iter__(self):
    return iter(self._tensor_axis_to_mesh_axis)

  @property
  def tensor_axis_to_mesh_axis(self):
    """Converts to a tuple of optional integers."""
    return self._tensor_axis_to_mesh_axis

  @property
  def is_fully_replicated(self):
    """Whether all tensor dimensions map to None."""
    return self.tensor_axis_to_mesh_axis == (None,) * len(self)

  def mesh_axis_to_tensor_axis(self, mesh_ndims):
    """For each mesh axis, which Tensor axis maps to it.

    Args:
      mesh_ndims: int.

    Returns:
      Tuple of optional integers, with length mesh_ndims.
    """
    ta2ma = self._tensor_axis_to_mesh_axis
    return tuple(
        [ta2ma.index(mesh_axis) if mesh_axis in ta2ma else None
         for mesh_axis in xrange(mesh_ndims)])


class Graph(object):
  """Mesh-TensorFlow graph."""

  def __init__(self):
    self._operations = []
    self._trainable_variables = []
    self._all_variables = []
    # Maps a name used in the graph to the next id to use for that name.
    self._names_in_use = {}
    self.name_to_variable = {}
    self.captured_variable_scope = tf.get_variable_scope()

  def __repr__(self):
    return self.to_string

  @property
  def operations(self):
    return self._operations

  @property
  def trainable_variables(self):
    return self._trainable_variables

  @property
  def all_variables(self):
    return self._all_variables

  @property
  def to_string(self):
    return "\n".join([op.to_string for op in self.operations])

  def unique_name(self, name, mark_as_used=True):
    """Like tf.Graph.unique_name, returns a unique operation name for `name`.

    Args:
      name: The name for an operation.
      mark_as_used: whether to mark this name as being used.

    Returns:
      A string to use as the name for the operation.
    """
    scope_name = tf.get_variable_scope().name
    if scope_name:
      name = scope_name + "/" + name

    # As in TensorFlow, treat names as case insensitive when deciding whether
    # they are in use.
    name_key = name.lower()
    i = self._names_in_use.get(name_key, 0)
    if mark_as_used:
      self._names_in_use[name_key] = i + 1
    if i > 0:
      base_name_key = name_key
      while name_key in self._names_in_use:
        name_key = "%s_%d" % (base_name_key, i)
        i += 1
      if mark_as_used:
        self._names_in_use[name_key] = 1
      name = "%s_%d" % (name, i-1)

    return name

  def rewrite_stack_variables(self,
                              max_combined_variable_size=2 ** 30,
                              max_combined_slice_size=2 ** 27,
                              mesh_to_impl=None):
    """Rewrite the current graph to combine variables.

    This helps speed up graph construction times in the case of large meshes
    and large numbers of variables.

    This function should be called after graph construction  (it is called by
    default in the Lowering constuctor).

    When we find a set of variables with the same shape/dtype/etc, we replace
    them with one StackedVariable and an "unstack" operation.  The
    StackedVariable has multiple master variables (so as to maintain checkpiont
    compatibility), but only one slice variable per device.  We point the inputs
    of later operations to the outputs of the "unstack" operations, instead of
    the outputs of the defunct single variables.

    In order for variables to be combinable, they must be set in the same Assign
    operation(s) - so it is necessary to call mtf.grouped_assign() from the
    optimizer instead of many separate calls to mtf.assign().  The assign
    operations get rewritten to set the appropriate stacked variables.

    TODO(noam): Combining to larger sizes seems to cause errors on TPU.
      debug this.  Perhaps we should try to keep the combined master variables
      on the same device.

    Args:
      max_combined_variable_size: an integer
      max_combined_slice_size: an integer
      mesh_to_impl: an optional dictionary from Mesh to MeshImpl
    """
    # pylint: disable=protected-access
    all_variables = self._all_variables
    operations = self._operations
    self._operations = []
    self._all_variables = []
    self._trainable_variables = []
    # We can only stack varaibles which share the same set of assignment
    # operations.
    var_to_assign_ops = collections.defaultdict(str)
    for op in operations:
      if isinstance(op, Assign):
        for v in op._variables:
          var_to_assign_ops[v] += op.name + ", "
    # Two variables with the same "key" can be stacked together.
    def var_key(v):
      return str([v.mesh,
                  v.shape,
                  str(v.dtype.__dict__),
                  v.trainable,
                  var_to_assign_ops[v]])
    key_to_vars = collections.defaultdict(collections.deque)
    for v in all_variables:
      key_to_vars[var_key(v)].append(v)
    individual_to_stacked = {}
    for op in operations:
      if isinstance(op, StackedVariable):
        raise ValueError("stack_variables() should not be called twice.")
      elif isinstance(op, Variable):
        if op.name in individual_to_stacked:
          continue
        similar_vars = key_to_vars[var_key(op)]
        num_to_stack = len(similar_vars)
        if max_combined_variable_size is not None:
          num_to_stack = min(
              num_to_stack, max_combined_variable_size // op.shape.size)
        if mesh_to_impl is not None:
          mesh_impl = mesh_to_impl[op.mesh]
          if mesh_impl.size == 1:
            num_to_stack = 1  # no point in stacking for single processors.
          slice_size = mesh_impl.slice_size(op.shape)
          num_to_stack = min(
              num_to_stack, max_combined_slice_size // slice_size)
        num_to_stack = max(1, num_to_stack)
        to_stack = [similar_vars.popleft() for _ in xrange(num_to_stack)]
        if num_to_stack > 1:
          stacked_var = StackedVariable(to_stack)
          stack_dim = stacked_var.shape.dims[0]
          unstacked = unstack(stacked_var.outputs[0], stack_dim)
          unstack_op = unstacked[0].operation
          # replace the output Tensors of the unstack operation with the
          # Tensors which were the outputs of the original variable operations.
          # Later operations use these Tensors as inputs.
          unstack_op._outputs = [v.outputs[0] for v in to_stack]
          for t in unstack_op._outputs:
            t._operation = unstack_op
          for idx, v in enumerate(to_stack):
            individual_to_stacked[v.name] = stacked_var, idx
        else:
          assert op == to_stack[0]
          self._operations.append(op)
          self._all_variables.append(op)
          if op.trainable:
            self._trainable_variables.append(op)
      else:
        if isinstance(op, Assign):
          # Rewrite the grouped assignment to stack up the values and then
          # assign to the stacked variables.
          new_variables = []
          new_values = []
          var_to_val = dict(zip([v.name for v in op._variables], op._inputs))
          for var, val in zip(op._variables, op._inputs):
            if var.name in individual_to_stacked:
              stacked_var, pos = individual_to_stacked[var.name]
              if pos == 0:
                vals = [var_to_val[n] for n in stacked_var.original_names]
                new_variables.append(stacked_var)
                new_values.append(
                    stack(vals, stacked_var.shape.dims[0].name, 0))
            else:
              new_variables.append(var)
              new_values.append(val)
          op._variables = new_variables
          op._inputs = new_values
        self._operations.append(op)
    # pylint: enable=protected-access

  def combine_assignments(self, assignments):
    """Rewrite the current graph to combine "Assign" operations.

    Combine similar Assign operations into grouped Assign operations.
    This is useful when using the rewrite_stack_variables() optimization,
    since variables can only be stacked if they are present in the same set
    of Assign operations.

    This function takes a list of Assign operations and returns a possibly
    shorter list of Assign operations.  The input Assignment operations
    are removed from the graph and become invalid.

    Args:
      assignments: a list of Assign objects
    Returns:
      a list of Assign objects
    """
    group_by_fn = collections.defaultdict(list)
    for a in assignments:
      if not isinstance(a, Assign):
        raise ValueError("ops should be instances of mtf.Assign")
      group_by_fn[a.assign_fn].append(a)
    assignments_set = set(assignments)
    self._operations = [
        op for op in self._operations if op not in assignments_set]
    ret = []
    for fn, ops in six.iteritems(group_by_fn):
      variables = []
      values = []
      for a in ops:
        variables.extend(a.variables)
        values.extend(a.inputs)
      ret.append(Assign(variables, values, fn))
    return ret

  def make_variables_untrainable(self, variables):
    """Makes the variables untrainable.

    Args:
      variables: a list of Variable objects
    """
    variables = set(variables)
    for v in variables:
      v._trainable = False  # pylint: disable=protected-access
    self._trainable_variables = [
        v for v in self._trainable_variables if v not in variables
    ]

  def clone_operations(self, ops, input_mapping):
    """Clone a portion of the graph, but with different inputs.

    The differnt inputs are specified by the `input_mapping` dictionary, which
    maps from input Tensor in the original operations to input Tensor in the
    cloned operations.  If an original operation uses an external input that is
    not in `input_mapping`, then the original input is used for the cloned
    operation.

    The function returns a list of cloned operations as well an
    `extended_mapping` dictionary which consits of the union of the input
    mapping and the map from original-operation-output to
    cloned-operation-output.

    Variables and Random operations are not cloned.

    Args:
      ops: a list of operations
      input_mapping: a dictionary from Tensor to Tensor
    Returns:
      cloned_operations: a list of operations
      extended_mapping: a dictionary from Tensor to Tensor
    """
    # pylint: disable=protected-access
    mapping = copy.copy(input_mapping)
    prev_num_operations = len(self.operations)
    for op in ops:
      if isinstance(op, Variable):
        continue
      if isinstance(op, RandomOperation):
        # The random values will be copied instead of recomputed.
        # TODO(noam): Use stateless_random to allow for recompute.
        tf.logging.warning(
            "Not cloning random operation, so as to ensure the same values.")
        continue
      new_op = copy.copy(op)
      # new_op._name = self.unique_name(op.name)
      self._operations.append(new_op)
      new_op._inputs = [mapping.get(t, t) for t in op._inputs]
      new_op._outputs = []
      for i, t in enumerate(op.outputs):
        new_t = Tensor(new_op, t.shape, t.dtype, t.name, i)
        new_t.usable = True
        new_op._outputs.append(new_t)
        if t in mapping:
          raise ValueError(
              "input mapping should not contain any of the outputs"
              " of the cloned operations")
        mapping[t] = new_t
    # pylint: enable=protected-access
    return self.operations[prev_num_operations:], mapping

  def capture_operations(self, fn):
    """Run a function and capture the list of operations it generates.

    Args:
      fn: a function taking no arguments
    Returns:
      fn_output: the function output
      captured_operations: a list of Operation
    """
    n = len(self.operations)
    y = fn()
    return y, self.operations[n:]


class Lowering(object):
  """Lowering of a Graph from Mesh-TensorFlow to TensorFlow.

  #### Examples

  Below we form a Graph with one Tensor and lower it to recover the original
  tf.Tensor.

  ```python
  from mesh_tensorflow import placement_mesh_impl

  graph = mtf.Graph()
  mesh = mtf.Mesh(graph, "my_mesh")
  inputs = tf.constant(0.)
  mtf_inputs = mtf.import_tf_tensor(mesh,
                                    inputs=inputs,
                                    shape=mtf.Shape([]))
  mesh_impl = placement_mesh_impl.PlacementMeshImpl(
      shape=[], layout={}, devices=[""])
  lowering = mtf.Lowering(graph, {mesh: mesh_impl})
  outputs = lowering.export_to_tf_tensor(mtf_inputs)  # tf.constant(0.)
  ```
  """

  def __init__(self, graph, mesh_to_impl, autostack=True, log_file=None):
    """Creates a Lowering of a Graph.

    Args:
      graph: Graph.
      mesh_to_impl: {Mesh: MeshImpl}. Keys are the Mesh's in the graph and
        their values are MeshImpl's, which map Tensor Dimension names to
        Mesh Dimension names.
      autostack: a boolean.  If True, then the graph gets rewritten to
        reduce the number of variables (see rewrite_stack_variables()).
        This is a helpful performance optimization for large meshes.
        For more fine-grained control, you can call
        graph.rewrite_stack_variables() yourself before creating the Lowering.
      log_file: an optional string. If provided, information about the variables
        and operations will also be logged to this file.
    """
    # tf.logging.info("LOWERING GRAPH:\n%s" % graph.to_string)
    self.mesh_to_impl = mesh_to_impl   # {Mesh: MeshImpl}
    self.graph = graph
    if autostack:
      self.autostack()
    self._counters = []
    self.tensors = {}                  # {Tensor: Mesh.LaidOutTensor}
    self.operations = {}               # {Operation: tf.Operation}
    self.variables = {}                # {Variable: LaidOutVariable}
    for op in graph.operations:
      # tf.logging.info("Lowering operation %s" % op.to_string)
      with tf.name_scope(op.name):
        op.lower(self)
      for out in op.outputs:
        self.add_counter(
            "output/%s" % type(op).__name__, self.laid_out_size(out))
        self.add_counter("output_unique/%s" % type(op).__name__, out.size)

    def log_info(f=None):
      """Log the variables and operations, possibly to file `f` as well."""
      log_variable_sizes(
          graph.trainable_variables,
          "Trainable Variables",
          verbose=True,
          mesh_to_impl=self.mesh_to_impl,
          log_file=f)
      log_variable_sizes(
          graph.all_variables,
          "All Variables",
          verbose=False,
          mesh_to_impl=self.mesh_to_impl,
          log_file=f)
      _log_info_also_to_file(
          "Counters:\n" + pretty_print_counters(self._counters), log_file=f)

    if log_file:
      with tf.io.gfile.GFile(log_file, mode="w") as f:
        log_info(f)
    else:
      log_info()

  def mesh_impl(self, m):
    if not isinstance(m, Mesh):
      m = m.mesh
    return self.mesh_to_impl[m]

  def export_to_tf_tensor(self, x):
    """Turn a Tensor into a tf.Tensor.

    Args:
      x: Tensor.

    Returns:
      tf.Tensor.
    """
    mesh_impl = self.mesh_impl(x)
    return mesh_impl.export_to_tf_tensor(
        x, self.tensors[x].to_laid_out_tensor())

  def lowered_operation(self, op):
    return self.operations[op]

  def copy_masters_to_slices(self):
    if os.environ.get("MTF_SEQUENCE_MODE", "") == "1":
      mesh_impls = [impl for impl in six.itervalues(self.mesh_to_impl)]
      assert len(mesh_impls) == 1
      mesh_impl = mesh_impls[0]
      return mesh_impl.copy_master_to_slice_ops[-1]
    else:
      return tf.group(
          [v.copy_master_to_slices for v in six.itervalues(self.variables)])

  def copy_slices_to_masters(self):
    return tf.group(
        [v.copy_slices_to_master for v in six.itervalues(self.variables)])

  def add_counter(self, key, value):
    assert isinstance(value, int)
    self._counters.append((key, value))

  @property
  def counters(self):
    return self._counters

  def laid_out_size(self, tensor):
    """Total size of all slices.

    Args:
      tensor: Tensor.

    Returns:
      int.
    """
    return self.mesh_impl(tensor).laid_out_size(tensor.shape)

  def set_tensor_lowering(self, tensor, laid_out_tensor):
    self.verify_slice_shapes(tensor, laid_out_tensor)
    self.tensors[tensor] = laid_out_tensor

  def verify_slice_shapes(self, tensor, laid_out_tensor):
    mesh_impl = self.mesh_impl(tensor)
    correct_shape = mesh_impl.slice_shape(tensor.shape)
    actual_shape = laid_out_tensor.slice_shape
    if actual_shape != correct_shape:
      raise ValueError(
          "Wrong slice shape: correct_shape = %s actual shape = %s"
          % (correct_shape, actual_shape))

  def autostack(self):
    """Rewrite graph to combine similarly-shaped variables (faster startup)."""
    num_slices = 0
    for v in self.graph.all_variables:
      num_slices += self.mesh_to_impl[v.mesh].size
    if num_slices >= 2 ** 16:
      # Startup times are slow with lots of variable slices.
      # Perform more aggressive stacking
      max_combined_slice_size = 2 ** 27
    else:
      # Stacking hurts memory utilization - only stack small variables.
      max_combined_slice_size = 2 ** 16
    self.graph.rewrite_stack_variables(
        mesh_to_impl=self.mesh_to_impl,
        max_combined_slice_size=max_combined_slice_size)


class Mesh(object):
  """A placeholder with no functionality.

  A Graph is built with each Tensor assigned to a Mesh. The Mesh does not
  know its shape or its implementation.

  A Lowering assigns each Mesh to a MeshImpl.
  """

  def __init__(self, graph, name, variable_placer=None):
    self._graph = graph
    self._name = name
    self._variable_placer = variable_placer

  @property
  def graph(self):
    return self._graph

  @property
  def variable_placer_fn(self):
    if self._variable_placer is not None:
      return self._variable_placer.device_function
    else:
      return "cpu:0"


class MeshImpl(object):
  """Implementation of a Mesh.

  Unlike Mesh, MeshImpl carries Shape and LayoutRules. Subclasses of MeshImpl
  also carry devices.

  #### Examples

  ```python
  shape = mtf.Shape([mtf.Dimension("batch", 4),
                     mtf.Dimension("model", 8)])
  layout_rules = mtf.LayoutRules([("batch", "batch"),
                                  ("d_ff", "model"),
                                  ("heads", "model")])
  mesh_impl = mtf.MeshImpl(shape=shape, layout_rules=layout_rules)
  ```
  """

  def __init__(self, shape, layout_rules):
    """Creates a mesh implementation.

    Args:
      shape: Shape.
      layout_rules: LayoutRules.
    """
    self._shape = convert_to_shape(shape)
    self._layout_rules = convert_to_layout_rules(layout_rules)

  @property
  def shape(self):
    return self._shape

  @property
  def ndims(self):
    return len(self._shape)

  @property
  def layout_rules(self):
    return self._layout_rules

  @property
  def size(self):
    return self.shape.size

  @property
  def supports_control_dependencies(self):
    return True

  def tensor_dimension_to_mesh_axis(self, tensor_dimension):
    """Mesh axis associated with tensor dimension (or None).

    Args:
      tensor_dimension: Dimension.

    Returns:
      int or None.
    """
    return self.layout_rules.tensor_dimension_to_mesh_axis(
        tensor_dimension, self.shape)

  def tensor_layout(self, arg):
    """Compute TensorLayout for a Tensor or a Shape.

    Args:
      arg: Tensor or Shape.

    Returns:
      TensorLayout.
    """
    if isinstance(arg, Tensor):
      arg = arg.shape
    return self.layout_rules.tensor_layout(arg, self.shape)

  def mesh_axis_to_cumprod(self, tensor_shape):
    """For each mesh axis, give the product of previous tensor axes.

    Args:
      tensor_shape: Shape.

    Returns:
      list with length self.ndims where each element is an integer or None.
    """
    tensor_layout = self.tensor_layout(tensor_shape)
    ma2ta = tensor_layout.mesh_axis_to_tensor_axis(self.ndims)
    ta2cumprod = tensor_shape.cumprod
    return [None if ta is None else ta2cumprod[ta] for ta in ma2ta]

  def slice_shape(self, tensor_shape):
    """Shape of each slice of the Tensor.

    Args:
      tensor_shape: Shape.

    Returns:
      list of integers with length tensor_shape.ndims.

    Raises:
      ValueError: If a Tensor dimension is not divisible by the corresponding
        Mesh dimension.
    """
    tensor_layout = self.tensor_layout(tensor_shape)
    ret = []
    for tensor_dim, mesh_axis in zip(
        tensor_shape, tensor_layout.tensor_axis_to_mesh_axis):
      if mesh_axis is None:
        ret.append(tensor_dim.size)
      else:
        mesh_dim = self.shape[mesh_axis]
        if tensor_dim.size % mesh_dim.size != 0:
          raise ValueError(
              "Tensor dimension size not divisible by mesh dimension size:"
              " tensor_shape=%s tensor_layout=%s"
              % (tensor_shape, tensor_layout))
        ret.append(tensor_dim.size // mesh_dim.size)
    return ret

  def slice_begin(self, tensor_shape, pnum):
    """Begin position for the tensor slice for the given processor.

    Args:
      tensor_shape: Shape.
      pnum: int <= self.size.

    Returns:
      list of integers with length tensor_shape.ndims.
    """
    tensor_layout = self.tensor_layout(tensor_shape)
    coordinates = pnum_to_processor_coordinates(self.shape, pnum)
    ret = []
    for dim_size, mesh_axis in zip(
        tensor_shape.to_integer_list, tensor_layout.tensor_axis_to_mesh_axis):
      if mesh_axis is None:
        ret.append(0)
      else:
        ret.append(
            dim_size // self.shape[mesh_axis].size * coordinates[mesh_axis])
    return ret

  def slice_size(self, tensor_shape):
    return list_product(self.slice_shape(tensor_shape))

  def laid_out_size(self, tensor_shape):
    """Total size of all slices.

    Args:
      tensor_shape: Shape.

    Returns:
      int.
    """
    return list_product(self.slice_shape(tensor_shape)) * self.size

  def slicewise(self, fn, *inputs):
    """Executes a function in parallel on all slices.

    Args:
      fn: function from tf.Tensors to tf.Tensor or a tuple of tf.Tensors.
      *inputs: list of inputs.  Each input is either a LaidOutTensor or
        is convertible to a tf.Tensor.

    Returns:
      LaidOutTensor, or a tuple of LaidOutTensors if fn returns a tuple.
    """
    raise NotImplementedError("Slicewise not implemented")

  def Print(self, x, data, message, **kwargs):  # pylint: disable=invalid-name
    """Calls tf.Print.

    Args:
      x: LaidOutTensor.
      data: list of LaidOutTensor.
      message: str.
      **kwargs: keyword arguments to tf.print.

    Returns:
      LaidOutTensor.
    """
    del data, message, kwargs
    tf.logging.warning("Warning - mtf.Print not implemented for this mesh type")
    return x

  def allreduce(self, x, mesh_axes, reduction_fn_string):
    """Grouped allreduce, (summed across the given dimensions).

    Args:
      x: LaidOutTensor.
      mesh_axes: list of integers, the mesh dimensions to be reduced.
      reduction_fn_string: "SUM" or "MAX".

    Returns:
      LaidOutTensor.
    """
    raise NotImplementedError("Allreduce not implemented")

  def allsplit(self, x, mesh_axis, split_axis, which=None):
    """Inverse of allconcat - split each slice and keep only one piece of it.

    The number of ways to split is the number of processors in the group.
    The part that is kept corresponds to the processor's index in the group.

    Args:
      x: LaidOutTensor.
      mesh_axis: int, the mesh axis along which to split.
      split_axis: int, the Tensor axis along which to split.
      which: an optional LaidOutTensor of integer scalars. Selects the slice to
        to keep, instead of the coordinate.

    Returns:
      LaidOutTensor.
    """
    if which is None:
      which = self.laid_out_pcoord(mesh_axis)
    num_splits = self.shape[mesh_axis].size
    def my_fn(x, which):
      slice_begin = [
          dimsize // num_splits * which if i == split_axis else 0
          for i, dimsize in enumerate(x.shape.as_list())]
      slice_size = [
          dimsize // num_splits if i == split_axis else dimsize
          for i, dimsize in enumerate(x.shape.as_list())]
      return tf.slice(x, slice_begin, slice_size)
    return self.slicewise(my_fn, x, which)

  def allconcat(self, x, mesh_axis, concat_axis):
    """Grouped allconcat (like MPI allgather followed by concat).

    Args:
      x: LaidOutTensor.
      mesh_axis: int, the mesh axis along which to group.
      concat_axis: int, the Tensor axis along which to concatenate.

    Returns:
      LaidOutTensor.
    """
    raise NotImplementedError("Allconcat not implemented")

  def alltoall(self, x, mesh_axis, split_axis, concat_axis):
    """Grouped alltoall (like MPI alltoall with splitting and concatenation).

    Args:
      x: LaidOutTensor.
      mesh_axis: int, the mesh axis along which to group.
      split_axis: int, the Tensor axis along which to split.
      concat_axis: int, the Tensor axis along which to concatenate.

    Returns:
      LaidOutTensor.
    """
    raise NotImplementedError("Alltoall not implemented")

  def receive(self, x, mesh_axis, source_pcoord):
    """Collective receive in groups.

    Each group contains the processors that differ only in mesh_axis.

    ```python
    group_size = self.shape[mesh_axis].size
    ```

    Args:
      x: a LaidOutTensor
      mesh_axis: an integer
      source_pcoord: a list of optional integers. Each element is either None
        or an integer in [0, group_size). If source_pcoord[k] is None, then the
        output for the k-th processor in each group is a zero tensor. If
        source_pcoord[k] is not None, then the output for the k-th processor in
        each group is equal to the input for the source_pcoord[k]-th processor
        in that group.

    Returns:
      a LaidOutTensor
    """
    raise NotImplementedError("Receive not implemented")

  def shift_by_n_processors(self, x, mesh_axis, offset, wrap):
    """Receive the slice from processor pcoord - offset.

    Args:
      x: a LaidOutTensor
      mesh_axis: an integer
      offset: an integer
      wrap: a boolean. If True, then wrap around. Otherwise, pad with zeros.

    Returns:
      a LaidOutTensor
    """
    n = self.shape[mesh_axis].size
    source_pcoord = []
    for i in xrange(n):
      c = i - offset
      if c != c % n:
        if wrap:
          c = c % n
        else:
          c = None
      source_pcoord.append(c)
    return self.receive(x, mesh_axis, source_pcoord)

  def laid_out_pnum(self):
    """Returns a LaidOutTensor containing the processor number.

    Returns:
      LaidOutTensor where each slice is an integer scalar.
    """
    raise NotImplementedError("laid_out_pnum not implemented")

  def laid_out_pcoord(self, mesh_axis):
    """Returns a LaidOutTensor containing the processor coordinate.

    Args:
      mesh_axis: int.

    Returns:
      LaidOutTensor where each slice is an integer scalar.
    """
    divisor = list_product(self.shape.to_integer_list[mesh_axis + 1:])
    modulus = self.shape[mesh_axis].size
    def my_fn(pnum):
      return (pnum // divisor) % modulus
    return self.slicewise(my_fn, self.laid_out_pnum())

  def laid_out_slice_num(self, tensor_shape):
    """A LaidOutTensor with an int32 scalar, identical for identical slices.

    This is useful for synchronizing random operations.

    Args:
      tensor_shape: a TensorShape
    Returns:
      a LaidOutTensor where each slice is an integer scalar.
    """
    ret = self.slicewise(lambda: tf.to_int32(0))
    tensor_layout = self.tensor_layout(tensor_shape)
    for mesh_axis in tensor_layout.tensor_axis_to_mesh_axis:
      if mesh_axis is not None:
        def my_fn(x, pcoord, mesh_dim_size):
          return x * mesh_dim_size + pcoord
        ret = self.slicewise(
            my_fn, ret, self.laid_out_pcoord(mesh_axis),
            self.shape[mesh_axis].size)
    return ret

  def broadcast_impl(self, old_slices, old_shape, new_shape):
    """Implementation of a broadcast operation.

    Args:
      old_slices: LaidOutTensor.
      old_shape: Shape.
      new_shape: Shape.

    Returns:
      LaidOutTensor.
    """
    new_slice_shape = self.slice_shape(new_shape)
    def tf_fn(x):
      return (tf.zeros(new_slice_shape, dtype=x.dtype) +
              _expand_dims(x, old_shape, new_shape))
    return self.slicewise(tf_fn, old_slices)

  def make_slices(self, tf_tensor, tensor_shape):
    """Turns a single tf.Tensor into a list of slices, one for each processor.

    Args:
      tf_tensor: tf.Tensor.
      tensor_shape: Shape.

    Returns:
      list of tf.tensor with length self.size.
    """
    tensor_layout = self.tensor_layout(tensor_shape)
    slice_shape = self.slice_shape(tensor_shape)
    def my_fn(pnum):
      if tensor_layout.is_fully_replicated:
        return tf_tensor
      else:
        slice_begin = self.slice_begin(tensor_shape, pnum)
        return tf.slice(tf_tensor, slice_begin, slice_shape)

    return parallel([tf_tensor.device] * self.size, my_fn,
                    list(xrange(self.size)))

  def combine_slices(self, slices, tensor_shape, device=None):
    """Turns a set of slices into a single tensor.

    Args:
      slices: list of tf.Tensor with length self.size.
      tensor_shape: Shape.
      device: optional str. If absent, we use the devices of the slices.

    Returns:
      tf.Tensor.
    """
    if tensor_shape.ndims == 0:
      return slices[0]

    ret = slices[:]
    tensor_layout = self.tensor_layout(tensor_shape)
    for mesh_dim, tensor_axis in zip(
        self.shape, tensor_layout.mesh_axis_to_tensor_axis(self.ndims)):
      slice_size = len(ret) // mesh_dim.size
      if tensor_axis is None:
        ret = ret[:slice_size]
      else:
        if device:
          devices = [device] * slice_size
        else:
          devices = [ret[i].device for i in xrange(slice_size)]
        concat_inputs = []
        for i in xrange(slice_size):
          concat_inputs.append(
              [ret[i + slice_size * j] for j in xrange(mesh_dim.size)])
        ret = parallel(
            devices, tf.concat, concat_inputs,
            axis=[tensor_axis] * len(devices))
    assert len(ret) == 1
    return ret[0]

  def export_to_tf_tensor(self, x, laid_out_x):
    """Turns a Tensor into a tf.Tensor.

    Args:
      x: Tensor.
      laid_out_x: LaidOutTensor.

    Returns:
      tf.Tensor.
    """
    raise NotImplementedError("export_to_tf_tensor not implemented")

  def import_tf_tensor(self, x, tf_x):
    """Imports a tf.Tensor, producing a LaidOutTensor.

    Args:
      x: Tensor.
      tf_x: tf.Tensor.

    Returns:
      LaidOutTensor.
    """
    raise NotImplementedError("Import not implemented")

  def einsum(self, equation, *slices):
    """Override this for custom einsum implementation.

    Args:
      equation: a string
      *slices: a list of tf.Tensor
    Returns:
      a Tensor
    """
    return tf.einsum(equation, *slices)


class LazyAllreduceSum(object):
  """Represents a LaidOutTensor with a lazy allreduce.

  The purpose of delaying allreduce is that it saves bandwidth to first add
  and then allreduce, as opposed to the other way around.
  """

  def __init__(self,
               mesh_impl,
               laid_out_input,
               mesh_axes,
               add_counter_fn=None):
    """Create a LazyAllreduceSum.

    Args:
      mesh_impl: a mesh_impl
      laid_out_input: a LaidOutTensor
      mesh_axes: a list of mesh axes
      add_counter_fn: a function taking no arguments which calls
        lowering.add_counter if and when the allreduce executes.
    Returns:
      a LazyAllreduceSum
    """
    self.mesh_impl = mesh_impl
    self.laid_out_input = laid_out_input
    self.mesh_axes = mesh_axes
    self.add_counter_fn = add_counter_fn
    self._reduced = None

  def to_laid_out_tensor(self):
    if not self._reduced:
      self._reduced = self.mesh_impl.allreduce(
          self.laid_out_input, self.mesh_axes, "SUM")
      if self.add_counter_fn:
        self.add_counter_fn()
    return self._reduced

  def __add__(self, other):
    """Add to another LazyAllreduceSum.

    Args:
      other: a LazyAllreduceSum or a LaidOutTensor
    Returns:
      a LazyAllreduceSum or a LaidOutTensor
    """
    if (isinstance(other, LazyAllreduceSum) and
        self.mesh_impl == other.mesh_impl and
        self.mesh_axes == other.mesh_axes):
      return LazyAllreduceSum(
          self.mesh_impl,
          self.mesh_impl.slicewise(
              tf.add, self.laid_out_input, other.laid_out_input),
          self.mesh_axes,
          add_counter_fn=self.add_counter_fn)
    else:
      return self.mesh_impl.slicewise(
          tf.add, self.to_laid_out_tensor(), other.to_laid_out_tensor())

  @property
  def slice_shape(self):
    return self.laid_out_input.slice_shape


def convert_args_to_laid_out_tensors(xs):
  """Convert list elements to laid-out-tensors when possible.

  Args:
    xs: a list
  Returns:
    a list
  """
  ret = []
  for x in xs:
    if hasattr(x, "to_laid_out_tensor"):
      ret.append(x.to_laid_out_tensor())
    else:
      ret.append(x)
  return ret


class Tensor(object):
  """A Distributed Tensor."""

  def __init__(self, operation, shape, dtype, name=None, index=0):
    """Create a Tensor.

    Args:
      operation: the Operation that outputs this tensor
      shape: a Shape
      dtype: a tf.DType
      name: an optional string
      index: optional integer, the index among operation's output tensors
    """
    if not isinstance(shape, Shape):
      raise ValueError("shape must be a Shape got %s" % shape.to_string)
    if not isinstance(dtype, tf.DType):
      raise ValueError("dtype must be a tf.DType got %s" % dtype)
    self._mesh = operation.mesh
    self._operation = operation
    self._shape = shape
    self._dtype = dtype
    if name is None:
      name = self.operation.name + ":" + str(index)
    self._name = name
    # A flag that we can turn off to assert that no one uses the tensor
    #   as the input to an operation.
    self.usable = True

  @property
  def shape(self):
    return self._shape

  @property
  def size(self):
    return self.shape.size

  @property
  def mesh(self):
    return self._mesh

  @property
  def graph(self):
    return self._mesh.graph

  @property
  def operation(self):
    return self._operation

  @property
  def dtype(self):
    return self._dtype

  @property
  def name(self):
    return self._name

  def __repr__(self):
    return self.to_string

  def __add__(self, other):
    return add(self, other)

  def __radd__(self, other):
    return add(self, other)

  def __sub__(self, other):
    return sub(self, other)

  def __rsub__(self, other):
    return sub(other, self)

  def __mul__(self, other):
    return multiply(self, other)

  def __rmul__(self, other):
    return multiply(self, other)

  def __neg__(self):
    return negative(self)

  def __truediv__(self, other):
    return divide(self, other)

  def __rtruediv__(self, other):
    return divide(other, self)

  def __floordiv__(self, other):
    return floordiv(self, other)

  def __rfloordiv__(self, other):
    return floordiv(other, self)

  def __mod__(self, other):
    return mod(self, other)

  def __rmod__(self, other):
    return mod(other, self)

  @property
  def to_string(self):
    return "Tensor[%s, %s, %s]" % (self.name, self.shape.to_string, self.dtype)


class Operation(object):
  """A Distributed Operation."""

  def __init__(self, inputs, mesh=None, name=None):
    """Initializer.

    Args:
      inputs: a list of Tensor
      mesh: an optional Mesh (if unspecified, will be inferred from first input)
      name: a string, which will get uniquified (in TensorFlow style)

    Raises:
      ValueError: mesh was not provided and there were no inputs to infer from.
    """
    if mesh is None:
      if not inputs:
        raise ValueError("mesh must be specified if no inputs")
      mesh = inputs[0].mesh
    self._inputs = inputs[:]
    self._outputs = []
    self._mesh = mesh
    # In a default operation, all dimensions are splittable.
    self._splittable_dims, self._unsplittable_dims = (
        self._initialize_all_dimensions_as_splittable())
    assert name is not None
    self._name = mesh.graph.unique_name(name)
    mesh.graph.operations.append(self)
    for t in inputs:
      if not t.usable:
        raise ValueError("Operation %s has unusable input %s" % (self, t))

  @property
  def graph(self):
    return self._mesh.graph

  @property
  def mesh(self):
    return self._mesh

  @property
  def name(self):
    return self._name

  @property
  def inputs(self):
    return self._inputs[:]

  @property
  def outputs(self):
    return self._outputs[:]

  @property
  def splittable_dims(self):
    """Frozenset of the names of dims safe to split when lowering this op."""
    return self._splittable_dims

  @property
  def unsplittable_dims(self):
    """Frozenset of the names of dims unsafe to split when lowering this op."""
    return self._unsplittable_dims

  @property
  def to_string(self):
    return "%s[Inputs=(%s) Outputs=(%s)]" % (
        type(self).__name__,
        ", ".join([t.to_string for t in self.inputs]),
        ", ".join([t.to_string for t in self.outputs]))

  @property
  def has_gradient(self):
    return (
        [t for t in self.inputs if t.dtype.is_floating] and
        [t for t in self.outputs if t.dtype.is_floating])

  def gradient(self, unused_grad_ys):
    raise NotImplementedError("Gradient not implemented")

  def lower(self, lowering):
    raise NotImplementedError("Lower not implemented")

  def _initialize_splittable_and_unsplittable_dims(
      self, default_splittability, exception_dims_iterable=None):
    """Initializer for splittable_dims and unsplittable_dims.

    Helper method to categorize all dimensions in the input/output tensors as
    either splittable or unsplittable.

    Args:
      default_splittability: a string which is either "splittable" or
        "unsplittable".
      exception_dims_iterable: an optional iterable of names of dimensions
        which are exceptions to the default splittability.

    Returns:
      splittable_dims and unsplittable_dims, two frozensets of names of
        dimensions (strings)

    Raises:
      ValueError: default_splittability is not one of "splittable" or
        "unsplittable".
    """
    default_dims = set()
    exception_dims = set()
    if exception_dims_iterable:
      exception_dims.update(exception_dims_iterable)

    for t in itertools.chain(self.inputs, self.outputs):
      for dim_name in t.shape.dimension_names:
        if dim_name not in exception_dims:
          default_dims.add(dim_name)

    if default_splittability == "splittable":
      return frozenset(default_dims), frozenset(exception_dims)
    elif default_splittability == "unsplittable":
      return frozenset(exception_dims), frozenset(default_dims)
    else:
      raise ValueError("default_splittability should be either \"splittable\" "
                       "or \"unsplittable\" but was {}"
                       .format(default_splittability))

  def _initialize_all_dimensions_as_splittable(self):
    """Helper init for the most common case: all dimensions may be split."""
    return self._initialize_splittable_and_unsplittable_dims("splittable")


class SlicewiseOperation(Operation):
  """Apply any tensorflow function slice-wise.

  Calls the Tensorflow function on each slice of the inputs to produce the
  corresponding slice of the outputs.  Gradients are computed through
  tensorflow.

  The user must specify "splittable_dims": a list of Dimensions which can
  be split while still keeping this computation valid.  For example, for
  component-wise functions, all the dimensions are splittable, but if the
  function is a reduction, the reduced dimensions are not splittable.
  """

  def __init__(self,
               tf_fn,
               inputs,
               output_shapes,
               output_dtypes,
               splittable_dims,
               grad_function=None,
               name=None):
    """Create a SlicewiseOperation.

    grad_function is a python function taking this operation and a gradients
    Tensor and producing input gradients tensors.
    e.g.
    def _square_grad(op, dy):
      return [dy * op.inputs[0] * 2]

    Args:
      tf_fn: a function taking n tf.Tensors and returning a tf.Tensor
      inputs: a list of n Tensors
      output_shapes: a list of Shapes
      output_dtypes: a list of dtypes
      splittable_dims: a list of Dimensions which are ok to split
      grad_function: an optional python function. Default to using tf.gradients
        pass in the number 0 to indicate no gradient
      name: an optional string
    """
    super(SlicewiseOperation, self).__init__(inputs, name=name or "slicewise")
    self._tf_fn = tf_fn
    self._outputs = [Tensor(self, shape, dtype) for (shape, dtype)
                     in zip(output_shapes, output_dtypes)]
    self._splittable_dims, self._unsplittable_dims = (
        self._initialize_splittable_and_unsplittable_dims(
            "unsplittable", [dim.name for dim in splittable_dims]))
    self._grad_function = grad_function

  @property
  def has_gradient(self):
    if self._grad_function == 0:
      return False
    return super(SlicewiseOperation, self).has_gradient

  def gradient(self, grad_ys):
    if self._grad_function is not None:
      return self._grad_function(self, *grad_ys)
    return GenericGradOperation(self, grad_ys).outputs

  def lower(self, lowering):
    # Check that only splittable dims are split
    mesh_impl = lowering.mesh_impl(self)
    for t in self.inputs + self.outputs:
      layout = mesh_impl.tensor_layout(t)
      for d, mesh_axis in zip(t.shape.dims, layout.tensor_axis_to_mesh_axis):
        if mesh_axis is not None and d.name not in self._splittable_dims:
          raise ValueError("dimension %s is not declared as splittable" % d)
    values = mesh_impl.slicewise(
        self._tf_fn, *[lowering.tensors[x] for x in self.inputs])
    if len(self.outputs) == 1:
      values = values,
    for output, value in zip(self.outputs, values):
      lowering.set_tensor_lowering(output, value)


def slicewise(tf_fn,
              xs,
              output_shape=None,
              output_dtype=None,
              splittable_dims=None,
              grad_function=None,
              name=None):
  """Slice-wise call to any tensorflow function.

  The output shape and dtype default to those of the first input.
  splittable_dims is a list of Dimensions which can be split while keeping the
  computation valid.

  Args:
    tf_fn: a function taking n tf.Tensors and returning a tf.Tensor
    xs: a list of n Tensors
    output_shape: a Shape (or list of shapes)
    output_dtype: a dtype (or list of dtypes)
    splittable_dims: a list of Dimensions which are ok to split
    grad_function: an optional gradients function.  If None, use tf gradient.
    name: an optional string

  Returns:
    a Tensor (or a tuple of Tensors)
  """
  multiple_outputs = isinstance(output_dtype, list)
  output_shapes = output_shape if multiple_outputs else [output_shape]
  output_dtypes = output_dtype if multiple_outputs else [output_dtype]

  op = SlicewiseOperation(
      tf_fn,
      xs,
      [convert_to_shape(shape) or xs[0].shape for shape in output_shapes],
      [dtype or xs[0].dtype for dtype in output_dtypes],
      splittable_dims,
      grad_function,
      name=name)
  return tuple(op.outputs) if multiple_outputs else op.outputs[0]


def cwise(tf_fn, xs, output_dtype=None, grad_function=None, name=None):
  """Component-wise operation with no broadcasting.

  Args:
    tf_fn: a component-wise function taking n tf.Tensor inputs and producing
      a tf.Tensor output
    xs: n Tensors
    output_dtype: an optional dtype
    grad_function: an optional python function
    name: an optional string

  Returns:
    a Tensor
  """
  return slicewise(
      tf_fn, xs, output_dtype=output_dtype, splittable_dims=xs[0].shape.dims,
      grad_function=grad_function, name=name or "cwise")


def identity(x, name="identity"):
  return cwise(tf.identity, [x], name=name)


def sin(x, name="sin"):
  return cwise(tf.sin, [x], name=name)


def cos(x, name="cos"):
  return cwise(tf.cos, [x], name=name)


def square(x, name="square"):
  return cwise(
      tf.square, [x], name=name,
      grad_function=lambda op, dy: [dy * op.inputs[0] * 2])


def sqrt(x, name="sqrt"):
  return cwise(
      tf.sqrt, [x], name=name,
      grad_function=lambda op, dy: [dy * 0.5 / op.outputs[0]])


def _rsqrt_grad(op, dy):
  return [dy * -0.5 * op.outputs[0] * op.outputs[0] * op.outputs[0]]


def rsqrt(x, name="rsqrt"):
  return cwise(
      tf.math.rsqrt, [x], name=name, grad_function=_rsqrt_grad)


def log(x, name="log"):
  return cwise(
      tf.math.log, [x], name=name,
      grad_function=lambda op, dy: [dy / op.inputs[0]])


def exp(x, name="exp"):
  return cwise(tf.exp, [x], name=name,
               grad_function=lambda op, dy: [dy * op.outputs[0]])


def sigmoid(x, name="sigmoid"):
  def grad_function(op, dy):
    y = op.outputs[0]
    return [y * (1.0 - y) * dy]
  return cwise(tf.sigmoid, [x], name=name, grad_function=grad_function)


def tanh(x, name="tanh"):
  def grad_function(op, dy):
    y = op.outputs[0]
    return [(1.0 - square(y)) * dy]
  return cwise(tf.tanh, [x], name=name, grad_function=grad_function)


def mtf_pow(x, y):
  """Call externally as mtf.pow()."""
  return exp(log(x) * y)


def negative(x, name="negative"):
  return cwise(tf.negative, [x], name=name,
               grad_function=lambda op, dy: [negative(dy)])


def logical_not(x, name="logical_not"):
  return cwise(tf.logical_not, [x], name=name)


def swish(x):
  """Swish activation from https://arxiv.org/abs/1710.05941 ."""
  return x * sigmoid(x)


def gelu(x):
  """Gaussian Error Linear Unit.

  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1606.08415
  Args:
    x: float Tensor to perform activation.

  Returns:
    `x` with the GELU activation applied.
  """
  cdf = 0.5 * (1.0 + tanh((np.sqrt(2 / np.pi) * (x + 0.044715 * x * x * x))))
  return x * cdf


def elu(x):
  """Exponential Linear Unit.

  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1511.07289
  Args:
    x: float Tensor to perform activation.

  Returns:
    'x' with the ELU activation applied.
  """
  return cwise(tf.nn.elu, [x], name="elu")


def selu(x):
  """Scaled Exponential Linear Unit.

  This is a smoother version of the RELU.
  Original paper: https://arxiv.org/abs/1706.02515
  Args:
    x: float Tensor to perform activation.

  Returns:
    'x' with the SELU activation applied.
  """
  return cwise(tf.nn.selu, [x], name="selu")


def softplus(x):
  """Softplus activation."""
  return cwise(tf.math.softplus, [x], name="softplus")


def reciprocal(x, name="reciprocal"):
  return cwise(
      tf.math.reciprocal, [x], name=name,
      grad_function=lambda op, dy: [negative(dy * square(op.outputs[0]))])


def _relu_grad(op, dy):
  return [dy * cast(greater(op.inputs[0], 0), op.inputs[0].dtype)]


def relu(x, name="relu"):
  return cwise(tf.nn.relu, [x], name=name, grad_function=_relu_grad)


def leaky_relu(x, alpha=0.2, name="leaky_relu"):
  def forward_function(x):
    return tf.nn.leaky_relu(x, alpha)

  def grad_function(op, dy):
    return [dy * cast(greater(op.inputs[0], 0), op.inputs[0].dtype) + \
            dy * cast(less_equal(op.inputs[0], 0), op.inputs[0].dtype) * alpha]

  return cwise(forward_function, [x], name=name, grad_function=grad_function)


def sign(x, name="sign"):
  ret = cwise(tf.sign, [x], name=name, grad_function=0)
  return ret


def mtf_abs(x):
  """Call externally as mtf.abs()."""
  return x * sign(x)


def cast(x, dtype, name="cast"):
  if dtype == x.dtype:
    return x
  return cwise(
      lambda x: tf.cast(x, dtype), [x], output_dtype=dtype, name=name,
      grad_function=lambda op, dy: [cast(dy, op.inputs[0].dtype)])


def to_float(x, name="to_float"):
  return cast(x, tf.float32, name=name)


def to_bfloat16(x, name="to_bfloat16"):
  return cast(x, tf.bfloat16, name=name)


def to_int32(x, name="to_int32"):
  return cast(x, tf.int32, name=name)


class GenericGradOperation(Operation):
  """Gradients that follow regular TF.

  Calling tf.gradients multiple times seems really slow in python.
  TODO(noam): can we speed this up using functions or some other method?
  """

  def __init__(self, forward_op, grad_ys, name=None):
    # tf.logging.info("forward inp %s, operations %s, grad_ys: %s",
    #                 forward_op.inputs, forward_op.outputs, grad_ys)
    super(GenericGradOperation, self).__init__(
        forward_op.inputs + forward_op.outputs + grad_ys,
        name=name or "generic_grad")
    self._grad_ys = grad_ys
    self._forward_op = forward_op
    self._outputs = [Tensor(self, x.shape, x.dtype, index=i)
                     for i, x in enumerate(forward_op.inputs)]

  def lower(self, lowering):
    # lists of lists of tf.Tensor
    all_ys = transpose_list_of_lists(
        [lowering.tensors[y].tensor_list for y in self._forward_op.outputs])
    all_xs = transpose_list_of_lists(
        [lowering.tensors[x].tensor_list for x in self._forward_op.inputs])
    all_grad_ys = transpose_list_of_lists(
        [lowering.tensors[dy].tensor_list for dy in self._grad_ys])
    all_grad_xs = [
        tf.gradients(  # pylint: disable=g-complex-comprehension
            ys=ys,
            xs=xs,
            grad_ys=grad_ys,
            unconnected_gradients=tf.UnconnectedGradients.ZERO)
        for ys, xs, grad_ys in zip(all_ys, all_xs, all_grad_ys)
    ]
    grad_xs = transpose_list_of_lists(all_grad_xs)
    for out, grad_x in zip(self.outputs, grad_xs):
      lowering.set_tensor_lowering(
          out,
          lowering.mesh_impl(self).LaidOutTensor.from_tensor_list(grad_x))


class ScalarMultiplyOperation(Operation):
  """Multiply by a tf Scalar (no backprop to scalar)."""

  def __init__(self, x, scalar, name=None):
    super(ScalarMultiplyOperation, self).__init__(
        [x], name=name or "scalar_mul")
    self._outputs = [Tensor(self, x.shape, x.dtype)]
    self._scalar = scalar

  def gradient(self, grad_ys):
    dy = grad_ys[0]
    return [dy * self._scalar]

  def lower(self, lowering):
    lowering.set_tensor_lowering(
        self.outputs[0],
        lowering.mesh_impl(self).slicewise(
            lambda x: x * self._scalar, lowering.tensors[self.inputs[0]]))


class ScalarAddOperation(Operation):
  """Add a tf Scalar (no backprop to scalar)."""

  def __init__(self, x, scalar, name=None):
    super(ScalarAddOperation, self).__init__([x], name=name or "scalar_add")
    self._outputs = [Tensor(self, x.shape, x.dtype)]
    self._scalar = scalar

  def gradient(self, grad_ys):
    return grad_ys

  def lower(self, lowering):
    lowering.set_tensor_lowering(
        self.outputs[0],
        lowering.mesh_impl(self).slicewise(
            lambda x: x + self._scalar, lowering.tensors[self.inputs[0]]))


class BinaryOpWithBroadcasting(Operation):
  """Binary operation with broadcasting."""

  def __init__(self, tf_fn, x1, x2, output_shape, output_dtype, name=None):
    super(BinaryOpWithBroadcasting, self).__init__(
        [x1, x2], name=name or "binary_op")
    if x1.dtype != x2.dtype:
      # If there is ever a binary operation with different operand types, then
      # we should add an argument allow_different_operand_dtypes=False.
      raise ValueError("Dtypes must be equal- got %s and %s"
                       % (x1.dtype, x2.dtype))
    assert isinstance(output_dtype, tf.DType)
    self._outputs = [Tensor(self, output_shape, output_dtype)]
    self._tf_fn = tf_fn

    # Rerun to take the new output into account.
    self._splittable_dims, self._unsplittable_dims = (
        self._initialize_all_dimensions_as_splittable())

  def gradient(self, unused_grad_ys):
    raise ValueError("Gradient not implememnted")

  def lower(self, lowering):
    x1 = self.inputs[0]
    x2 = self.inputs[1]
    output = self.outputs[0]
    laid_out_x1 = lowering.tensors[x1]
    laid_out_x2 = lowering.tensors[x2]
    mesh_impl = lowering.mesh_impl(self)
    if x1.shape != output.shape:
      laid_out_x1 = mesh_impl.slicewise(
          _expand_dims, laid_out_x1, x1.shape, output.shape)
    if x2.shape != output.shape:
      laid_out_x2 = mesh_impl.slicewise(
          _expand_dims, laid_out_x2, x2.shape, output.shape)
    lowering.set_tensor_lowering(
        self.outputs[0],
        mesh_impl.slicewise(
            self._tf_fn, laid_out_x1, laid_out_x2))


def binary_arguments_to_tensors(x1, x2):
  """Convert argument of a binary operation to Tensors.

  Args:
    x1: a Tensor or something convertible to a tf Scalar
    x2: a Tensor or something convertible to a tf Scalar

  Returns:
    new_x1: a Tensor
    new_x2: a Tensor

  Raises:
    ValueError: on failure
  """
  if not isinstance(x1, Tensor) and not isinstance(x2, Tensor):
    raise ValueError("at least one of x1 and x2 must be an mtf Tensor")
  elif isinstance(x1, Tensor) and isinstance(x2, Tensor):
    return x1, x2
  elif isinstance(x1, Tensor):
    return x1, import_tf_tensor(
        x1.mesh, tf.convert_to_tensor(x2, dtype=x1.dtype), Shape([]))
  else:
    return import_tf_tensor(x2.mesh, tf.convert_to_tensor(x1, dtype=x2.dtype),
                            Shape([])), x2


def binary_op_with_broadcasting(
    tf_fn, x1, x2, output_shape=None, output_dtype=None):
  x1, x2 = binary_arguments_to_tensors(x1, x2)
  output_shape = _infer_binary_broadcast_shape(x1.shape, x2.shape, output_shape)
  output_dtype = output_dtype or x1.dtype
  assert isinstance(output_dtype, tf.DType)
  return BinaryOpWithBroadcasting(
      tf_fn, x1, x2, convert_to_shape(output_shape),
      output_dtype).outputs[0]


def less(x1, x2, output_shape=None):
  return binary_op_with_broadcasting(
      tf.less, x1, x2, output_dtype=tf.bool, output_shape=output_shape)


def greater(x1, x2, output_shape=None):
  return binary_op_with_broadcasting(
      tf.greater, x1, x2, output_dtype=tf.bool, output_shape=output_shape)


def less_equal(x1, x2, output_shape=None):
  return binary_op_with_broadcasting(
      tf.less_equal, x1, x2, output_dtype=tf.bool, output_shape=output_shape)


def greater_equal(x1, x2, output_shape=None):
  return binary_op_with_broadcasting(
      tf.greater_equal, x1, x2, output_dtype=tf.bool, output_shape=output_shape)


def equal(x1, x2, output_shape=None):
  return binary_op_with_broadcasting(
      tf.equal, x1, x2, output_dtype=tf.bool, output_shape=output_shape)


def not_equal(x1, x2, output_shape=None):
  return binary_op_with_broadcasting(
      tf.not_equal, x1, x2, output_dtype=tf.bool, output_shape=output_shape)


def logical_and(x1, x2, output_shape=None):
  return binary_op_with_broadcasting(
      tf.logical_and, x1, x2, output_dtype=tf.bool, output_shape=output_shape)


def logical_or(x1, x2, output_shape=None):
  return binary_op_with_broadcasting(
      tf.logical_or, x1, x2, output_dtype=tf.bool, output_shape=output_shape)


def floordiv(x1, x2, output_shape=None):
  output_dtype = x1.dtype if isinstance(x1, Tensor) else x2.dtype
  return binary_op_with_broadcasting(
      tf.floordiv, x1, x2, output_dtype=output_dtype, output_shape=output_shape)


def mod(x1, x2, output_shape=None):
  output_dtype = x1.dtype if isinstance(x1, Tensor) else x2.dtype
  return binary_op_with_broadcasting(
      tf.mod, x1, x2, output_dtype=output_dtype, output_shape=output_shape)


class AddOperation(BinaryOpWithBroadcasting):
  """Binary addition with broadcasting."""

  def __init__(self, x1, x2, output_shape, name=None):
    super(AddOperation, self).__init__(
        tf.add, x1, x2, output_shape, x1.dtype, name=name or "add")

  def gradient(self, grad_ys):
    dy = grad_ys[0]
    return [reduce_sum(dy, output_shape=self.inputs[0].shape),
            reduce_sum(dy, output_shape=self.inputs[1].shape)]


class MinMaxOperation(BinaryOpWithBroadcasting):
  """Binary minimum/maximum with broadcasting."""

  def __init__(self, tf_fn, x1, x2, output_shape, name=None):
    super(MinMaxOperation, self).__init__(
        tf_fn, x1, x2, output_shape, x1.dtype, name=name or "add")

  def gradient(self, grad_ys):
    dy = grad_ys[0]
    return [dy * cast(equal(self.inputs[0], self.outputs[0]), dy.dtype),
            dy * cast(equal(self.inputs[1], self.outputs[0]), dy.dtype)]


def minimum(x1, x2, output_shape=None, name=None):
  """Binary minimum with broadcsting.

  Args:
    x1: a Tensor
    x2: a Tensor
    output_shape: an optional Shape
    name: an optional string
  Returns:
    a Tensor
  """
  output_shape = convert_to_shape(output_shape)
  with tf.name_scope(name, default_name="minimum"):
    x1, x2 = binary_arguments_to_tensors(x1, x2)
    return MinMaxOperation(
        tf.minimum, x1, x2, output_shape=_infer_binary_broadcast_shape(
            x1.shape, x2.shape, output_shape)).outputs[0]


def maximum(x1, x2, output_shape=None, name=None):
  """Binary maximum with broadcsting.

  Args:
    x1: a Tensor
    x2: a Tensor
    output_shape: an optional Shape
    name: an optional string
  Returns:
    a Tensor
  """
  output_shape = convert_to_shape(output_shape)
  with tf.name_scope(name, default_name="maximum"):
    x1, x2 = binary_arguments_to_tensors(x1, x2)
    return MinMaxOperation(
        tf.maximum, x1, x2, output_shape=_infer_binary_broadcast_shape(
            x1.shape, x2.shape, output_shape)).outputs[0]


class BroadcastOperation(Operation):
  """Broadcast - output dims are a superset of input dims, in any order."""

  def __init__(self, x, output_shape, name=None):
    super(BroadcastOperation, self).__init__([x], name=name or "broadcast")
    self._outputs = [Tensor(self, output_shape, x.dtype)]
    # Rerun to take the new output into account.
    self._splittable_dims, self._unsplittable_dims = (
        self._initialize_all_dimensions_as_splittable())

  def gradient(self, grad_ys):
    return [reduce_sum(grad_ys[0], output_shape=self.inputs[0].shape)]

  def lower(self, lowering):
    ret = lowering.mesh_impl(self).broadcast_impl(
        lowering.tensors[self.inputs[0]], self.inputs[0].shape,
        self.outputs[0].shape)
    lowering.set_tensor_lowering(self.outputs[0], ret)


def broadcast(x, new_shape):
  new_shape = convert_to_shape(new_shape)
  if x.shape == new_shape:
    return x
  return BroadcastOperation(x, new_shape).outputs[0]


def _reduce_helper(input_shape,
                   output_shape,
                   input_tensor_layout,
                   reduction_fn_string="SUM"):
  """Returns slicewise function and reduced mesh dimensions.

  Args:
    input_shape: a Shape
    output_shape: a Shape
    input_tensor_layout: a TensorLayout
    reduction_fn_string: "SUM" or "MAX"
  Returns:
    reduce_slice_fn: a function from tf.Tensor to tf.Tensor
    reduced_mesh_axes: a list of integers
  """
  reduce_dims_indices = [
      i for i, d in enumerate(input_shape.dims) if d not in output_shape.dims]
  reduced_input_shape = Shape([
      d for d in input_shape.dims if d in output_shape.dims])
  perm = [reduced_input_shape.dims.index(d) for d in output_shape.dims]
  def reduce_slice_fn(xslice):
    ret = xslice
    if reduce_dims_indices:
      ret = reduction_fn(reduction_fn_string)(xslice, reduce_dims_indices)
    if perm != list(xrange(len(perm))):
      ret = tf.transpose(ret, perm)
    return ret
  reduced_mesh_axes = []
  for i in reduce_dims_indices:
    mesh_axis = input_tensor_layout[i]
    if mesh_axis is not None:
      reduced_mesh_axes.append(mesh_axis)
  return reduce_slice_fn, reduced_mesh_axes


class ReduceOperation(Operation):
  """Reduction - output dims are a subset of input dims, in any order."""

  def __init__(self, x, output_shape, reduction_fn_string, name=None):
    super(ReduceOperation, self).__init__([x], name=name or "reduce")
    self._outputs = [Tensor(self, output_shape, x.dtype)]
    self._reduction_fn_string = reduction_fn_string

  def gradient(self, grad_ys):
    if self._reduction_fn_string == "SUM":
      return [broadcast(grad_ys[0], self.inputs[0].shape)]
    elif (self._reduction_fn_string == "MAX" or
          self._reduction_fn_string == "MIN"):
      return [cast(equal(self.inputs[0], self.outputs[0]), self.inputs[0].dtype)
              * grad_ys[0]]
    else:
      raise ValueError("Gradients to other reductions not implemented")

  def lower(self, lowering):
    mesh_impl = lowering.mesh_impl(self)
    slicewise_fn, reduced_mesh_axes = _reduce_helper(
        self.inputs[0].shape, self.outputs[0].shape,
        mesh_impl.tensor_layout(self.inputs[0]),
        self._reduction_fn_string)
    y = mesh_impl.slicewise(slicewise_fn, lowering.tensors[self.inputs[0]])
    if reduced_mesh_axes:
      def add_counter_fn():
        lowering.add_counter("allreduce/%s/reduce_op" % reduced_mesh_axes,
                             lowering.laid_out_size(self.outputs[0]))
      if self._reduction_fn_string == "SUM":
        y = LazyAllreduceSum(
            mesh_impl, y, reduced_mesh_axes, add_counter_fn=add_counter_fn)
      else:
        y = mesh_impl.allreduce(
            y, reduced_mesh_axes, self._reduction_fn_string)
        add_counter_fn()
    lowering.set_tensor_lowering(self.outputs[0], y)


def _pool_helper(ksize,
                 strides,
                 pool_fn_string="MAX_2D"):
  """Returns slicewise function and reduced mesh dimensions.

  Args:
    ksize: kernel size, a tuple or list.
    strides: a tuple or list.
    pool_fn_string: "MAX" or "AVERAGE"
  Returns:
    pool_slice_fn: a function from tf.Tensor to tf.Tensor
  """
  def pool_slice_fn(xslice):
    ret = pool_fn(pool_fn_string)(xslice, ksize, strides, "VALID")
    return ret
  return pool_slice_fn


def _tf_upscale(x, dim_idx_start, dim_idx_end, xscales):
  """Upscale the tf.Tensor x.

  N-dimensional version of tf.image.resize_images with NEAREST interpolation.
  Similar to: https://github.com/tensorflow/tensorflow/issues/2169

  Args:
    x: a tf.Tensor
    dim_idx_start: the index of starting dimension
    dim_idx_end: the index of ending dimension
    xscales: an integer list of upscaling factors
  Returns:
    a tf Tensor. Dimensions in [dim_idx_start, dim_idx_end - 1] will be upscaled
    xscales[i]-times.
  """

  xscales = list(xscales)
  if dim_idx_start < 0:
    dim_idx_start += len(x.get_shape().as_list())

  def _tf_upscale_one_trailing_dim(x_1tdim):
    """Upscaling with dim_idx_end = -1."""
    x_shape = x_1tdim.get_shape().as_list()
    x_scaled_shape = [ori_size * scale for ori_size, scale \
                      in zip(x_shape[dim_idx_start:-1], xscales)]

    dim_idx_len = len(x_shape[dim_idx_start:-1])
    x_1tdim = tf.reshape(x_1tdim, [-1] + x_shape[-dim_idx_len:])

    for dim_idx in range(dim_idx_len, 0, -1):
      x_1tdim = tf.concat([x_1tdim] * xscales.pop(), dim_idx)
    output_shape = x_shape[:dim_idx_start] + x_scaled_shape + x_shape[-1:]
    x_1tdim = tf.reshape(x_1tdim, output_shape)
    return x_1tdim

  x_shape = x.get_shape().as_list()
  trailing_shape = x_shape[dim_idx_end:]
  x = tf.reshape(x, x_shape[:dim_idx_end] + [-1])
  x = _tf_upscale_one_trailing_dim(x)
  x = tf.reshape(x, x.shape.as_list()[:-1] + trailing_shape)

  return x


class PoolOperation(Operation):
  """Pooling - average or max pool data along HW (2D) or DHW (3D) dimensions.

  For the current implementation of backpropagation, we only handle cases
  when strides == ksize and the input dimensions are divisible by ksize.
  """

  def __init__(self, x, ksize, strides, pool_fn_string, name=None):
    super(PoolOperation, self).__init__([x], name=name or "pool")
    assert ksize == strides
    if "2D" in pool_fn_string:
      assert len(ksize) == 2
    else:
      assert "3D" in pool_fn_string
      assert len(ksize) == 3

    self._ksize = ksize
    self._strides = strides
    self._pool_fn_string = pool_fn_string

    if "2D" in pool_fn_string:
      batch_dims = x.shape.dims[:-3]
      spatial_dims = x.shape.dims[-3:-1]
      channel_dim = x.shape.dims[-1:]
    else:
      batch_dims = x.shape.dims[:-4]
      spatial_dims = x.shape.dims[-4:-1]
      channel_dim = x.shape.dims[-1:]

    # Compute output_shape and allocate output Tensor.
    output_spatial_dims = []
    for spatial_dim, kernel_size, stride_size in zip(
        spatial_dims, ksize, strides):
      output_dim_size = (spatial_dim.size - kernel_size) // stride_size + 1
      output_spatial_dim = Dimension(spatial_dim.name, output_dim_size)
      output_spatial_dims.append(output_spatial_dim)

    output_shape = Shape(batch_dims + output_spatial_dims + channel_dim)
    self._outputs = [Tensor(self, output_shape, x.dtype)]

    # Claim unsplittable dims.
    self._splittable_dims, self._unsplittable_dims = (
        self._initialize_splittable_and_unsplittable_dims(
            "splittable", [dim.name for dim in spatial_dims]))

  def gradient(self, grad_ys):
    """Returns the gradient to input, for unoverlapping pooling."""
    x = self.inputs[0]
    y = self.outputs[0]
    dy = grad_ys[0]
    dx = pool_backprop(x, y, dy,
                       self._ksize, self._strides, self._pool_fn_string)
    return [dx]

  def lower(self, lowering):
    mesh_impl = lowering.mesh_impl(self)
    slicewise_fn = _pool_helper(
        self._ksize, self._strides, self._pool_fn_string)

    x = lowering.tensors[self.inputs[0]]
    y = mesh_impl.slicewise(slicewise_fn, x)

    lowering.set_tensor_lowering(self.outputs[0], y)


class PoolBackPropOperation(Operation):
  """Pooling backpropagation.

  For the current implementation, we only handle cases when
  strides == ksize and the input dimensions are divisible by ksize.
  """

  def __init__(self, x, y, dy,
               ksize, strides, pool_fn_string, name=None):
    super(PoolBackPropOperation, self).__init__(
        [x, y, dy], name=name or "pool_backprop")
    assert ksize == strides
    if "2D" in pool_fn_string:
      assert len(ksize) == 2
    else:
      assert "3D" in pool_fn_string
      assert len(ksize) == 3

    self._ksize = ksize
    self._strides = strides
    self._pool_fn_string = pool_fn_string
    self._outputs = [Tensor(self, x.shape, dy.dtype)]

  def lower(self, lowering):
    """Returns the gradient to input, for unoverlapping pooling."""
    mesh_impl = lowering.mesh_impl(self)

    if self._pool_fn_string == "MAX_2D":
      def slicewise_fn(x, y, dy):
        y_scaled_back = _tf_upscale(y, -3, -1, self._strides)
        dy_scaled_back = _tf_upscale(dy, -3, -1, self._strides)
        return tf.cast(tf.equal(x, y_scaled_back), x.dtype) * dy_scaled_back
    elif self._pool_fn_string == "MAX_3D":
      def slicewise_fn(x, y, dy):
        y_scaled_back = _tf_upscale(y, -4, -1, self._strides)
        dy_scaled_back = _tf_upscale(dy, -4, -1, self._strides)
        return tf.cast(tf.equal(x, y_scaled_back), x.dtype) * dy_scaled_back
    elif self._pool_fn_string == "AVG_2D":
      def slicewise_fn(x, y, dy):
        del y
        dy_scaled_back = _tf_upscale(dy, -3, -1, self._strides)
        return dy_scaled_back / tf.constant(
            self._strides[0] * self._strides[1], dtype=x.dtype)
    elif self._pool_fn_string == "AVG_3D":
      def slicewise_fn(x, y, dy):
        del y
        dy_scaled_back = _tf_upscale(dy, -4, -1, self._strides)
        return dy_scaled_back / tf.constant(
            self._strides[0] * self._strides[1] * self._strides[2],
            dtype=x.dtype)
    else:
      raise ValueError("Pooling %s is not implemented." % self._pool_fn_string)

    dx = mesh_impl.slicewise(
        slicewise_fn, *[lowering.tensors[x] for x in self.inputs])

    lowering.set_tensor_lowering(self.outputs[0], dx)


def pool_backprop(x, y, dy, ksize, strides, pool_fn_string, name=None):
  return PoolBackPropOperation(x, y, dy,
                               ksize, strides, pool_fn_string,
                               name).outputs[0]


class ConcatOperation(Operation):
  """tf.concat.

  All inputs have the same shape, except for the size of the dimension named
  dim_name.
  """

  def __init__(self, xs, concat_dim_name, name=None):
    super(ConcatOperation, self).__init__(xs, name=name or "concat")
    # verify that the shapes are all compatible
    dim_names = [dim.name for dim in xs[0].shape.dims]
    self._concat_dim_name = concat_dim_name

    if concat_dim_name not in dim_names:
      raise ValueError("xs[0] does not contain a dimension named dim_name")
    self._axis = dim_names.index(concat_dim_name)

    should_be_equal = [
        x.shape.resize_dimension(concat_dim_name, 0) for x in xs]
    if not all(s == should_be_equal[0] for s in should_be_equal):
      raise ValueError("shapes are not compatible %s" % xs)

    self._input_sizes = [x.shape.dims[self._axis].size for x in xs]
    output_size = sum(self._input_sizes)
    self._outputs = [
        Tensor(self, xs[0].shape.resize_dimension(concat_dim_name, output_size),
               xs[0].dtype)]

    self._splittable_dims, self._unsplittable_dims = (
        self._initialize_splittable_and_unsplittable_dims(
            "splittable", [concat_dim_name]))

  def gradient(self, grad_ys):
    dy = grad_ys[0]
    return split(dy, self.outputs[0].shape.dims[self._axis], self._input_sizes)

  def lower(self, lowering):
    mesh_impl = lowering.mesh_impl(self)
    if mesh_impl.tensor_dimension_to_mesh_axis(
        Dimension(self._concat_dim_name, 0)) is not None:
      raise ValueError("can't concat along split axis")
    def slicewise_fn(*args):
      return tf.concat(args, axis=self._axis, name="concat")
    y = mesh_impl.slicewise(
        slicewise_fn, *[lowering.tensors[x] for x in self._inputs])
    lowering.set_tensor_lowering(self.outputs[0], y)


def concat(xs, concat_dim_name, name=None):
  """Like tf.concat.

  All inputs must have equal shape except for the sizes in the concatenated
  dimension.  The dimension names should be the same, even that of the
  concatenated dimension.

  Args:
    xs: a list of Tensors
    concat_dim_name: a string
    name: an optional string
  Returns:
    a Tensor
  """
  return ConcatOperation(xs, concat_dim_name, name).outputs[0]


class SplitOperation(Operation):
  """like tf.split.

  TODO(noam, nikip): this code has never been run.  Run it and test it.
  """

  def __init__(self, x, split_dim, num_or_size_splits, name=None):
    super(SplitOperation, self).__init__([x], name=name or "split")

    self._split_dim = split_dim
    if split_dim not in x.shape.dims:
      raise ValueError("%s does not contain dimension %s" % (x, split_dim))
    self._axis = x.shape.dims.index(split_dim)

    if isinstance(num_or_size_splits, list):
      self._output_sizes = num_or_size_splits
      if sum(num_or_size_splits) != split_dim.size:
        raise ValueError(
            "Sizes do not add up %s %s" % (num_or_size_splits, split_dim))
    else:
      assert isinstance(num_or_size_splits, int)
      assert split_dim.size % num_or_size_splits == 0
      self._output_sizes = (
          [split_dim.size // num_or_size_splits] * num_or_size_splits)

    self._outputs = [
        Tensor(self, x.shape.resize_dimension(split_dim.name, output_size),
               x.dtype, index=i)
        for i, output_size in enumerate(self._output_sizes)]

    self._splittable_dims, self._unsplittable_dims = (
        self._initialize_splittable_and_unsplittable_dims(
            "splittable", [split_dim.name]))

  def gradient(self, grad_ys):
    grad_ys = [g or zeros_like(o) for g, o in zip(grad_ys, self._outputs)]
    return [concat(grad_ys, self._split_dim.name)]

  def lower(self, lowering):
    mesh_impl = lowering.mesh_impl(self)
    if mesh_impl.tensor_dimension_to_mesh_axis(self._split_dim) is not None:
      raise ValueError("can't split along split axis")
    def slicewise_fn(x):
      # Since we return a tuple of tf.Tensor, slicewise will collate the
      # outputs and return a tuple of LaidOutTensors.
      return tuple(tf.split(x, self._output_sizes, axis=self._axis))
    values = mesh_impl.slicewise(
        slicewise_fn, lowering.tensors[self.inputs[0]])
    for t, v in zip(self._outputs, values):
      lowering.set_tensor_lowering(t, v)


def split(x, split_dim, num_or_size_splits, name=None):
  """Like tf.split.

  Args:
    x: a Tensor
    split_dim: a Dimension in x.shape.dims
    num_or_size_splits: either an integer dividing split_dim.size
       or a list of integers adding up to split_dim.size
    name: an optional string
  Returns:
    a list of Tensors.
  """
  return SplitOperation(x, split_dim, num_or_size_splits, name=name).outputs


class StackOperation(Operation):
  """Like tf.stack."""

  def __init__(self, xs, dim_name, axis, name=None):
    super(StackOperation, self).__init__(xs, name=name or "stack")
    self._axis = axis
    self._new_dim = Dimension(dim_name, len(xs))
    input_shape = xs[0].shape
    for x in xs:
      if x.shape != xs[0].shape:
        raise ValueError(
            "inputs to stack must have the same shape, got %s" % xs)
    output_shape = Shape(
        input_shape.dims[:axis] + [self._new_dim]+ input_shape.dims[axis:])
    self._outputs = [Tensor(self, output_shape, xs[0].dtype)]

    self._splittable_dims, self._unsplittable_dims = (
        self._initialize_splittable_and_unsplittable_dims(
            "splittable", [dim_name]))

  def gradient(self, grad_ys):
    return unstack(grad_ys[0], self._new_dim)

  def lower(self, lowering):
    mesh_impl = lowering.mesh_impl(self)
    if mesh_impl.tensor_dimension_to_mesh_axis(self._new_dim) is not None:
      raise ValueError("can't stack along split axis")
    inputs = [lowering.tensors[t] for t in self._inputs]
    def slicewise_fn(*args):
      return tf.stack(args, axis=self._axis)
    ret = mesh_impl.slicewise(slicewise_fn, *inputs)
    lowering.set_tensor_lowering(self.outputs[0], ret)


def stack(xs, dim_name, axis=0, name=None):
  """Stack multiple Tensors to make a new dimension.

  Args:
    xs: a list of Tensors with identical shapes.
    dim_name: a string (name of the new dimension)
    axis: an integer (index of the new dimension in the output shape)
    name: an optional string

  Returns:
    a Tensor
  """
  if axis < 0:
    axis = xs[0].shape.ndims + 1 + axis
  ret = StackOperation(xs, dim_name, axis, name).outputs[0]
  return ret


class UnstackOperation(Operation):
  """Split into multiple Tensors, eliminating a dimension."""

  def __init__(self, x, dim, name=None):
    super(UnstackOperation, self).__init__([x], name=name or "unstack")
    self._dim = dim
    self._axis = x.shape.dims.index(dim)
    output_shape = x.shape - dim
    self._outputs = [
        Tensor(self, output_shape, x.dtype, index=i) for i in xrange(dim.size)]

    self._splittable_dims, self._unsplittable_dims = (
        self._initialize_splittable_and_unsplittable_dims(
            "splittable", [dim.name]))

  def gradient(self, grad_ys):
    return [stack(grad_ys, self._dim.name, self._axis)]

  def lower(self, lowering):
    mesh_impl = lowering.mesh_impl(self)
    if mesh_impl.tensor_dimension_to_mesh_axis(self._dim) is not None:
      raise ValueError("can't unstack along split axis")
    def slicewise_fn(x):
      return tuple(tf.unstack(x, num=self._dim.size, axis=self._axis))
    output_values = mesh_impl.slicewise(
        slicewise_fn, lowering.tensors[self._inputs[0]])
    for t, v in zip(self.outputs, list(output_values)):
      lowering.set_tensor_lowering(t, v)


def unstack(x, dim, name=None):
  """Split into multiple Tensors, eliminating a dimension.

  Args:
    x: a Tensor
    dim: a Dimension
    name: an optional string

  Returns:
    a list of dim.size Tensors, each with shape (x.shape - dim)
  """
  return UnstackOperation(x, dim, name).outputs


def cumsum(x, dim, exclusive=False):
  """Cumulative sum.

  Args:
    x: a Tensor
    dim: a Dimension
    exclusive: a boolean

  Returns:
    a Tensor with the same shape as x.
  """
  with tf.variable_scope("cumsum"):
    new_name = "tmp_dim_cumsum"
    new_dim = Dimension(new_name, dim.size)
    new_shape = x.shape.rename_dimension(dim.name, new_name)
    comparator = less if exclusive else less_equal
    m = cast(
        comparator(mtf_range(x.mesh, dim, dtype=tf.float32),
                   mtf_range(x.mesh, new_dim, dtype=tf.float32)), x.dtype)
    ret = einsum([x, m], output_shape=new_shape)
    return reshape(ret, x.shape)


def _einsum_helper(input_shapes, output_shape, mesh_impl):
  """Returns slicewise function and reduced mesh dimensions.

  Assumes the output shape contains no new dimensions.

  Args:
    input_shapes: a list of Shapes
    output_shape: a Shape
    mesh_impl: a MeshImpl
  Returns:
    einsum_slice_fn: a function from tf.Tensors to tf.Tensor
    reduced_mesh_axes: a list of integers
  """
  input_shape_union = _shape_union(input_shapes)
  total_num_dims = input_shape_union.ndims
  # list of input shapes that contain all dimensions.
  full_shapes = [
      s for s in input_shapes + [output_shape] if s.ndims == total_num_dims]
  full_shape = full_shapes[0] if full_shapes else input_shape_union
  reduce_slice_fn, reduced_mesh_axes = _reduce_helper(
      full_shape, output_shape, mesh_impl.tensor_layout(full_shape))
  def einsum_slice_fn_naive(*slices):
    # naive einsum implementation where we broadcast all inputs to the full
    # shape, multiply componentwise, then reduce.
    return reduce_slice_fn(functools.reduce(tf.multiply, [
        _expand_dims(x, input_shape, full_shape)
        for x, input_shape in zip(slices, input_shapes)]))
  if full_shapes:
    # it is not wasteful of space to broadcast fully and then reduce.
    # this helps to avoid some inefficient GPU implementations.
    einsum_slice_fn = einsum_slice_fn_naive
  else:
    # call tf.einsum
    equation = _einsum_equation(input_shapes, output_shape)
    def einsum_slice_fn(*slices):
      if slices[0].dtype.is_floating:
        return mesh_impl.einsum(equation, *slices)
      else:
        return einsum_slice_fn_naive(*slices)
  return einsum_slice_fn, reduced_mesh_axes


class EinsumOperation(Operation):
  """Einstein summation (matmul, etc).

  The equation follows the dimensions in the input and output shapes.

  Every dimension must occur in at least two of the input/output Tensors.
  i.e. no new dimensions in the output, and no reduction of dimensions that
  occur in only one input.
  """

  def __init__(self, inputs, output_shape, name=None):
    super(EinsumOperation, self).__init__(inputs, name=name or "einsum")
    if not inputs:
      raise ValueError("Einsum needs at least one input")
    for x in inputs:
      if x.dtype != inputs[0].dtype:
        raise ValueError("Input dtypes must be equal got %s"
                         % ([y.dtype for y in inputs],))
    self._outputs = [Tensor(self, output_shape, inputs[0].dtype)]

  def gradient(self, grad_ys):
    dy = grad_ys[0]
    xs = self.inputs
    ret = []
    for i in xrange(len(self.inputs)):
      ret.append(
          einsum([dy] + [xs[j] for j in xrange(len(xs)) if j != i], xs[i].shape)
      )
    return ret

  def lower(self, lowering):
    mesh_impl = lowering.mesh_impl(self)
    xs = self.inputs
    input_shape_set = set(sum([x.shape.dims for x in xs], []))
    output_shape = self.outputs[0].shape
    intersection_shape = Shape(
        [d for d in output_shape.dims if d in input_shape_set])
    einsum_slice_fn, reduced_mesh_axes = _einsum_helper(
        [x.shape for x in self.inputs], intersection_shape, mesh_impl)
    y = mesh_impl.slicewise(
        einsum_slice_fn, *[lowering.tensors[x] for x in self.inputs])
    if reduced_mesh_axes:
      def add_counter_fn():
        lowering.add_counter(
            "allreduce/%s/einsum_op" % reduced_mesh_axes,
            mesh_impl.laid_out_size(intersection_shape))
      y = LazyAllreduceSum(
          mesh_impl, y, reduced_mesh_axes, add_counter_fn=add_counter_fn)
    # broadcast from intersection_shape to output_shape
    if intersection_shape != output_shape:
      y = mesh_impl.broadcast_impl(y, intersection_shape, output_shape)
    lowering.set_tensor_lowering(self.outputs[0], y)
    computation_shape = Shape(list(input_shape_set))
    lowering.add_counter("einsum", mesh_impl.laid_out_size(computation_shape))
    lowering.add_counter("einsum_unique", computation_shape.size)


class Conv2dOperation(Operation):
  """like tf.nn.conv2d.

  Always data format "NHWC".
  # TODO(nikip): support dilations
  Always dilation rate of 1
  padding: "SAME" or "VALID"

  TODO(noam): implement more options.
  """

  def __init__(self, conv_input, conv_filter, strides, padding, name=None):
    super(Conv2dOperation, self).__init__(
        [conv_input, conv_filter], name=name or "conv2d")
    self._padding = padding
    self._batch_dims = conv_input.shape.dims[:-3]
    self._in_h_dim, self._in_w_dim, self._in_dim = conv_input.shape.dims[-3:]
    self._fh_dim, self._fw_dim = conv_filter.shape.dims[:2]
    f_in_dim, self._out_dim = conv_filter.shape.dims[2:]
    if f_in_dim != self._in_dim:
      raise ValueError("Dimensions do not match input=%s filter=%s"
                       % (conv_input, conv_filter))
    out_h = self._in_h_dim.size
    out_w = self._in_w_dim.size
    if padding == "VALID":
      out_h -= (self._fh_dim.size - 1)
      out_w -= (self._fw_dim.size - 1)

    self._strides = strides
    if strides is not None:
      out_h //= strides[1]
      out_w //= strides[2]
    self._out_h_dim = Dimension(self._in_h_dim.name, out_h)
    self._out_w_dim = Dimension(self._in_w_dim.name, out_w)
    output_shape = Shape(
        self._batch_dims + [self._out_h_dim, self._out_w_dim, self._out_dim])
    self._outputs = [Tensor(self, output_shape, conv_input.dtype)]

    unsplittable_dims = [self._in_h_dim, self._in_w_dim, self._fh_dim,
                         self._fw_dim]
    self._splittable_dims, self._unsplittable_dims = (
        self._initialize_splittable_and_unsplittable_dims(
            "splittable", [dim.name for dim in unsplittable_dims]))

  def gradient(self, grad_ys):
    dy = grad_ys[0]
    conv_input, conv_filter = self.inputs
    return [
        conv2d_backprop_input(self._inputs[0].shape,
                              conv_filter,
                              dy,
                              self._strides,
                              self._padding),
        conv2d_backprop_filter(conv_input,
                               self._inputs[1].shape,
                               dy,
                               self._strides,
                               self._padding)]

  def lower(self, lowering):
    mesh_impl = lowering.mesh_impl(self)
    conv_input, conv_filter = self.inputs
    if mesh_impl.tensor_dimension_to_mesh_axis(self._in_h_dim) is not None:
      raise ValueError("can't slice along dimension h")
    if mesh_impl.tensor_dimension_to_mesh_axis(self._in_w_dim) is not None:
      raise ValueError("can't slice along dimension w")
    if mesh_impl.tensor_dimension_to_mesh_axis(self._fh_dim) is not None:
      raise ValueError("can't slice along dimension fh")
    if mesh_impl.tensor_dimension_to_mesh_axis(self._fw_dim) is not None:
      raise ValueError("can't slice along dimension fw")
    def tf_fn(tf_input, tf_filter):
      output = tf.nn.conv2d(
          _tf_flatten_batch_dims(tf_input, 3),
          tf_filter, self._strides, self._padding)
      return _tf_restore_batch_dims(output, 3, tf_input)
    y = mesh_impl.slicewise(
        tf_fn, lowering.tensors[conv_input], lowering.tensors[conv_filter])
    # reducing out input channels - may need to allreduce
    in_mesh_axis = mesh_impl.tensor_dimension_to_mesh_axis(self._in_dim)
    if in_mesh_axis is not None:
      def add_counter_fn():
        lowering.add_counter(
            "allreduce/%s/conv2d_op" % [in_mesh_axis],
            mesh_impl.laid_out_size(self.outputs[0].shape))
      y = LazyAllreduceSum(mesh_impl, y, [in_mesh_axis], add_counter_fn)
    lowering.set_tensor_lowering(self.outputs[0], y)
    computation_shape = _shape_union([conv_filter.shape, self.outputs[0].shape])
    lowering.add_counter("conv2d/forward",
                         mesh_impl.laid_out_size(computation_shape))
    lowering.add_counter("conv2d_unique/forward", computation_shape.size)


class Conv2or3dBackpropInputOperation(Operation):
  """like tf.nn.conv2d/conv3d_backprop_input."""

  def __init__(self, conv_dimension, is_transpose,
               input_shape, conv_filter, dy, strides, padding, name=None):
    assert conv_dimension in [2, 3]
    self._trans = "_trans" if is_transpose else ""
    default_name = "conv%dd%s_backprop" % (conv_dimension, self._trans)
    super(Conv2or3dBackpropInputOperation, self).__init__(
        [dy, conv_filter], name=name or default_name)

    self._conv_dimension = conv_dimension
    self._is_transpose = is_transpose
    self._padding = padding
    self._strides = strides
    self._input_shape = input_shape
    self._outputs = [Tensor(self, input_shape, dy.dtype)]
    self._num_nonbatch_dims = conv_dimension + 1

    # Rerun to take the new output into account.
    self._splittable_dims, self._unsplittable_dims = (
        self._initialize_all_dimensions_as_splittable())

  def lower(self, lowering):
    mesh_impl = lowering.mesh_impl(self)
    dy, conv_filter = self.inputs
    input_sizes = mesh_impl.slice_shape(self.outputs[0].shape)
    input_sizes = [list_product(input_sizes[:-self._num_nonbatch_dims])] + (
        input_sizes[-self._num_nonbatch_dims:])

    if self._is_transpose:
      if self._conv_dimension == 2:
        backprop_fn = tf.nn.conv2d
      else:
        backprop_fn = tf.nn.conv3d
      def tf_fn(tf_dy, tf_filter):
        return _tf_restore_batch_dims(
            backprop_fn(
                _tf_flatten_batch_dims(tf_dy, self._num_nonbatch_dims),
                tf_filter,
                self._strides, self._padding),
            self._num_nonbatch_dims, tf_dy)
      dx = mesh_impl.slicewise(
          tf_fn, lowering.tensors[dy], lowering.tensors[conv_filter])

    else:  # if not self._is_transpose:
      if self._conv_dimension == 2:
        backprop_fn = tf.nn.conv2d_backprop_input
      else:
        backprop_fn = conv3d_backprop_input_v2
      def tf_fn(tf_dy, tf_filter):
        return _tf_restore_batch_dims(
            backprop_fn(
                input_sizes, tf_filter,
                _tf_flatten_batch_dims(tf_dy, self._num_nonbatch_dims),
                self._strides, self._padding),
            self._num_nonbatch_dims, tf_dy)
      dx = mesh_impl.slicewise(
          tf_fn, lowering.tensors[dy], lowering.tensors[conv_filter])

    # reducing out output channels - may need to allreduce
    out_mesh_axis = mesh_impl.tensor_dimension_to_mesh_axis(dy.shape.dims[-1])
    if out_mesh_axis is not None:
      def add_counter_fn():
        lowering.add_counter(
            "allreduce/%s/conv%dd%s_op" % (
                [out_mesh_axis], self._conv_dimension, self._trans),
            mesh_impl.laid_out_size(self.outputs[0].shape))
      dx = LazyAllreduceSum(mesh_impl, dx, [out_mesh_axis], add_counter_fn)
    lowering.set_tensor_lowering(self.outputs[0], dx)
    computation_shape = _shape_union([conv_filter.shape, dy.shape])
    lowering.add_counter(
        "conv%dd%s/backprop_input" % (self._conv_dimension, self._trans),
        mesh_impl.laid_out_size(computation_shape))
    lowering.add_counter(
        "conv%dd%s_unique/backprop_input" % (self._conv_dimension, self._trans),
        computation_shape.size)


def conv2d_backprop_input(input_shape,
                          conv_filter,
                          dy,
                          strides,
                          padding, name=None):
  return Conv2or3dBackpropInputOperation(2, False,
                                         input_shape,
                                         conv_filter,
                                         dy,
                                         strides,
                                         padding,
                                         name=name).outputs[0]


class Conv2or3dBackpropFilterOperation(Operation):
  """Like tf.nn.conv2d_backprop_filter."""

  def __init__(self, conv_dimension, is_transpose,
               conv_input, filter_shape, dy, strides, padding, name=None):
    assert conv_dimension in [2, 3]
    self._trans = "_trans" if is_transpose else ""
    default_name = "conv%dd%s_backprop_filter" % (conv_dimension, self._trans)
    super(Conv2or3dBackpropFilterOperation, self).__init__(
        [conv_input, dy], name=name or default_name)

    self._conv_dimension = conv_dimension
    self._is_transpose = is_transpose
    self._padding = padding
    self._strides = strides
    self._filter_shape = filter_shape
    self._outputs = [Tensor(self, filter_shape, dy.dtype)]
    self._num_nonbatch_dims = conv_dimension + 1

    # Rerun to take the new output into account.
    self._splittable_dims, self._unsplittable_dims = (
        self._initialize_all_dimensions_as_splittable())

  def lower(self, lowering):
    mesh_impl = lowering.mesh_impl(self)
    conv_input, dy = self.inputs
    filter_sizes = mesh_impl.slice_shape(self.outputs[0].shape)

    if self._conv_dimension == 2:
      backprop_fn = tf.nn.conv2d_backprop_filter
    else:
      backprop_fn = conv3d_backprop_filter_v2

    def tf_fn(tf_input, tf_dy):
      if self._is_transpose:
        y, x = tf_input, tf_dy
      else:
        x, y = tf_input, tf_dy
      return backprop_fn(
          _tf_flatten_batch_dims(x, self._num_nonbatch_dims),
          filter_sizes,
          _tf_flatten_batch_dims(y, self._num_nonbatch_dims),
          self._strides,
          self._padding)

    df = mesh_impl.slicewise(
        tf_fn, lowering.tensors[conv_input], lowering.tensors[dy])

    # reducing out batch dimensions - may need to allreduce
    reduced_mesh_axes = [
        mesh_impl.tensor_dimension_to_mesh_axis(d)
        for d in dy.shape.dims[:-self._num_nonbatch_dims]]
    reduced_mesh_axes = [a for a in reduced_mesh_axes if a is not None]

    if reduced_mesh_axes:
      def add_counter_fn():
        lowering.add_counter(
            "allreduce/%s/conv%dd%s_backprop_filter" % (
                reduced_mesh_axes, self._conv_dimension, self._trans),
            mesh_impl.laid_out_size(self.outputs[0].shape))
      df = LazyAllreduceSum(mesh_impl, df, reduced_mesh_axes, add_counter_fn)

    lowering.set_tensor_lowering(self.outputs[0], df)
    computation_shape = _shape_union([self.outputs[0].shape, dy.shape])
    lowering.add_counter("conv%dd%s/backprop_filter" % (self._conv_dimension,
                                                        self._trans),
                         mesh_impl.laid_out_size(computation_shape))
    lowering.add_counter(
        "conv%dd%s_unique/backprop_filter" % (self._conv_dimension,
                                              self._trans),
        computation_shape.size)


def conv2d_backprop_filter(conv_input,
                           filter_shape,
                           dy,
                           strides,
                           padding, name=None):
  return Conv2or3dBackpropFilterOperation(2, False,
                                          conv_input,
                                          filter_shape,
                                          dy,
                                          strides,
                                          padding,
                                          name=name).outputs[0]


class Conv3dOperation(Operation):
  """like tf.nn.conv3d.

  Currently we assume that the data format is always "NDHWC".
  # TODO(lehou): support more options such as dilation.
  Always dilation rate of 1
  padding: "SAME" or "VALID"
  """

  def __init__(self, conv_input, conv_filter, strides, padding, name=None):
    super(Conv3dOperation, self).__init__(
        [conv_input, conv_filter], name=name or "conv3d")
    self._padding = padding
    self._batch_dims = conv_input.shape.dims[:-4]
    self._in_d_dim, self._in_h_dim, self._in_w_dim, self._in_dim = (
        conv_input.shape.dims[-4:])
    self._fd_dim, self._fh_dim, self._fw_dim = conv_filter.shape.dims[:3]
    f_in_dim, self._out_dim = conv_filter.shape.dims[3:]
    if f_in_dim != self._in_dim:
      raise ValueError("Dimensions do not match input=%s filter=%s"
                       % (conv_input, conv_filter))
    out_d = self._in_d_dim.size
    out_h = self._in_h_dim.size
    out_w = self._in_w_dim.size
    if padding == "VALID":
      out_d -= (self._fd_dim.size - 1)
      out_h -= (self._fh_dim.size - 1)
      out_w -= (self._fw_dim.size - 1)

    self._strides = strides
    if strides is not None:
      out_d //= strides[1]
      out_h //= strides[2]
      out_w //= strides[3]
    self._out_d_dim = Dimension(self._in_d_dim.name, out_d)
    self._out_h_dim = Dimension(self._in_h_dim.name, out_h)
    self._out_w_dim = Dimension(self._in_w_dim.name, out_w)
    output_shape = Shape(
        self._batch_dims + [self._out_d_dim, self._out_h_dim,
                            self._out_w_dim, self._out_dim])
    self._outputs = [Tensor(self, output_shape, conv_input.dtype)]

    unsplittable_dims = [self._in_d_dim, self._in_h_dim, self._in_w_dim,
                         self._fd_dim, self._fh_dim, self._fw_dim]
    self._splittable_dims, self._unsplittable_dims = (
        self._initialize_splittable_and_unsplittable_dims(
            "splittable", [dim.name for dim in unsplittable_dims]))

  def gradient(self, grad_ys):
    dy = grad_ys[0]
    conv_input, conv_filter = self.inputs
    return [
        conv3d_backprop_input(self._inputs[0].shape,
                              conv_filter,
                              dy,
                              self._strides,
                              self._padding),
        conv3d_backprop_filter(conv_input,
                               self._inputs[1].shape,
                               dy,
                               self._strides,
                               self._padding)]

  def lower(self, lowering):
    mesh_impl = lowering.mesh_impl(self)
    conv_input, conv_filter = self.inputs
    if mesh_impl.tensor_dimension_to_mesh_axis(self._in_d_dim) is not None:
      raise ValueError("can't slice along dimension d")
    if mesh_impl.tensor_dimension_to_mesh_axis(self._in_h_dim) is not None:
      raise ValueError("can't slice along dimension h")
    if mesh_impl.tensor_dimension_to_mesh_axis(self._in_w_dim) is not None:
      raise ValueError("can't slice along dimension w")
    if mesh_impl.tensor_dimension_to_mesh_axis(self._fd_dim) is not None:
      raise ValueError("can't slice along dimension fd")
    if mesh_impl.tensor_dimension_to_mesh_axis(self._fh_dim) is not None:
      raise ValueError("can't slice along dimension fh")
    if mesh_impl.tensor_dimension_to_mesh_axis(self._fw_dim) is not None:
      raise ValueError("can't slice along dimension fw")
    def tf_fn(tf_input, tf_filter):
      output = tf.nn.conv3d(
          _tf_flatten_batch_dims(tf_input, 4),
          tf_filter, self._strides, self._padding)
      return _tf_restore_batch_dims(output, 4, tf_input)
    y = mesh_impl.slicewise(
        tf_fn, lowering.tensors[conv_input], lowering.tensors[conv_filter])
    # reducing out input channels - may need to allreduce
    in_mesh_axis = mesh_impl.tensor_dimension_to_mesh_axis(self._in_dim)
    if in_mesh_axis is not None:
      def add_counter_fn():
        lowering.add_counter(
            "allreduce/%s/conv3d_op" % [in_mesh_axis],
            mesh_impl.laid_out_size(self.outputs[0].shape))
      y = LazyAllreduceSum(mesh_impl, y, [in_mesh_axis], add_counter_fn)
    lowering.set_tensor_lowering(self.outputs[0], y)
    computation_shape = _shape_union([conv_filter.shape, self.outputs[0].shape])
    lowering.add_counter("conv3d/forward",
                         mesh_impl.laid_out_size(computation_shape))
    lowering.add_counter("conv3d_unique/forward", computation_shape.size)


def conv3d_backprop_input(input_shape,
                          conv_filter,
                          dy,
                          strides,
                          padding, name=None):
  return Conv2or3dBackpropInputOperation(3, False,
                                         input_shape,
                                         conv_filter,
                                         dy,
                                         strides,
                                         padding,
                                         name=name).outputs[0]


def conv3d_backprop_filter(conv_input,
                           filter_shape,
                           dy,
                           strides,
                           padding, name=None):
  return Conv2or3dBackpropFilterOperation(3, False,
                                          conv_input,
                                          filter_shape,
                                          dy,
                                          strides,
                                          padding,
                                          name=name).outputs[0]


class Conv2dTransposeOperation(Operation):
  """like tf.nn.conv2d_transpose.

  Currently we assume that the data format is always "NHWC".
  # TODO(lehou): support more options such as dilation.
  Always dilation rate of 1
  padding: "SAME" or "VALID"
  """

  def __init__(self, conv_input, conv_filter, strides, padding, name=None):
    super(Conv2dTransposeOperation, self).__init__(
        [conv_input, conv_filter], name=name or "conv2d_transpose")
    self._padding = padding
    self._batch_dims = conv_input.shape.dims[:-3]
    self._in_h_dim, self._in_w_dim, self._in_dim = conv_input.shape.dims[-3:]
    self._fh_dim, self._fw_dim = conv_filter.shape.dims[:2]

    # Filter shape is transposed.
    self._out_dim, f_in_dim = conv_filter.shape.dims[2:]
    if f_in_dim != self._in_dim:
      raise ValueError("Dimensions do not match input=%s filter=%s"
                       % (conv_input, conv_filter))

    # compute output shape.
    # now we assume the padding doesn't change the output shape.
    # TODO(lehou): work out the output shape in general cases.
    out_h = self._in_h_dim.size
    out_w = self._in_w_dim.size
    self._strides = strides
    if strides is not None:
      out_h *= strides[1]
      out_w *= strides[2]

    # name output shape.
    self._out_h_dim = Dimension(self._in_h_dim.name, out_h)
    self._out_w_dim = Dimension(self._in_w_dim.name, out_w)
    output_shape = Shape(self._batch_dims + [
        self._out_h_dim, self._out_w_dim, self._out_dim])
    self._outputs = [Tensor(self, output_shape, conv_input.dtype)]

    unsplittable_dims = [self._in_h_dim, self._in_w_dim,
                         self._fh_dim, self._fw_dim]
    self._splittable_dims, self._unsplittable_dims = (
        self._initialize_splittable_and_unsplittable_dims(
            "splittable", [dim.name for dim in unsplittable_dims]))

  def gradient(self, grad_ys):
    dy = grad_ys[0]
    conv_input, conv_filter = self.inputs
    return [
        conv2d_transpose_backprop_input(self._inputs[0].shape,
                                        conv_filter,
                                        dy,
                                        self._strides,
                                        self._padding),
        conv2d_transpose_backprop_filter(conv_input,
                                         self._inputs[1].shape,
                                         dy,
                                         self._strides,
                                         self._padding)]

  def lower(self, lowering):
    mesh_impl = lowering.mesh_impl(self)
    conv_input, conv_filter = self.inputs
    if mesh_impl.tensor_dimension_to_mesh_axis(self._in_h_dim) is not None:
      raise ValueError("can't slice along dimension h")
    if mesh_impl.tensor_dimension_to_mesh_axis(self._in_w_dim) is not None:
      raise ValueError("can't slice along dimension w")
    if mesh_impl.tensor_dimension_to_mesh_axis(self._fh_dim) is not None:
      raise ValueError("can't slice along dimension fh")
    if mesh_impl.tensor_dimension_to_mesh_axis(self._fw_dim) is not None:
      raise ValueError("can't slice along dimension fw")

    # run conv2d_transpose in each slice.
    def tf_fn(tf_input, tf_filter):
      """conv2d_transpose in tensorflow."""
      # Get the output shape.
      # Here, we compute flattened batch size from tf_input, since there can be
      # split along batch dimensions.
      flattened_batch_size = 1
      for dim in tf_input.shape[:-3]:
        flattened_batch_size *= dim
      flattened_output_shape = [
          flattened_batch_size, self._out_h_dim.size,
          self._out_w_dim.size, self._out_dim.size]

      output = tf.nn.conv2d_backprop_input(
          flattened_output_shape, tf_filter,
          _tf_flatten_batch_dims(tf_input, 3),
          self._strides, self._padding)
      return _tf_restore_batch_dims(output, 3, tf_input)

    y = mesh_impl.slicewise(
        tf_fn, lowering.tensors[conv_input], lowering.tensors[conv_filter])

    # reducing out input channels - may need to allreduce
    in_mesh_axis = mesh_impl.tensor_dimension_to_mesh_axis(self._in_dim)
    if in_mesh_axis is not None:
      def add_counter_fn():
        lowering.add_counter(
            "allreduce/%s/conv2d_transpose_op" % [in_mesh_axis],
            mesh_impl.laid_out_size(self.outputs[0].shape))
      y = LazyAllreduceSum(mesh_impl, y, [in_mesh_axis], add_counter_fn)
    lowering.set_tensor_lowering(self.outputs[0], y)
    computation_shape = _shape_union([conv_filter.shape, self.outputs[0].shape])
    lowering.add_counter("conv2d_transpose/forward",
                         mesh_impl.laid_out_size(computation_shape))
    lowering.add_counter("conv2d_transpose_unique/forward",
                         computation_shape.size)


def conv2d_transpose_backprop_input(input_shape,
                                    conv_filter,
                                    dy,
                                    strides,
                                    padding, name=None):
  return Conv2or3dBackpropInputOperation(2, True,
                                         input_shape,
                                         conv_filter,
                                         dy,
                                         strides,
                                         padding,
                                         name=name).outputs[0]


def conv2d_transpose_backprop_filter(conv_input,
                                     filter_shape,
                                     dy,
                                     strides,
                                     padding, name=None):
  return Conv2or3dBackpropFilterOperation(2, True,
                                          conv_input,
                                          filter_shape,
                                          dy,
                                          strides,
                                          padding,
                                          name=name).outputs[0]


class Conv3dTransposeOperation(Operation):
  """like tf.nn.conv3d_transpose.

  Currently we assume that the data format is always "NDHWC".
  # TODO(lehou): support more options such as dilation.
  Always dilation rate of 1
  padding: "SAME" or "VALID"
  """

  def __init__(self, conv_input, conv_filter, strides, padding, name=None):
    super(Conv3dTransposeOperation, self).__init__(
        [conv_input, conv_filter], name=name or "conv3d_transpose")
    self._padding = padding
    self._batch_dims = conv_input.shape.dims[:-4]
    self._in_d_dim, self._in_h_dim, self._in_w_dim, self._in_dim = (
        conv_input.shape.dims[-4:])
    self._fd_dim, self._fh_dim, self._fw_dim = conv_filter.shape.dims[:3]

    # Filter shape is transposed.
    self._out_dim, f_in_dim = conv_filter.shape.dims[3:]
    if f_in_dim != self._in_dim:
      raise ValueError("Dimensions do not match input=%s filter=%s"
                       % (conv_input, conv_filter))

    # compute output shape.
    # now we assume the padding doesn't change the output shape.
    # TODO(lehou): work out the output shape in general cases.
    out_d = self._in_d_dim.size
    out_h = self._in_h_dim.size
    out_w = self._in_w_dim.size
    self._strides = strides
    if strides is not None:
      out_d *= strides[1]
      out_h *= strides[2]
      out_w *= strides[3]

    # name output shape.
    self._out_d_dim = Dimension(self._in_d_dim.name, out_d)
    self._out_h_dim = Dimension(self._in_h_dim.name, out_h)
    self._out_w_dim = Dimension(self._in_w_dim.name, out_w)
    output_shape = Shape(self._batch_dims + [self._out_d_dim, self._out_h_dim,
                                             self._out_w_dim, self._out_dim])
    self._outputs = [Tensor(self, output_shape, conv_input.dtype)]

    unsplittable_dims = [self._in_d_dim, self._in_h_dim, self._in_w_dim,
                         self._fd_dim, self._fh_dim, self._fw_dim]
    self._splittable_dims, self._unsplittable_dims = (
        self._initialize_splittable_and_unsplittable_dims(
            "splittable", [dim.name for dim in unsplittable_dims]))

  def gradient(self, grad_ys):
    dy = grad_ys[0]
    conv_input, conv_filter = self.inputs
    return [
        conv3d_transpose_backprop_input(self._inputs[0].shape,
                                        conv_filter,
                                        dy,
                                        self._strides,
                                        self._padding),
        conv3d_transpose_backprop_filter(conv_input,
                                         self._inputs[1].shape,
                                         dy,
                                         self._strides,
                                         self._padding)]

  def lower(self, lowering):
    mesh_impl = lowering.mesh_impl(self)
    conv_input, conv_filter = self.inputs
    if mesh_impl.tensor_dimension_to_mesh_axis(self._in_d_dim) is not None:
      raise ValueError("can't slice along dimension d")
    if mesh_impl.tensor_dimension_to_mesh_axis(self._in_h_dim) is not None:
      raise ValueError("can't slice along dimension h")
    if mesh_impl.tensor_dimension_to_mesh_axis(self._in_w_dim) is not None:
      raise ValueError("can't slice along dimension w")
    if mesh_impl.tensor_dimension_to_mesh_axis(self._fd_dim) is not None:
      raise ValueError("can't slice along dimension fd")
    if mesh_impl.tensor_dimension_to_mesh_axis(self._fh_dim) is not None:
      raise ValueError("can't slice along dimension fh")
    if mesh_impl.tensor_dimension_to_mesh_axis(self._fw_dim) is not None:
      raise ValueError("can't slice along dimension fw")

    # run conv3d_transpose in each slice.
    def tf_fn(tf_input, tf_filter):
      """conv3d_transpose in tensorflow."""
      # Get the output shape.
      # Here, we compute flattened batch size from tf_input, since there can be
      # split along batch dimensions.
      flattened_batch_size = 1
      for dim in tf_input.shape[:-4]:
        flattened_batch_size *= dim
      flattened_output_shape = [flattened_batch_size,
                                self._out_d_dim.size, self._out_h_dim.size,
                                self._out_w_dim.size, self._out_dim.size]

      output = conv3d_backprop_input_v2(
          flattened_output_shape, tf_filter,
          _tf_flatten_batch_dims(tf_input, 4),
          self._strides, self._padding)
      return _tf_restore_batch_dims(output, 4, tf_input)

    y = mesh_impl.slicewise(
        tf_fn, lowering.tensors[conv_input], lowering.tensors[conv_filter])

    # reducing out input channels - may need to allreduce
    in_mesh_axis = mesh_impl.tensor_dimension_to_mesh_axis(self._in_dim)
    if in_mesh_axis is not None:
      def add_counter_fn():
        lowering.add_counter(
            "allreduce/%s/conv3d_transpose_op" % [in_mesh_axis],
            mesh_impl.laid_out_size(self.outputs[0].shape))
      y = LazyAllreduceSum(mesh_impl, y, [in_mesh_axis], add_counter_fn)
    lowering.set_tensor_lowering(self.outputs[0], y)
    computation_shape = _shape_union([conv_filter.shape, self.outputs[0].shape])
    lowering.add_counter("conv3d_transpose/forward",
                         mesh_impl.laid_out_size(computation_shape))
    lowering.add_counter("conv3d_transpose_unique/forward",
                         computation_shape.size)


def conv3d_transpose_backprop_input(input_shape,
                                    conv_filter,
                                    dy,
                                    strides,
                                    padding, name=None):
  return Conv2or3dBackpropInputOperation(3, True,
                                         input_shape,
                                         conv_filter,
                                         dy,
                                         strides,
                                         padding,
                                         name=name).outputs[0]


def conv3d_transpose_backprop_filter(conv_input,
                                     filter_shape,
                                     dy,
                                     strides,
                                     padding, name=None):
  return Conv2or3dBackpropFilterOperation(3, True,
                                          conv_input,
                                          filter_shape,
                                          dy,
                                          strides,
                                          padding,
                                          name=name).outputs[0]


class ShiftOperation(Operation):
  """Shift by a static offset in one dimension."""

  def __init__(self, x, offset, dim, wrap, name=None):
    """Create a shift operation.

    Shift x right by +offset in dimension dim.
    If offset is negative, shift left.
    If wrap is true then wrap-around.  Else, pad with zeros.

    Args:
      x: a Tensor
      offset: an integer
      dim: a Dimension of x
      wrap: a boolean - whether to wrap or pad.
      name: an optional string
    """
    super(ShiftOperation, self).__init__([x], name=name or "shift")
    self._dim = dim
    self._axis = x.shape.dims.index(dim)
    self._offset = offset
    self._wrap = wrap
    self._outputs = [Tensor(self, x.shape, x.dtype)]

  def gradient(self, grad_ys):
    return [shift(grad_ys[0], -self._offset, self._dim, self._wrap)]

  def lower(self, lowering):
    mesh_impl = lowering.mesh_impl(self)
    mesh_axis = mesh_impl.tensor_dimension_to_mesh_axis(self._dim)
    inputs = self._inputs[0]
    ndims = self._inputs[0].shape.ndims
    axis = self._axis
    dim = self._dim
    lowered_x = lowering.tensors[inputs]
    if not self._wrap and abs(self._offset) >= dim.size:
      lowering.set_tensor_lowering(
          self.outputs[0],
          mesh_impl.slicewise(tf.zeros_like, lowered_x))
      return
    def my_slice(x, start, size):
      assert size >= 0
      begin = [0] * axis + [start] + [0] * (ndims - axis - 1)
      size = [-1] * axis + [size] + [-1] * (ndims - axis - 1)
      return tf.slice(x, begin, size)
    if mesh_axis is None:
      def slicewise_fn(x):
        """Slicewise function."""
        def my_pad(s, begin_pad, end_pad):
          paddings = ([[0, 0]] * axis + [[begin_pad, end_pad]]
                      + [[0, 0]] * (ndims - axis - 1))
          return tf.pad(s, paddings)
        if self._wrap:
          offset = self._offset % dim.size
          return tf.concat([my_slice(x, dim.size - offset, offset),
                            my_slice(x, 0, dim.size - offset)], axis=axis)
        elif self._offset > 0:
          return my_pad(
              my_slice(x, 0, dim.size - self._offset), self._offset, 0)
        else:
          neg_offset = -self._offset
          return my_pad(
              my_slice(x, neg_offset, dim.size - neg_offset), 0, neg_offset)
      lowered_y = mesh_impl.slicewise(slicewise_fn, lowered_x)
    else:
      mesh_dim_size = mesh_impl.shape.dims[mesh_axis].size
      tensor_dim_size = self._dim.size
      block_size = tensor_dim_size // mesh_dim_size
      odiv = self._offset // block_size
      omod = self._offset % block_size
      laid_out_size = mesh_impl.laid_out_size(inputs.shape)
      if omod == 0:
        # shift by an integral number of processors.
        lowered_y = mesh_impl.shift_by_n_processors(
            lowered_x, mesh_axis, odiv, self._wrap)
        lowering.add_counter("shift[%d]" % odiv, laid_out_size)
      else:
        # shift by odiv processors + omod positions
        sliced = mesh_impl.slicewise(
            lambda x: my_slice(x, 0, block_size - omod), lowered_x)
        second_part = mesh_impl.shift_by_n_processors(
            sliced, mesh_axis, odiv, self._wrap)
        lowering.add_counter(
            "shift[%d]" % odiv,
            laid_out_size * (block_size - omod) // block_size)
        sliced = mesh_impl.slicewise(
            lambda x: my_slice(x, block_size - omod, omod), lowered_x)
        first_part = mesh_impl.shift_by_n_processors(
            sliced, mesh_axis, odiv + 1, self._wrap)
        lowered_y = mesh_impl.slicewise(
            lambda a, b: tf.concat([a, b], axis), first_part, second_part)
        lowering.add_counter(
            "shift[%d]" % (odiv + 1), laid_out_size * omod // block_size)
    lowering.set_tensor_lowering(self.outputs[0], lowered_y)


def shift(x, offset, dim, wrap, name=None):
  """Shift operation.

  Shift x right by +offset in dimension dim.

  Args:
    x: a Tensor
    offset: an integer. If negative, shift left instead of right.
    dim: a Dimension of x
    wrap: a boolean - whether to wrap (True) or pad with zeros (False).
    name: an optional string

  Returns:
    a Tensor with the same shape and dtype as x
  """
  return ShiftOperation(x, offset, dim, wrap, name=name).outputs[0]


def dynamic_shift(x, offset, dim, wrap):
  """Shift with dynamic offset.

  Shift x right by +offset in dimension dim.

  Args:
    x: a Tensor
    offset: an Tensor whose shape is a subset of x.shape.dims - [dim]
    dim: a Dimension of x
    wrap: a boolean - whether to wrap (True) or pad with zeros (False).

  Returns:
    a Tensor with the same shape and dtype as x
  """
  if dim not in x.shape.dims:
    raise ValueError("dim must be a dimension of x")
  if dim in offset.shape.dims:
    raise ValueError("dim may not appear in offset")
  for d in offset.shape.dims:
    if d not in x.shape.dims:
      raise ValueError("offset.shape %s must be a subset of x.shape %s"
                       % (offset.shape, x.shape))
  tmp_dim = Dimension("dynamic_shift_tmp", dim.size)
  x_reshaped = replace_dimensions(x, dim, tmp_dim)
  dim_range = mtf_range(x.mesh, dim, dtype=tf.int32)
  tmp_dim_range = mtf_range(x.mesh, tmp_dim, dtype=tf.int32)
  tmp_dim_range_offset = tmp_dim_range + offset
  if wrap:
    tmp_dim_range_offset = mod(tmp_dim_range_offset, dim.size)
  perm = cast(equal(dim_range, tmp_dim_range_offset), x.dtype)
  return einsum([x_reshaped, perm], output_shape=x.shape)


class SliceOperation(Operation):
  """tf.slice.

  We support the slice operation along one axis. Similar to tf.slice, specify
  the begin and size values for the slice_dim.
  """

  def __init__(self, x, begin, size, slice_dim_name, name=None):
    super(SliceOperation, self).__init__([x], name=name or "slice")
    dim_names = x.shape.dimension_names
    self._axis = axis = dim_names.index(slice_dim_name)
    self._begin = begin
    self._slice_dim = Dimension(slice_dim_name, size)
    input_shape = self._inputs[0].shape
    output_shape = Shape(
        input_shape.dims[:axis] + [self._slice_dim] + input_shape.dims[axis+1:])
    self._outputs = [Tensor(self, output_shape, x.dtype)]
    self._splittable_dims, self._unsplittable_dims = (
        self._initialize_splittable_and_unsplittable_dims(
            "splittable", [slice_dim_name]))

  def gradient(self, grad_ys):
    actual_size = self._inputs[0].shape.dims[self._axis].size
    return [
        pad(grad_ys[0],
            [self._begin, actual_size - self._slice_dim.size - self._begin],
            self._slice_dim.name)]

  def lower(self, lowering):
    mesh_impl = lowering.mesh_impl(self)
    if mesh_impl.tensor_dimension_to_mesh_axis(self._slice_dim) is not None:
      raise ValueError("can't slice along split axis")
    inputs = self._inputs[0]
    ndims = self._inputs[0].shape.ndims
    axis = self._axis
    begin = [0] * axis + [self._begin] + [0] * (ndims - axis - 1)
    size = [-1] * axis + [self._slice_dim.size] + [-1] * (ndims - axis - 1)

    def slicewise_fn(x, begin, size):
      return tf.slice(x, begin, size, name="slice")
    y = mesh_impl.slicewise(
        slicewise_fn, lowering.tensors[inputs], begin, size)
    lowering.set_tensor_lowering(self.outputs[0], y)


class PadOperation(Operation):
  """tf.pad.

  Similar to tf.pad but we only pad along one axis given by pad_dim_name
  with values specified by paddings. paddings is a list of two
  values, giving the padding value before and after pad_dim.
  """

  def __init__(self, x, paddings, pad_dim_name, name=None):
    super(PadOperation, self).__init__([x], name=name or "pad")
    assert len(paddings) == 2
    input_shape = self._inputs[0].shape
    dim_names = [dim.name for dim in x.shape.dims]
    if pad_dim_name not in dim_names:
      raise ValueError("Padding dim name %s not found in input." % pad_dim_name)
    self._paddings = paddings
    self._axis = axis = dim_names.index(pad_dim_name)
    output_size = input_shape.dims[axis].size + sum(paddings)
    self._output_dim = Dimension(pad_dim_name, output_size)
    output_shape = Shape(
        input_shape.dims[:axis] +
        [self._output_dim] + input_shape.dims[axis+1:])
    self._outputs = [Tensor(self, output_shape, x.dtype)]
    self._splittable_dims, self._unsplittable_dims = (
        self._initialize_splittable_and_unsplittable_dims(
            "splittable", [pad_dim_name]))

  def gradient(self, grad_ys):
    slice_dim_name = self._output_dim.name
    slice_size = self._inputs[0].shape.dims[self._axis].size
    return [mtf_slice(grad_ys[0], self._paddings[0],
                      slice_size, slice_dim_name)]

  def lower(self, lowering):
    mesh_impl = lowering.mesh_impl(self)
    if mesh_impl.tensor_dimension_to_mesh_axis(self._output_dim) is not None:
      raise ValueError("can't pad along split axis")
    inputs = self._inputs[0]
    ndims = self._inputs[0].shape.ndims
    axis = self._axis
    paddings = [[0, 0]] * axis + [self._paddings] + [[0, 0]]* (ndims - axis - 1)

    def slicewise_fn(x, paddings):
      return tf.pad(x, paddings, name="pad")
    y = mesh_impl.slicewise(
        slicewise_fn, lowering.tensors[inputs], paddings)
    lowering.set_tensor_lowering(self.outputs[0], y)


class OneHotOperation(Operation):
  """Like tf.one_hot.
  """

  def __init__(self, indices, output_dim, on_value, off_value, dtype,
               name=None):
    super(OneHotOperation, self).__init__([indices], name=name or "one_hot")
    if not indices.dtype.is_integer:
      raise ValueError("indices requires an integer dtype got %s" % indices)
    self._output_dim = output_dim
    self._on_value = on_value
    self._off_value = off_value
    self._dtype = dtype
    output_shape = Shape(indices.shape.dims + [output_dim])
    self._outputs = [Tensor(self, output_shape, dtype)]

    # Rerun to take the new output into account.
    self._splittable_dims, self._unsplittable_dims = (
        self._initialize_all_dimensions_as_splittable())

  def lower(self, lowering):
    mesh_impl = lowering.mesh_impl(self)
    indices = self.inputs[0]
    output_shape = self.outputs[0].shape
    output_slice_shape = mesh_impl.slice_shape(output_shape)
    mesh_axis = mesh_impl.tensor_dimension_to_mesh_axis(self._output_dim)
    depth = output_slice_shape[-1]
    if mesh_axis is None:
      offset = 0
    else:
      offset = mesh_impl.slicewise(
          tf.multiply, mesh_impl.laid_out_pcoord(mesh_axis), depth)

    def slicewise_fn(indices_slice, offset):
      return tf.one_hot(indices_slice - offset,
                        depth,
                        on_value=tf.cast(self._on_value, self._dtype),
                        off_value=tf.cast(self._off_value, self._dtype),
                        dtype=self._dtype)
    y = mesh_impl.slicewise(
        slicewise_fn, lowering.tensors[indices], offset)
    lowering.set_tensor_lowering(self.outputs[0], y)


class ImportOperation(Operation):
  """Import a tf.Tensor onto a mesh."""

  def __init__(self, mesh, tf_tensor, shape, name=None):
    super(ImportOperation, self).__init__([], mesh=mesh, name=name or "import")
    tf_tensor = tf.convert_to_tensor(tf_tensor)
    if not tf_tensor.shape.is_compatible_with(shape.to_integer_list):
      raise ValueError("Incompatible Shape - trying to import %s with shape %s"
                       % (tf_tensor, shape))
    self._outputs = [Tensor(self, shape, tf_tensor.dtype)]
    self._tf_tensor = tf_tensor

    # Rerun to take the new output into account.
    self._splittable_dims, self._unsplittable_dims = (
        self._initialize_all_dimensions_as_splittable())

  def lower(self, lowering):
    mesh_impl = lowering.mesh_impl(self)
    lowering.set_tensor_lowering(
        self.outputs[0],
        mesh_impl.import_tf_tensor(self.outputs[0], self._tf_tensor))


class ImportLaidOutTensorOperation(Operation):
  """Import LaidOutTensor."""

  def __init__(self, mesh, laid_out_tensor, shape, name=None):
    super(ImportLaidOutTensorOperation, self).__init__([],
                                                       mesh=mesh,
                                                       name=name or "import")
    dtype = laid_out_tensor.tensor_list[0].dtype
    self._outputs = [Tensor(self, shape, dtype)]
    self._laid_out_tensor = laid_out_tensor

    # For this operation, it doesn't make sense to talk about the splittability
    # of dimensions, because laid_out_tensor depends on a particular layout.
    self._splittable_dims, self._unsplittable_dims = (
        self._initialize_splittable_and_unsplittable_dims("unsplittable"))

  def lower(self, lowering):
    lowering.set_tensor_lowering(self.outputs[0], self._laid_out_tensor)


def anonymous_shape(shape):
  shape = convert_to_shape(shape)
  return Shape([Dimension("_anonymous_%i" % i, d.size)
                for i, d in enumerate(shape)])


def anonymize(x):
  return reshape(x, anonymous_shape(x.shape))


def import_tf_tensor(mesh, tf_tensor, shape=None, name=None):
  tf_tensor = tf.convert_to_tensor(tf_tensor)
  if shape is None:
    shape = Shape([])
    assert not tf_tensor.shape.as_list()
  return ImportOperation(
      mesh, tf_tensor, convert_to_shape(shape), name=name).outputs[0]


def import_laid_out_tensor(mesh, laid_out_tensor, shape, name=None):
  """Import a laid_out_tensor.

  For expert users.
  The input must be laid out appropriately given the eventual MeshImpl,
  and layout.

  Args:
    mesh: a Mesh
    laid_out_tensor: a LaidOutTensor
    shape: a mtf.Shape
    name: an optional string

  Returns:
   a mtf.Tensor
  """
  return ImportLaidOutTensorOperation(
      mesh, laid_out_tensor, convert_to_shape(shape), name=name).outputs[0]


def import_fully_replicated(mesh, tf_tensor, shape, name=None):
  return reshape(import_tf_tensor(
      mesh, tf_tensor, anonymous_shape(shape), name), shape)


class LazyLaidOutTensor(object):
  """Computes a function later to create a LaidOutTensor.

  The given to_laid_out_tensor_fn() is called every time
  the to_laid_out_tensor() method is called.  Really, we should not need this
  class, since XLA rematerialization should do it all for us.
  """

  def __init__(self, to_laid_out_tensor_fn, slice_shape):
    self._to_laid_out_tensor_fn = to_laid_out_tensor_fn
    self._slice_shape = slice_shape

  def to_laid_out_tensor(self):
    return self._to_laid_out_tensor_fn()

  @property
  def slice_shape(self):
    return self._slice_shape


class VariableDType(object):
  """Class containing datatype information for a variable.

  A variable has three datatypes.

  master_dtype:
    the datatype used for storing the variable to checkpoints

  slice_dtype:
    the datatype used for maintaining and updating the value during training

  activation_dtype:
    the datatype used for computation.  Calls to get_variable return a Tensor
    with this datatype.

  If slice_dtype=tf.bfloat16 during training, then repeated roundoff errors
  interfere with model quality - use tf.float32 instead.  Otherwise, tf.bfloat16
  can help reduce memory usage and checkpoint size.  It is necessary to keep
  master_dtype the same between training/inference/evaluation in order to read
  and write checkpoints.

  We will later extend this functionality to allow for custom quantization code.
  """

  def __init__(self,
               master_dtype=tf.float32,
               slice_dtype=None,
               activation_dtype=None):
    self._master_dtype = master_dtype
    self._slice_dtype = slice_dtype or master_dtype
    self._activation_dtype = activation_dtype or master_dtype

  @property
  def master_dtype(self):
    return self._master_dtype

  @property
  def slice_dtype(self):
    return self._slice_dtype

  @property
  def activation_dtype(self):
    return self._activation_dtype


class Variable(Operation):
  """Variable."""

  def __init__(
      self, mesh, name, shape, dtype, initializer, trainable, **kwargs):
    super(Variable, self).__init__([], mesh, name="name_will_be_set_later")
    if not isinstance(dtype, VariableDType):
      raise ValueError("dtype must be a VariableDType got %s" % dtype)
    self._dtype = dtype
    self._trainable = trainable
    if not isinstance(self, StackedVariable):
      with tf.device(mesh.variable_placer_fn), utils.outside_all_rewrites():
        self._master = tf.get_variable(
            name,
            shape.to_integer_list,
            dtype=self.master_dtype,
            initializer=initializer,
            trainable=trainable,
            **kwargs)
      self._name = self._master.name[:self._master.name.find(":")]
    self._outputs = [Tensor(self, shape, dtype.activation_dtype)]

    # Rerun to take the new output into account.
    self._splittable_dims, self._unsplittable_dims = (
        self._initialize_all_dimensions_as_splittable())

    self.graph.all_variables.append(self)
    if trainable:
      self.graph.trainable_variables.append(self)

  def __repr__(self):
    return "Variable(%s)" % self.value

  def lower(self, lowering):
    mesh_impl = lowering.mesh_impl(self)
    with utils.outside_all_rewrites():
      sv = mesh_impl.LaidOutVariable(self, mesh_impl)
    lowering.variables[self] = sv
    lowering.set_tensor_lowering(
        self.outputs[0],
        mesh_impl.slicewise(
            tf.cast, sv.laid_out_tensor, self.activation_dtype))
    if self._trainable:
      lowering.add_counter("variables/trainable", self.outputs[0].size)
    else:
      lowering.add_counter("variables/untrainable", self.outputs[0].size)

  @property
  def value(self):
    return self.outputs[0]

  @property
  def shape(self):
    return self.value.shape

  @property
  def size(self):
    return self.shape.size

  @property
  def dtype(self):
    return self._dtype

  @property
  def master_dtype(self):
    return self._dtype.master_dtype

  @property
  def slice_dtype(self):
    return self._dtype.slice_dtype

  @property
  def activation_dtype(self):
    return self._dtype.activation_dtype

  @property
  def trainable(self):
    return self._trainable

  @property
  def master_device(self):
    return self._master.device

  def get_master(self):
    return self._master

  def assign_to_master(self, val):
    return tf.assign(self._master, val)


class StackedVariable(Variable):
  """A Variable which combines many variables into one.

  This is a performance optimization to reduce the time associated with large
  numbers of slice variables.  See Graph.rewrite_stack_variables() for usage.
  """

  def __init__(self, vs):
    """Create a StackedVariable.

    Args:
      vs: a list of Variables
    """
    shape = Shape([Dimension("stacked", len(vs))] + vs[0].shape.dims)
    name = "stacked/" + vs[0].name
    # TODO(noam): verify that vs are the same shape, etc.
    super(StackedVariable, self).__init__(
        vs[0].mesh, name, shape, vs[0].dtype, None, vs[0].trainable)
    self._name = name
    self._masters = [v.get_master() for v in vs]
    self._original_names = [v.name for v in vs]

    # Rerun to take the new output into account.
    self._splittable_dims, self._unsplittable_dims = (
        self._initialize_all_dimensions_as_splittable())

  @property
  def original_names(self):
    return self._original_names

  @property
  def master_device(self):
    return self._masters[0].device

  def get_master(self):
    with tf.device(self.master_device):
      return tf.stack(self._masters)

  def assign_to_master(self, val):
    return tf.group([
        tf.assign(var_slice, val_slice) for var_slice, val_slice
        in zip(self._masters, tf.unstack(val))])


class ReadVariable(Operation):
  """Read a variable."""

  def __init__(self, var, name=None):
    super(ReadVariable, self).__init__(
        var.outputs, name=name or "read_variable")
    self._var = var
    self._outputs = [Tensor(self, var.shape, var.activation_dtype)]

  def gradient(self, grad_ys):
    return grad_ys

  def lower(self, lowering):
    mesh_impl = lowering.mesh_impl(self)
    sv = lowering.variables[self._var]
    lowering.set_tensor_lowering(
        self.outputs[0], mesh_impl.slicewise(
            tf.cast, sv.laid_out_tensor, self._var.activation_dtype))


def get_variable(mesh, name, shape, dtype=tf.float32,
                 master_dtype=None, slice_dtype=None, activation_dtype=None,
                 initializer=None, trainable=True,
                 **kwargs):
  """Create a new variable or retrieve an already-created one.

  Args:
    mesh: a Mesh
    name: a string (uses the existing tf.variable_scope())
    shape: a Shape
    dtype: a VariableDType or a tf.DType
    master_dtype: an optional tf.DType (deprecated - use dtype arg)
    slice_dtype: an optional tf.DType (deprecated - use dtype arg)
    activation_dtype: an optional tf.DType (deprecated - use dtype arg)
    initializer: an optional tf initializer function
    trainable: a boolean
    **kwargs: additional keyword arguments to tf.get_variable

  Returns:
    a Tensor with the given shape and dtype equal to dtype.activation_dtype
  """
  if dtype is None:
    dtype = VariableDType(master_dtype, slice_dtype, activation_dtype)
  elif isinstance(dtype, tf.DType):
    dtype = VariableDType(
        master_dtype or dtype, slice_dtype or dtype, activation_dtype or dtype)
  elif not isinstance(dtype, VariableDType):
    raise ValueError("dtype should be a tf.dtype or a mtf.VariableDType")
  scope_name = tf.get_variable_scope().name
  if scope_name:
    full_name = scope_name + "/" + name
  else:
    full_name = name
  if initializer is None:
    tf.logging.warning(
        "Using default tf glorot_uniform_initializer for variable %s "
        " The initialzer will guess the input and output dimensions "
        " based on dimension order." % full_name)
  if full_name in mesh.graph.name_to_variable:
    var = mesh.graph.name_to_variable[full_name]
  else:
    var = Variable(
        mesh, name, convert_to_shape(shape), dtype, initializer, trainable,
        **kwargs)
    if var.name != full_name:
      raise ValueError(
          "Expected var.name == full_name.  %s vs %s" % (var.name, full_name))
    mesh.graph.name_to_variable[full_name] = var
  return var.outputs[0]


def read_variable(var):
  return ReadVariable(var).outputs[0]


def assign_slice(variable, slice_var, val):
  return tf.assign(
      slice_var,
      tf.cast(val, variable.slice_dtype))


def assign_add_slice(variable, slice_var, val):
  val = tf.cast(val, variable.slice_dtype)
  return tf.assign(slice_var, slice_var + val)


def assign_sub_slice(variable, slice_var, val):
  val = tf.cast(val, variable.slice_dtype)
  return tf.assign(slice_var, slice_var - val)


class Assign(Operation):
  """Assign to one or more variables."""

  def __init__(self, variables, new_values, assign_fn=assign_slice, name=None):
    super(Assign, self).__init__(
        new_values, variables[0].mesh, name=name or "assign")
    self._variables = variables
    self._assign_fn = assign_fn
    self._outputs = []

  def lower(self, lowering):
    ops = []
    for var, val in zip(self._variables, self.inputs):
      ops.append(lowering.variables[var].assign_to_slices(
          self._assign_fn,
          lowering.tensors[val].to_laid_out_tensor().all_slices))
    lowering.operations[self] = tf.group(ops)

  @property
  def assign_fn(self):
    return self._assign_fn

  @property
  def variables(self):
    return self._variables


def assign(var, new_val, assign_fn=assign_slice, name=None):
  """Assign a new value to a variable.

  Args:
    var: either a Variable operation or its output Tensor,
      or the output of a chain of unary operations starting with a Variable.
    new_val: a Tensor
    assign_fn: a function from
        (mtf.Variable, tf.Variable, tf.Tensor) -> tf.Operation
    name: a string for the Assign op.
  Returns:
    an Operation
  Raises:
    ValueError: if var is not a Variable and var.operation is not a Variable
  """
  # find the original Variable operation.
  if isinstance(var, Tensor):
    var = var.operation
  while not isinstance(var, Variable) and len(var.inputs) == 1:
    var = var.inputs[0].operation
  if not isinstance(var, Variable):
    raise ValueError("var must be a mtf.Variable or its output Tensor.")
  return Assign([var], [new_val], assign_fn=assign_fn, name=name)


def assign_add(var, new_val):
  return assign(var, new_val, assign_fn=assign_add_slice)


def assign_sub(var, new_val):
  return assign(var, new_val, assign_fn=assign_sub_slice)


class Depend(Operation):
  """Control dependency."""

  def __init__(self, x, dependencies, name=None):
    super(Depend, self).__init__([x], x.mesh, name=name or "depend")
    for d in dependencies:
      if not isinstance(d, Operation) and not isinstance(d, Tensor):
        raise ValueError("dependencies must be mtf.Operations or mtf.Tensor."
                         "got %s" % d)
    self._dependencies = dependencies
    self._outputs = [Tensor(self, x.shape, x.dtype)]

  def lower(self, lowering):
    mesh_impl = lowering.mesh_impl(self)
    if not mesh_impl.supports_control_dependencies:
      raise ValueError("Mesh does not suppport control dependencies.")

    control_inputs = []
    for d in self._dependencies:
      if isinstance(d, Operation):
        control_inputs.append(lowering.operations[d])
      else:
        control_inputs.append(lowering.tensors[d].tensor_list)

    with tf.control_dependencies(tf.nest.flatten(control_inputs)):
      lowering.set_tensor_lowering(
          self.outputs[0],
          mesh_impl.slicewise(tf.identity,
                              lowering.tensors[self.inputs[0]]))

  def gradient(self, grad_ys):
    return grad_ys


def depend(x, dependencies):
  """Identity of Tensor x that depends on operation dependencies.

  Args:
    x: a Tensor
    dependencies: a list of Operations or Tensors
  Returns:
    an tensor
  """
  return Depend(x, dependencies).outputs[0]


class Constant(Operation):
  """A tensor where every element is the same constant value."""

  def __init__(self, mesh, value, shape, dtype, name=None):
    super(Constant, self).__init__([], mesh, name=name or "constant")
    self._outputs = [Tensor(self, shape, dtype)]
    self._value = value
    # Rerun to take the new output into account.
    self._splittable_dims, self._unsplittable_dims = (
        self._initialize_all_dimensions_as_splittable())

  def lower(self, lowering):
    mesh_impl = lowering.mesh_impl(self)
    slice_shape = mesh_impl.slice_shape(self.outputs[0].shape)
    def tf_fn():
      return tf.constant(value=self._value,
                         dtype=self.outputs[0].dtype,
                         shape=slice_shape)
    lowering.set_tensor_lowering(self.outputs[0], mesh_impl.slicewise(tf_fn))


def constant(mesh, value, shape=None, dtype=tf.float32):
  shape = convert_to_shape(shape)
  return Constant(mesh, value,
                  shape if shape is not None else Shape([]),
                  dtype).outputs[0]


def zeros(mesh, shape, dtype=tf.float32):
  return constant(mesh, 0, shape=convert_to_shape(shape), dtype=dtype)


def zeros_like(t):
  return zeros(t.mesh, t.shape, dtype=t.dtype)


def ones(mesh, shape, dtype=tf.float32):
  return constant(mesh, 1, shape=convert_to_shape(shape), dtype=dtype)


def ones_like(t):
  return ones(t.mesh, t.shape, dtype=t.dtype)


class StopGradient(Operation):
  """Similar to tf.stop_gradient."""

  def __init__(self, x, name=None):
    super(StopGradient, self).__init__(
        [x], x.mesh, name=name or "stop_gradient")
    self._outputs = [Tensor(self, x.shape, x.dtype)]

  def lower(self, lowering):
    lowering.set_tensor_lowering(self.outputs[0],
                                 lowering.tensors[self.inputs[0]])

  @property
  def has_gradient(self):
    return False


def stop_gradient(x):
  return StopGradient(x).outputs[0]


class ScalarSummaryOperation(Operation):
  """Similar to tf.Print."""

  def __init__(self, name, x):
    super(ScalarSummaryOperation, self).__init__(
        [x], x.mesh, name=name)
    if x.shape.dims:
      raise ValueError("ScalarSummaryOperation takes a scalar")
    self._outputs = [Tensor(self, x.shape, x.dtype)]

  def lower(self, lowering):
    lowered_input = lowering.tensors[self.inputs[0]].to_laid_out_tensor()
    tf.add_to_collection(utils.SCALAR_SUMMARIES_COLLECTION_KEY,
                         (self.name, lowered_input.tensor_list[0]))
    lowering.set_tensor_lowering(
        self.outputs[0], lowered_input)

  def gradient(self, grad_ys):
    return grad_ys


def scalar_summary(name, x):
  """Call tf.summary.scalar.

  Caveat - summaries do not generally work on TPU - they need to be rewritten
  into a host call.
  TODO(noam): provide a pointer to code for this.

  Args:
    name: a string
    x: a 0-dimensional Tensor
  Returns:
    a Tensor which is identical in value to x
  """
  return ScalarSummaryOperation(name, x)


class PrintOperation(Operation):
  """Similar to tf.Print."""

  def __init__(self, x, data, message, name=None, **kwargs):
    super(PrintOperation, self).__init__(
        [x], x.mesh, name=name or "Print")
    self._outputs = [Tensor(self, x.shape, x.dtype)]
    self._data = data
    self._message = message
    self._kwargs = kwargs

  def lower(self, lowering):
    lowering.set_tensor_lowering(
        self.outputs[0],
        lowering.mesh_impl(self).Print(
            lowering.tensors[self.inputs[0]],
            [lowering.tensors[d].to_laid_out_tensor() for d in self._data],
            self._message, **self._kwargs))

  def gradient(self, grad_ys):
    return grad_ys


def Print(x, data, message, **kwargs):  # pylint: disable=invalid-name
  """Call tf.Print.

  Args:
    x: a Tensor.
    data: a list of Tensor
    message: a string
    **kwargs: keyword arguments to tf.Print
  Returns:
    a Tensor which is identical in value to x
  """
  message += " %s" % data
  return PrintOperation(x, data, message, **kwargs).outputs[0]


class ReshapeOperation(Operation):
  """Similar to tf.stop_gradient."""

  def __init__(self, x, new_shape, name=None):
    super(ReshapeOperation, self).__init__([x], x.mesh, name=name or "reshape")
    if x.shape.size != new_shape.size:
      raise ValueError("Cannot reshape Tensor %s to shape %s - sizes differ."
                       % (x, new_shape))
    self._outputs = [Tensor(self, new_shape, x.dtype)]

    # Rerun to take the new output into account.
    self._splittable_dims, self._unsplittable_dims = (
        self._initialize_all_dimensions_as_splittable())

  def lower(self, lowering):
    """Lower the ReshapeOperation.

    Reshaping can require collective communication between processors.
    We haven't yet implemented all possible reshapes.  We try to handle the
    common cases here - otherwise we raise a NotImplementedError.

    Args:
      lowering: a Lowering
    Raises:
      NotImplementedError: if we haven't covered this case
    """
    old_shape = self.inputs[0].shape
    new_shape = self.outputs[0].shape
    mesh_impl = lowering.mesh_impl(self)
    slices = lowering.tensors[self.inputs[0]]
    mesh_axis_to_cumprod_old = mesh_impl.mesh_axis_to_cumprod(old_shape)
    mesh_axis_to_cumprod_new = mesh_impl.mesh_axis_to_cumprod(new_shape)
    # Figure out what needs to be done for different mesh-axes
    mesh_axes_allsplit = []
    mesh_axes_allconcat = []
    mesh_axes_alltoall = []
    for mesh_axis, (old_cumprod, new_cumprod) in enumerate(
        zip(mesh_axis_to_cumprod_old, mesh_axis_to_cumprod_new)):
      if new_cumprod != old_cumprod:
        if old_cumprod is None:
          # split in new layout but not in old layout - we need an allsplit
          mesh_axes_allsplit.append(mesh_axis)
        elif new_cumprod is None:
          # split in old layout but not in new layout - we need an allconcat
          mesh_axes_allconcat.append(mesh_axis)
        else:
          # split differently in old and new layouts - we need an alltoall
          mesh_axes_alltoall.append(mesh_axis)

    laid_out_size = mesh_impl.laid_out_size(old_shape)

    # list of (mesh_axis, tensor_axis) pairs to allsplit after the reshape
    # typically we do the allsplit before the reshape, to save communication,
    # but sometimes we need to delay it.
    allsplit_after_reshape = []
    for mesh_axis in mesh_axes_allsplit:
      tensor_axis = old_shape.cumprod_to_tensor_axis(
          mesh_axis_to_cumprod_new[mesh_axis])
      if tensor_axis is None:
        # delay allsplit until after reshape
        tensor_axis = new_shape.cumprod_to_tensor_axis(
            mesh_axis_to_cumprod_new[mesh_axis])
        allsplit_after_reshape.append((mesh_axis, tensor_axis))
      else:
        slices = mesh_impl.allsplit(slices, mesh_axis, tensor_axis)
        laid_out_size //= mesh_impl.shape[mesh_axis].size
    for mesh_axis in mesh_axes_alltoall:
      split_tensor_axis = old_shape.cumprod_to_tensor_axis(
          mesh_axis_to_cumprod_new[mesh_axis])
      if split_tensor_axis is None:
        # TODO(noam): try to handle this case
        raise NotImplementedError(
            "Try first reshaping to insert a new tf dimension,"
            " then changing layout. input_shape=%s output_shape=%s"
            % (self.inputs[0].shape, self.outputs[0].shape))
      concat_tensor_axis = old_shape.cumprod_to_tensor_axis(
          mesh_axis_to_cumprod_old[mesh_axis])
      assert concat_tensor_axis is not None
      slices = mesh_impl.alltoall(
          slices, mesh_axis, split_tensor_axis, concat_tensor_axis)
      lowering.add_counter(
          "alltoall/%s/reshape_op" % mesh_axis, laid_out_size)

    for mesh_axis in mesh_axes_allconcat:
      tensor_axis = old_shape.cumprod_to_tensor_axis(
          mesh_axis_to_cumprod_old[mesh_axis])
      assert tensor_axis is not None
      slices = mesh_impl.allconcat(slices, mesh_axis, tensor_axis)
      laid_out_size *= mesh_impl.shape[mesh_axis].size
      lowering.add_counter(
          "allconcat/%s/reshape_op" % mesh_axis, laid_out_size)
    # now reshape the slices
    new_slice_shape = mesh_impl.slice_shape(new_shape)
    for mesh_axis, tensor_axis in allsplit_after_reshape:
      new_slice_shape[tensor_axis] *= mesh_impl.shape[mesh_axis].size
    def reshape_fn(x):
      return tf.reshape(x, new_slice_shape)
    slices = mesh_impl.slicewise(reshape_fn, slices)
    for mesh_axis, tensor_axis in allsplit_after_reshape:
      slices = mesh_impl.allsplit(slices, mesh_axis, tensor_axis)
    lowering.set_tensor_lowering(self.outputs[0], slices)

  def gradient(self, grad_ys):
    return [reshape(grad_ys[0], self.inputs[0].shape)]


def reshape(x, new_shape, name="reshape"):
  return ReshapeOperation(x, convert_to_shape(new_shape), name=name).outputs[0]


def transpose(x, new_shape, name="transpose"):
  new_shape = convert_to_shape(new_shape)
  if set(x.shape.dims) != set(new_shape.dims):
    raise ValueError("x must have the same dimensions as new_shape %s vs %s"
                     % (x, new_shape))
  return einsum([x], output_shape=new_shape, name=name)


def rename_dimension(x, old_name, new_name):
  """Reshape a Tensor, renaming one dimension.

  Args:
    x: a Tensor
    old_name: a string
    new_name: a string

  Returns:
    a Tensor
  """
  return reshape(x, x.shape.rename_dimension(old_name, new_name))


def replace_dimensions(tensor_or_shape, old_dim_or_dims, new_dim_or_dims):
  """Replace dimensions in a Tensor or Shape.

  old_dim_or_dims consists of a single dimension or a list of dimensions
  that must occur consecutively in the input shape.  They are replaced
  by the dimensions in new_dim_or_dims.

  Args:
    tensor_or_shape: a Tensor or a Shape
    old_dim_or_dims: a Dimension or a list of Dimensions
    new_dim_or_dims: a Dimensions or a list of Dimensions
  Returns:
    a new Tensor or a Shape
  """
  if isinstance(tensor_or_shape, Tensor):
    return reshape(tensor_or_shape, replace_dimensions(
        tensor_or_shape.shape, old_dim_or_dims, new_dim_or_dims))
  if not isinstance(tensor_or_shape, Shape):
    raise ValueError(
        "tensor_or_shape must be a Tensor or Shape got %s" % (tensor_or_shape,))
  in_dims = tensor_or_shape.dims
  if isinstance(old_dim_or_dims, Dimension):
    old_dim_or_dims = [old_dim_or_dims]
  if isinstance(new_dim_or_dims, Dimension):
    new_dim_or_dims = [new_dim_or_dims]
  if not isinstance(old_dim_or_dims, list) or not old_dim_or_dims:
    raise ValueError(
        "old_dim_or_dims must be a Dimension or a list of Dimension got %s"
        % (old_dim_or_dims,))
  if not isinstance(new_dim_or_dims, list) or not new_dim_or_dims:
    raise ValueError(
        "new_dim_or_dims must be a Dimension or a list of Dimension got %s"
        % (new_dim_or_dims,))
  try:
    positions = [in_dims.index(d) for d in old_dim_or_dims]
    pos = positions[0]
    if positions != list(range(pos, pos + len(positions))):
      raise ValueError()
  except ValueError:
    raise ValueError(
        "old_dim_or_dims must be a subsequence of the input's dimensions"
        " old_dim_or_dims=%s input's dimensions=%s" %
        (old_dim_or_dims, in_dims))
  return Shape(in_dims[:pos] + new_dim_or_dims +
               in_dims[pos + len(old_dim_or_dims):])


def einsum(xs, output_shape=None, reduced_dims=None, name=None):
  """Einstein summation.

  einsum(xs, output_shape) is equivalent to broadcasting all inputs
  to the union of all of their shapes, multiplying them componentwise,
  and finally reduce_summing down to output_shape.

  One common case of this is matrix multiplication:
      x has shape [a, b]
      y has shape [b, c]
      matmul(x, y) == einsum([x, y], output_shape=[a, c])

  We provide a few options for specifying the output shape:

  If neither output_shape nor reduced_dims is specified, then the output
  shape is set to the contain all dimensions that appear exactly once in the
  inputs, in order of appearance.

  If output_shape is not specified, then the output shape is set to the contain
  all dimensions that appear in xs but not in reduced_dims, in the order
  that they appear in xs.  If reduced_dims is also not specified, then
  reduced_dims is set to the set of all dimensions that appear at least twice in
  xs.

  If both output_shape and reduced_dims are specified, then we check that
  reduced_dims matches the set of dimensions present in xs but not in
  output_shape, and throw an exception if it does not.  This helps to reduce
  bugs.

  Args:
    xs: a list of Tensors
    output_shape: an optional Shape.
    reduced_dims: an optional list of Dimensions.
    name: an optional string
  Returns:
    a Tensor
  Raises:
    ValueError: if reduced_dims contradicts output_shape
  """
  output_shape = convert_to_shape(output_shape)
  input_dim_count = collections.defaultdict(int)
  input_dims = []
  for x in xs:
    for d in x.shape.dims:
      if d not in input_dim_count:
        input_dims.append(d)
      input_dim_count[d] += 1
  if reduced_dims is not None:
    for d in reduced_dims:
      if not isinstance(d, Dimension):
        raise ValueError("reduced_dims must be a list of Dimensions.  Got %s."
                         % (reduced_dims,))
  if output_shape is None:
    if reduced_dims is None:
      reduced_dims = [d for d, c in six.iteritems(input_dim_count) if c > 1]
    output_shape = Shape([d for d in input_dims if d not in reduced_dims])
  elif reduced_dims is not None:
    computed_reduced_dims = [
        d for d in input_dims if d not in output_shape.dims]
    if set(computed_reduced_dims) != set(reduced_dims):
      raise ValueError(
          "Specified reduced_dims and output_shape do not match."
          " xs=%s output_shape=%s reduced_dims=%s " % (
              xs, output_shape, reduced_dims))
  return EinsumOperation(xs, output_shape, name=name).outputs[0]


def matmul(a, b, output_shape=None, reduced_dims=None, name=None):
  """Alias for einsum([a, b])."""
  return einsum(
      [a, b], output_shape=output_shape, reduced_dims=reduced_dims, name=name)


def _reduction_output_shape(x, output_shape, reduced_dim):
  """Helper function to reduce_sum, etc."""
  if output_shape is None:
    if reduced_dim is None:
      return Shape([])
    else:
      if reduced_dim not in x.shape.dims:
        raise ValueError(
            "reduced_dim=%s not in x.shape.dims=%s" % (reduced_dim, x.shape))
      return x.shape - reduced_dim
  if reduced_dim is not None:
    if [reduced_dim] != [d for d in x.shape.dims if d not in output_shape.dims]:
      raise ValueError(
          "reduced_dim contradicts output_shape:"
          "x=%s output_shape=%s reduced_dim=%s" %
          (x, output_shape, reduced_dim))
  return output_shape


def reduce_sum(x,
               disable_positional_args=None,
               output_shape=None,
               reduced_dim=None,
               name=None):
  """Reduction on 1 or more axes.

  If reduced_dim is present, then only that dimension is reduced out.
  Alternatively, specify output_shape.
  Do not specify both reduced_dim and output_shape.
  If neither is specified, then all dimensions are reduced out.

  Args:
    x: a Tensor
    disable_positional_args: None
    output_shape: an optional Shape.  Must be a subsequence of x.shape.
    reduced_dim: a mtf.Dimension
    name: an optional string
  Returns:
    a Tensor
  """
  output_shape = convert_to_shape(output_shape)
  reduced_dim = convert_to_dimension(reduced_dim)
  assert disable_positional_args is None
  output_shape = _reduction_output_shape(x, output_shape, reduced_dim)
  if output_shape == x.shape:
    return x
  return ReduceOperation(x, output_shape, "SUM", name=name).outputs[0]


def reduce_mean(x,
                disable_positional_args=None,
                output_shape=None,
                reduced_dim=None,
                name=None):
  """Reduction on 1 or more axes.

  If reduced_dim is present, then only that dimension is reduced out.
  Alternatively, specify output_shape.
  Do not specify both reduced_dim and output_shape.
  If neither is specified, then all dimensions are reduced out.

  Args:
    x: a Tensor
    disable_positional_args: None
    output_shape: an optional Shape. Must be a subsequence of x.shape.
    reduced_dim: a mtf.Dimension
    name: an optional string

  Returns:
    a Tensor
  """
  output_shape = convert_to_shape(output_shape)
  reduced_dim = convert_to_dimension(reduced_dim)
  assert disable_positional_args is None
  output_shape = _reduction_output_shape(x, output_shape, reduced_dim)
  with tf.variable_scope(name, default_name="reduce_mean"):
    if output_shape == x.shape:
      return x
    return reduce_sum(
        x, output_shape=output_shape) * (output_shape.size / x.shape.size)


def reduce_max(x,
               disable_positional_args=None,
               output_shape=None,
               reduced_dim=None,
               name=None):
  """Reduction on 1 or more axes.

  Args:
    x: a Tensor
    disable_positional_args: None
    output_shape: an optional Shape.  Must be a subsequence of x.shape.
    reduced_dim: an optional Dimension
    name: an optional string
  Returns:
    a Tensor
  """
  output_shape = convert_to_shape(output_shape)
  reduced_dim = convert_to_dimension(reduced_dim)
  assert disable_positional_args is None
  output_shape = _reduction_output_shape(x, output_shape, reduced_dim)
  if output_shape is None:
    output_shape = Shape([])
  if output_shape == x.shape:
    return x
  return ReduceOperation(
      x, output_shape, "MAX", name=name or "reduce_max").outputs[0]


def reduce_min(x,
               disable_positional_args=None,
               output_shape=None,
               reduced_dim=None,
               name=None):
  """Reduction on 1 or more axes.

  Args:
    x: a Tensor
    disable_positional_args: None
    output_shape: an optional Shape.  Must be a subsequence of x.shape.
    reduced_dim: an optional Dimension
    name: an optional string
  Returns:
    a Tensor
  """
  output_shape = convert_to_shape(output_shape)
  reduced_dim = convert_to_dimension(reduced_dim)
  assert disable_positional_args is None
  output_shape = _reduction_output_shape(x, output_shape, reduced_dim)
  if output_shape is None:
    output_shape = Shape([])
  if output_shape == x.shape:
    return x
  return ReduceOperation(
      x, output_shape, "MIN", name=name or "reduce_min").outputs[0]


def reduce_all(x,
               disable_positional_args=None,
               output_shape=None,
               reduced_dim=None,
               name=None):
  output_shape = convert_to_shape(output_shape)
  reduced_dim = convert_to_dimension(reduced_dim)
  return cast(reduce_min(to_float(x),
                         disable_positional_args=disable_positional_args,
                         output_shape=output_shape,
                         reduced_dim=reduced_dim,
                         name=name or "reduce_all"), tf.bool)


def reduce_any(x,
               disable_positional_args=None,
               output_shape=None,
               reduced_dim=None,
               name=None):
  output_shape = convert_to_shape(output_shape)
  reduced_dim = convert_to_dimension(reduced_dim)
  return cast(reduce_max(to_float(x),
                         disable_positional_args=disable_positional_args,
                         output_shape=output_shape,
                         reduced_dim=reduced_dim,
                         name=name or "reduce_any"), tf.bool)


class TopKOperation(Operation):
  """Compute top k indices and values - see comment on top_k() below."""

  def __init__(self, x, reduced_dim, k_dim, name=None):
    super(TopKOperation, self).__init__([x], name=name or "top_k")
    self._value_dtype = x.dtype
    if reduced_dim not in x.shape.dims:
      raise ValueError("reduced dim %s must be in x.shape %s"
                       % (reduced_dim, x.shape))
    if k_dim.size > reduced_dim.size:
      raise ValueError("k_dim.size must be <= reduced_dim.size: %s vs %s"
                       % (k_dim, reduced_dim))
    output_shape = x.shape - reduced_dim + k_dim
    self._outputs = [Tensor(self, output_shape, x.dtype),
                     Tensor(self, output_shape, tf.int32),]
    self._reduced_dim = reduced_dim
    self._k_dim = k_dim
    self._splittable_dims, self._unsplittable_dims = (
        self._initialize_splittable_and_unsplittable_dims(
            "splittable", [self._k_dim.name]))

  def gradient(self, grad_ys):
    dvalue = grad_ys[0]
    indices = self.outputs[1]
    mapping = one_hot(indices, self._reduced_dim, dtype=self._value_dtype)
    return [einsum([dvalue, mapping], output_shape=self.inputs[0].shape)]

  def lower(self, lowering):
    mesh_impl = lowering.mesh_impl(self)
    x = self.inputs[0]
    ndims = x.shape.ndims
    reduced_axis = x.shape.dims.index(self._reduced_dim)
    reduced_mesh_axis = mesh_impl.tensor_dimension_to_mesh_axis(
        self._reduced_dim)
    if reduced_mesh_axis is not None:
      reduced_dim_per_shard = (
          self._reduced_dim.size // mesh_impl.shape[reduced_mesh_axis].size)
    else:
      reduced_dim_per_shard = self._reduced_dim.size
    def _slicewise_top_k(t):
      t = tf.transpose(
          t, [i for i in range(ndims) if i != reduced_axis] + [reduced_axis])
      if self._k_dim.size == 1:
        # top_k seems to be slow on TPU - use reduce_max and argmax instead
        return (tf.expand_dims(tf.math.reduce_max(t, -1), -1),
                tf.expand_dims(tf.cast(tf.math.argmax(t, -1), tf.int32), -1))
      else:
        return tf.math.top_k(t, min(self._k_dim.size, reduced_dim_per_shard))
    values, indices = mesh_impl.slicewise(_slicewise_top_k, lowering.tensors[x])
    if reduced_mesh_axis is not None:
      # indices are now indices within a shard.  Make them global indices.
      indices = mesh_impl.slicewise(
          lambda idxs, pcoord: idxs + pcoord * reduced_dim_per_shard,
          indices, mesh_impl.laid_out_pcoord(reduced_mesh_axis))
      # concatenate values and indices across processors,
      #   duplicating the result across mesh axis `reduced_mesh_axis`.
      values = mesh_impl.allconcat(values, reduced_mesh_axis, ndims - 1)
      indices = mesh_impl.allconcat(indices, reduced_mesh_axis, ndims - 1)
      # final reduction to find top k among all shards
      def _global_top_k(vals, global_indices):
        vals, local_indices = tf.math.top_k(vals, self._k_dim.size)
        return vals, tf.gather(global_indices,
                               local_indices,
                               batch_dims=ndims-1)
      values, indices = mesh_impl.slicewise(_global_top_k, values, indices)
    lowering.set_tensor_lowering(self.outputs[0], values)
    lowering.set_tensor_lowering(self.outputs[1], indices)


def top_k(x, reduced_dim, k_dim, name=None):
  """Like tf.math.top_k.

  This operation returns two tensors with the same shape.  The output shape
  is identical to the shape of x, except that reduced_dim is removed and
  k_dim is inserted at the end.

  Args:
    x: a Tensor
    reduced_dim: a Dimension in x.shape.dims.
    k_dim: a Dimension.  The size determines k.
    name: optional string.
  Returns:
    values: a Tensor with same type as x.
    indices: a Tensor with dtype tf.int32
  """
  if k_dim.size > 1 and k_dim.size < 5:
    return _iterative_top_k(x, reduced_dim, k_dim, name=name)
  else:
    op = TopKOperation(x, reduced_dim, k_dim, name=name)
    return op.outputs[0], op.outputs[1]


def _iterative_top_k(x, reduced_dim, k_dim, name=None):
  """Like tf.top_k.

  Iterative implementation of top_k.
  This is faster for small k on TPU for now, since the implementation of
  tf.nn.top_k() seems to use sorting.

  Args:
    x: a Tensor
    reduced_dim: a Dimension in x.shape.dims.
    k_dim: a Dimension.  The size determines k.
    name: optional string.
  Returns:
    values: a Tensor with same type as x.
    indices: a Tensor with dtype tf.int32
  """
  reduced_dim = convert_to_dimension(reduced_dim)
  k_dim = convert_to_dimension(k_dim)
  indices = []
  values = []
  k = k_dim.size
  with tf.name_scope(name, default_name="top_k"):
    for i in xrange(k):
      max_val, max_index = top_1(x, reduced_dim)
      indices.append(max_index)
      values.append(max_val)
      if i + 1 < k:
        x += one_hot(max_index, reduced_dim, on_value=-1e9, dtype=x.dtype)
  return stack(values, k_dim.name, -1), stack(indices, k_dim.name, -1)


def top_1(x, reduced_dim, name=None):
  """Max and Argmax.

  Args:
    x: a Tensor
    reduced_dim: a Dimension in x.shape.dims
    name: an optional string
  Returns:
    values: Tensor equal to mtf.reduce_max(x, reduced_dim=reduced_dim)
    indices: a Tensor with dtype tf.int32
  """
  one_dim = Dimension("_one", 1)
  values, indices = top_k(x, reduced_dim, one_dim, name=name)
  values = reshape(values, values.shape - one_dim)
  indices = reshape(indices, indices.shape - one_dim)
  return values, indices


def argmax(x, reduced_dim, name=None):
  """Compute argmax.

  Args:
    x: a Tensor
    reduced_dim: a Dimension in x.shape.dims
    name: an optional string
  Returns:
    A Tensor with shape x.shape - reduced_dim and dtype tf.int32.
  """
  reduced_dim = convert_to_dimension(reduced_dim)
  return top_1(x, reduced_dim, name=name)[1]


def sample_with_temperature(x, dim, temperature=1.0, name=None):
  """Either argmax or random sampling.

  Args:
    x: a Tensor.
    dim: a Dimension in x.shape.dims
    temperature: a float  0.0=argmax 1.0=random
    name: an optional string

  Returns:
    a Tensor with type tf.int32.
  """
  dim = convert_to_dimension(dim)
  with tf.name_scope(name, default_name="sample_with_temperature"):
    if temperature != 0.0:
      # gumbel trick.
      # Note: we don't want to generate 0 or 1 because:
      # * -log(-log(0)) is -infinity
      # * -log(-log(1)) is +infinity.
      # The numerics may be weird in bfloat16 - use float32.
      x = cast(x, tf.float32)
      tiny_val = 1e-9
      g = -log(-log(
          random_uniform(
              x.mesh,
              x.shape,
              minval=tiny_val,
              maxval=1.,
              dtype=x.dtype)))
      x += g * temperature
    return argmax(x, dim, name)


def add(x1, x2, output_shape=None, name=None):
  """Binary addition with broadcsting.

  Args:
    x1: a Tensor
    x2: a Tensor
    output_shape: an optional Shape
    name: an optional string
  Returns:
    a Tensor
  """
  output_shape = convert_to_shape(output_shape)
  if not isinstance(x2, Tensor):
    return ScalarAddOperation(x1, x2).outputs[0]
  with tf.name_scope(name, default_name="add"):
    x1, x2 = binary_arguments_to_tensors(x1, x2)
    return AddOperation(
        x1, x2, output_shape=_infer_binary_broadcast_shape(
            x1.shape, x2.shape, output_shape)).outputs[0]


def add_n(xs):
  if not xs:
    return 0
  return functools.reduce(add, xs)


def sub(x1, x2, output_shape=None, name=None):
  """Binary subtraction with broadcsting.

  Args:
    x1: a Tensor
    x2: a Tensor
    output_shape: an optional Shape
    name: an optional string
  Returns:
    a Tensor
  """
  output_shape = convert_to_shape(output_shape)
  if not isinstance(x2, Tensor):
    return ScalarAddOperation(x1, -x2).outputs[0]
  with tf.name_scope(name, default_name="sub"):
    x1, x2 = binary_arguments_to_tensors(x1, x2)
    return add(x1, negative(x2), output_shape=output_shape)


def multiply(x1, x2, output_shape=None, name=None):
  """Binary multiplication with broadcasting.

  Args:
    x1: a Tensor
    x2: a Tensor
    output_shape: an optional Shape
    name: an optional string
  Returns:
    a Tensor
  """
  if not isinstance(x2, Tensor):
    return ScalarMultiplyOperation(x1, x2).outputs[0]
  with tf.name_scope(name, default_name="mul"):
    x1, x2 = binary_arguments_to_tensors(x1, x2)
    return einsum(
        [x1, x2],
        output_shape=_infer_binary_broadcast_shape(
            x1.shape, x2.shape, output_shape))


def divide(x1, x2, output_shape=None, name=None):
  """Binary division with broadcasting.

  Args:
    x1: a Tensor
    x2: a Tensor
    output_shape: an optional Shape
    name: an optional string
  Returns:
    a Tensor
  """
  output_shape = convert_to_shape(output_shape)
  if not isinstance(x2, Tensor):
    return ScalarMultiplyOperation(x1, 1.0 / x2).outputs[0]
  with tf.name_scope(name, default_name="divide"):
    x1, x2 = binary_arguments_to_tensors(x1, x2)
    return multiply(x1, reciprocal(x2), output_shape=output_shape)


def mtf_slice(x, begin, size, slice_dim_name, name=None):
  """Slice operation.

  Call externally as mtf.slice()

  Args:
    x: a list of Tensors
    begin: integer, where to begin slicing from along the axis
    size: integer, size to slice from axis.
    slice_dim_name: string, dimension name of slicing axis.
    name: an optional string
  Returns:
    a Tensor
  """
  return SliceOperation(
      x, begin, size, slice_dim_name, name=name).outputs[0]


def pad(x, paddings, dim_name, name=None):
  """Slice operation.

  Args:
    x: a list of Tensors
    paddings: list of integers of size 2, padding size before and after for dim.
    dim_name: string, name for the padding dim
    name: an optional string
  Returns:
    a Tensor
  """
  return PadOperation(
      x, paddings, dim_name, name=name).outputs[0]


def one_hot(indices, output_dim, on_value=1.0,
            off_value=0.0, dtype=tf.float32, name=None):
  """One hot operation.

  TODO(noam): Is there a good reason we need a special mtf.Operation here?
  We could just use some code like this:
  cast(equal(indices, mtf_range(indices.mesh, output_dim, dtype=indices.dtype)),
       dtype)

  Args:
    indices: a Tensor
    output_dim: a Dimension
    on_value: Value taken when indices are on at a location, default 1
    off_value: Value taken when indices are off at a location, default 0
    dtype: a tf.DType
    name: an optional string
  Returns:
    a Tensor with shape extended by output_dim for the last axis.
  """
  return OneHotOperation(
      indices, output_dim, on_value, off_value, dtype, name=name).outputs[0]


def gather(weights, indices, dim, output_shape=None):
  """Shorthand for einsum([one_hot(indices, dim)], weights, reduced_dims=[dim]).

  Args:
    weights: a Tensor
    indices: a Tensor with integer type
    dim: a Dimension
    output_shape: an optional mtf.Shape
  Returns:
    a Tensor
  """
  dim = convert_to_dimension(dim)
  output_shape = convert_to_shape(output_shape)
  if not isinstance(indices, Tensor):
    # TODO(noam): when `indices` is an integer, gather can be implemented
    #   more directly with mtf_slice() and reshape()
    indices = constant(weights.mesh, indices, dtype=tf.int32)
  if weights.dtype == tf.bool:
    return cast(gather(to_float(weights), indices, dim, output_shape), tf.bool)
  return einsum([one_hot(indices, dim, dtype=weights.dtype), weights],
                reduced_dims=[dim], output_shape=output_shape)


def gradients(ys, xs, grad_ys=None, operations=None):
  """Compute gradients in dtf.

  Args:
    ys: a list of Tensors
    xs: a list of Tensors
    grad_ys: an optional list of Tensors
    operations: list of operations through which to back-propagate gradients
      defaults to ys[0].graph.operations

  Returns:
    grad_xs: a list of Tensors
  """
  if operations is None:
    operations = ys[0].graph.operations
  if not grad_ys:
    grad_ys = [Constant(y.mesh, 1.0, y.shape, y.dtype).outputs[0] for y in ys]
  # figure out what Tensors are downstream of xs
  downstream = set(xs)
  for op in operations:
    if op.has_gradient:
      if set(op.inputs) & downstream:
        downstream |= set(op.outputs)
  tensor_to_gradient = {y: g for y, g in zip(ys, grad_ys) if g is not None}
  with tf.variable_scope(ys[0].graph.captured_variable_scope):
    for op in operations[::-1]:
      grad_outputs = [tensor_to_gradient.get(out) for out in op.outputs]
      if (op.has_gradient and any(grad_outputs)
          and (set(op.inputs) & downstream)):
        with tf.variable_scope(op.name + "/gradients"):
          input_grads = op.gradient(grad_outputs)
          for inp, grad in zip(op.inputs, input_grads):
            if inp in downstream and grad is not None:
              if inp in tensor_to_gradient:
                tensor_to_gradient[inp] += grad
              else:
                tensor_to_gradient[inp] = grad
  return [tensor_to_gradient.get(x, None) for x in xs]


def _infer_binary_broadcast_shape(shape1, shape2, given_output_shape=None):
  """Infer shape of the output of a binary op with broadcasting.

  If the output shape is not given with given_output_shape, then we check
  to see if one of the shapes is a subsequence of the other one, and we
  return the one that is the supersequence.  Otherwise, we list the dimensions
  of shape1, followed by all new dimensions in shape2.

  Args:
    shape1: a Shape
    shape2: a Shape
    given_output_shape: an optional Shape
  Returns:
    a Shape
  """
  shape1 = convert_to_shape(shape1)
  shape2 = convert_to_shape(shape2)
  given_output_shape = convert_to_shape(given_output_shape)
  if given_output_shape is not None:
    return given_output_shape
  if is_subsequence(shape1.dims, shape2.dims):
    return shape2
  if is_subsequence(shape2.dims, shape1.dims):
    return shape1
  return Shape(
      shape1.dims + [d for d in shape2.dims if d not in shape1.dims])


def _expand_dims(x, input_shape, output_shape):
  """Expand dimensions and transpose if necessary.

  Args:
    x: a tf.Tensor
    input_shape: a Shape
    output_shape: a Shape whose dimensions are a superset of
      those in input_shape

  Returns:
    a tf.Tensor
  """
  verify_no_new_dims([output_shape], input_shape)
  if input_shape == output_shape or input_shape.ndims == 0:
    return x
  perm = [input_shape.dims.index(d) for d in output_shape.dims
          if d in input_shape.dims]
  x = tf.transpose(x, perm)
  for i, d in enumerate(output_shape.dims):
    if d not in input_shape.dims:
      x = tf.expand_dims(x, i)
  return x


def _einsum_equation(input_shapes, output_shape):
  """Turn shapes into an einsum equation.

  e.g. "ij,jk->ik"

  Args:
    input_shapes: a list of Shapes
    output_shape: a Shape
  Returns:
    a string
  """
  ret = []
  next_letter = ord("a")
  dim_to_letter = {}
  for shape_num, shape in enumerate(input_shapes + [output_shape]):
    if shape_num == len(input_shapes):
      ret.append("->")
    elif shape_num > 0:
      ret.append(",")
    for d in shape.dims:
      if d not in dim_to_letter:
        dim_to_letter[d] = chr(next_letter)
        next_letter += 1
      ret.append(dim_to_letter[d])

  return "".join(ret)


def is_subsequence(short_seq, long_seq):
  """Is short_seq a subsequence of long_seq."""
  if not short_seq:
    return True
  pos = 0
  for x in long_seq:
    if pos == len(short_seq):
      return True
    if short_seq[pos] == x:
      pos += 1
  if pos == len(short_seq):
    return True
  return False


def verify_no_new_dims(input_shapes, output_shape):
  """Verifies that all dimensions in the output are in at least one input.

  Args:
    input_shapes: a list of Shapes
    output_shape: a Shape
  Raises:
    ValueError: if there are new dimensions in the output.
  """
  all_input_dims = set(sum([s.dims for s in input_shapes], []))
  all_output_dims = set(output_shape.dims)
  if not all_output_dims.issubset(all_input_dims):
    raise ValueError(
        "No new dimensions allowed in output"
        " input_shapes = %s output_shape= %s"
        % ([s.dims for s in input_shapes], output_shape.dims))


def pnum_to_processor_coordinates(mesh_shape, pnum):
  """Coordinates of a processor in the mesh.

  Args:
    mesh_shape: a Shape or a list of integers
    pnum: an integer less than len(mesh_shape)

  Returns:
    a list of integers with length len(mesh_shape)
  """
  if isinstance(mesh_shape, Shape):
    mesh_shape = mesh_shape.to_integer_list
  if not isinstance(mesh_shape, list):
    raise ValueError("mesh_shape must be a Shape or a list of integers")
  ret = []
  for dimsize in mesh_shape[::-1]:
    ret.append(pnum % dimsize)
    pnum //= dimsize
  return ret[::-1]


def processor_coordinates_to_pnum(mesh_shape, coord):
  """Inverse of pnum_to_processor_coordinates.

  Args:
    mesh_shape: a Shape or a list of integers
    coord: a list of integers with length len(mesh_shape)

  Returns:
    an integer less than len(mesh_shape)
  """
  if isinstance(mesh_shape, Shape):
    mesh_shape = mesh_shape.to_integer_list
  if not isinstance(mesh_shape, list):
    raise ValueError("mesh_shape must be a Shape or a list of integers")
  ret = 0
  multiplier = 1
  for c, d in zip(coord[::-1], mesh_shape[::-1]):
    ret += multiplier * c
    multiplier *= d
  return ret


def pnum_to_group(mesh_shape, group_dims, pnum):
  """Group number for grouped allreduce.

  Args:
    mesh_shape: a Shape
    group_dims: a list of integers (the dimensions reduced over)
    pnum: an integer

  Returns:
    an integer
  """
  coord = pnum_to_processor_coordinates(mesh_shape, pnum)
  remaining_shape = Shape(
      [d for i, d in enumerate(mesh_shape) if i not in group_dims])
  remaining_coord = [d for i, d in enumerate(coord) if i not in group_dims]
  return processor_coordinates_to_pnum(remaining_shape, remaining_coord)


def processor_groups(mesh_shape, group_dims):
  """Groups of processors which differ only in the given dimensions.

  Args:
    mesh_shape: a Shape
    group_dims: a list of integers

  Returns:
    a list of lists of integers (processor numbers)
  """
  group_numbers = [
      pnum_to_group(mesh_shape, group_dims, pnum)
      for pnum in xrange(mesh_shape.size)]
  ret = []
  for pnum, g in enumerate(group_numbers):
    while len(ret) <= g:
      ret.append([])
    ret[g].append(pnum)
  return ret


def list_product(l):
  return functools.reduce(operator.mul, l, 1)


def reduce_logsumexp(x, reduced_dim, extra_logit=None, name=None):
  """Numerically stable version of log(reduce_sum(exp(x))).

  Unlike other reductions, the output has the same shape as the input.
  Note: with a minor change, we could allow multiple reduced dimensions.

  Args:
    x: a Tensor
    reduced_dim: a dimension in x
    extra_logit: an optional Tensor broadcastable to (x.shape - reduced_dim)
    name: an optional string
  Returns:
    a Tensor with the same shape and dtype as x.
  """
  reduced_dim = convert_to_dimension(reduced_dim)
  with tf.variable_scope(name, default_name="reduce_logsumexp"):
    reduced_shape = x.shape - reduced_dim
    max_logit = reduce_max(stop_gradient(x), output_shape=reduced_shape)
    if extra_logit is not None:
      if isinstance(extra_logit, Tensor):
        extra_logit = stop_gradient(extra_logit)
      max_logit = maximum(max_logit, extra_logit)
    x -= max_logit
    exp_x = exp(x)
    sum_exp_x = reduce_sum(exp_x, output_shape=reduced_shape)
    if extra_logit is not None:
      sum_exp_x += exp(extra_logit - max_logit)
    return log(sum_exp_x) + max_logit


def log_softmax(x, reduced_dim, extra_logit=None, name=None):
  """log(softmax(x)).

  Args:
    x: a Tensor whose shape contains vocab_dim
    reduced_dim: a Dimension
    extra_logit: an optional Tensor broadcastable to (x.shape - reduced_dim)
    name: an optional string

  Returns:
    a Tensor with the same shape as x
  """
  return x - reduce_logsumexp(
      x, reduced_dim, extra_logit=extra_logit, name=name)


def softmax(x, reduced_dim, extra_logit=None, name=None):
  with tf.variable_scope(name, default_name="softmax"):
    return exp(log_softmax(x, reduced_dim, extra_logit=extra_logit))


class RangeOperation(Operation):
  """tf.range."""

  def __init__(self, mesh, dim, dtype, name=None):
    super(RangeOperation, self).__init__([], mesh, name=name or "range")
    dim = convert_to_dimension(dim)
    self._mesh = mesh
    self._dim = dim
    self._dtype = dtype

    self._outputs = [Tensor(self, Shape([dim]), dtype)]

  def lower(self, lowering):
    mesh_impl = lowering.mesh_impl(self)
    with tf.variable_scope(self.name, default_name="range"):
      if self._dtype == tf.bfloat16:
        # tf.range(dtype=bfloat16) gives the wrong shape.
        # TODO(noam): report the bug.
        tf_range = tf.cast(tf.range(self._dim.size), tf.bfloat16)
      else:
        tf_range = tf.range(self._dim.size, dtype=self._dtype)
      lowering.set_tensor_lowering(
          self.outputs[0],
          mesh_impl.import_tf_tensor(self.outputs[0], tf_range))


def mtf_range(mesh, dim, dtype, name=None):
  """Create a 1d mesh tensor with a range from [0, dim.size).

  Call externally as mtf.range()

  Args:
    mesh: a Mesh
    dim: a Dimension
    dtype: a tf.DType
    name: an optional string

  Returns:
    a Tensor
  """
  return RangeOperation(mesh, dim, dtype, name).outputs[0]


def pretty_print_counters(counters):
  """print counters hierarchically.

  Each counter is a pair of a string and a number.
  The string can have slashes, meaning that the number also counts towards
  each prefix.  e.g.  "parameters/trainable" counts towards both "parameters"
  and "parameters/trainable".

  Args:
    counters: a list of (string, number) pairs

  Returns:
    a string
  """
  totals = collections.defaultdict(int)
  for (name, val) in counters:
    prefixes = [name[:i] for i in xrange(len(name)) if name[i] == "/"] + [name]
    for p in prefixes:
      totals[p] += val
  parts = []
  for name, val in sorted(six.iteritems(totals)):
    parts.append(" " * name.count("/") + "%s: %.3g" % (name, val))
  return "\n".join(parts)


def _parse_string_to_list_of_pairs(s, seconds_to_int=False):
  r"""Parses a string into a list of pairs.

  In the input string, each pair is separated by a colon, and the delimiters
  between pairs are any of " ,.;".

  e.g. "rows:32,cols:32"

  Args:
    s: str to parse.
    seconds_to_int: Boolean. If True, then the second elements are returned
      as integers;  otherwise they are strings.

  Returns:
    List of tuple pairs.

  Raises:
    ValueError: Badly formatted string.
  """
  ret = []
  for p in [s.split(":") for s in re.sub("[,.;]", " ", s).split()]:
    if len(p) != 2:
      raise ValueError("bad input to _parse_string_to_list_of_pairs %s" % s)
    if seconds_to_int:
      ret.append((p[0], int(p[1])))
    else:
      ret.append(tuple(p))
  return ret


def parallel(devices, fn, *args, **kwargs):
  """Call a function once on each device.

  Args:
    devices: a list of n devices
    fn: a function
    *args: arguments, each of which is a list of length n
    **kwargs: keyword-args, each of which is a list of length n
  Returns:
    a list of length n
  Raises:
    ValueError: if the arguments are not all lists of length n
  """
  if not isinstance(devices, list):
    raise ValueError("devices must be a list")
  for x in list(args) + list(six.itervalues(kwargs)):
    if not isinstance(x, list) or len(x) != len(devices):
      raise ValueError(
          "Argument not a list with same length as devices "
          "arg=%s devices=%s" % (x, devices))
  ret = []
  for i, device in enumerate(devices):
    with tf.device(device):
      with tf.variable_scope("parallel_%d" % i):
        my_args = [x[i] for x in args]
        my_kwargs = {k: v[i] for k, v in six.iteritems(kwargs)}
        ret.append(fn(*my_args, **my_kwargs))
  return ret


def transpose_list_of_lists(lol):
  """Transpose a list of equally-sized python lists.

  Args:
    lol: a list of lists
  Returns:
    a list of lists
  Raises:
    ValueError: if list is empty
  """
  if not lol:
    raise ValueError("cannot transpose the empty list")
  return [list(x) for x in zip(*lol)]


def binary_reduction_fn(reduction_fn_string):
  if reduction_fn_string == "SUM":
    return tf.add
  elif reduction_fn_string == "MAX":
    return tf.maximum
  elif reduction_fn_string == "MIN":
    return tf.minimum
  else:
    raise ValueError("Unknown reduction_fn_string %s" % reduction_fn_string)


def reduction_fn(reduction_fn_string):
  if reduction_fn_string == "SUM":
    return tf.reduce_sum
  elif reduction_fn_string == "MAX":
    return tf.reduce_max
  elif reduction_fn_string == "MIN":
    return tf.reduce_min
  else:
    raise ValueError("Unknown reduction_fn_string %s" % reduction_fn_string)


def pool_fn(pool_fn_string):
  """Converts a string function name to actual function."""
  def avg_pool2d_fn(x, ksize, strides, padding):
    return _tf_restore_batch_dims(
        tf.nn.avg_pool2d(_tf_flatten_batch_dims(x, 3), ksize, strides, padding),
        3, x)
  def avg_pool3d_fn(x, ksize, strides, padding):
    return _tf_restore_batch_dims(
        tf.nn.avg_pool3d(_tf_flatten_batch_dims(x, 4), ksize, strides, padding),
        4, x)
  def max_pool2d_fn(x, ksize, strides, padding):
    return _tf_restore_batch_dims(
        tf.nn.max_pool2d(_tf_flatten_batch_dims(x, 3), ksize, strides, padding),
        3, x)
  def max_pool3d_fn(x, ksize, strides, padding):
    return _tf_restore_batch_dims(
        tf.nn.max_pool3d(_tf_flatten_batch_dims(x, 4), ksize, strides, padding),
        4, x)

  if pool_fn_string == "AVG_2D":
    return avg_pool2d_fn
  elif pool_fn_string == "AVG_3D":
    return avg_pool3d_fn
  elif pool_fn_string == "MAX_2D":
    return max_pool2d_fn
  elif pool_fn_string == "MAX_3D":
    return max_pool3d_fn
  else:
    raise ValueError("Unknown pool_fn_string %s" % pool_fn_string)


class MtfCheckpointSaverListener(tf.estimator.CheckpointSaverListener):
  """Copy slices to masters before saving."""

  def __init__(self, lowering):
    self._op = lowering.copy_slices_to_masters()

  def begin(self):
    # You can add ops to the graph here.
    tf.logging.info("Starting the session.")

  def before_save(self, session, global_step_value):
    # assigns
    tf.logging.info("Before Save.")
    session.run(self._op)
    tf.logging.info("About to write a checkpoint")

  def after_save(self, session, global_step_value):
    tf.logging.info("Done writing checkpoint.")

  def end(self, session, global_step_value):
    tf.logging.info("Done with the session.")


class MtfRestoreHook(tf.estimator.SessionRunHook):
  """Copy masters to slices after restoring."""

  def __init__(self, lowering):
    self._lowering = lowering

  def begin(self):
    # This namescope is useful in adding the hook operation when the graph is
    # constructed. It's also necessary to call the op when the exported model is
    # loaded in another session.
    with tf.name_scope("mtf_restore_hook"):
      self._op = self._lowering.copy_masters_to_slices()

  def after_create_session(self, session, coord):
    tf.logging.info("Before copy master to slices.")
    session.run(self._op)
    tf.logging.info("Done with copy master to slices.")


class RandomOperation(Operation):
  """Random operation such as tf.random.uniform."""

  def __init__(self, mesh, shape, tf_fn, **kwargs):
    super(RandomOperation, self).__init__(
        [], mesh=mesh, name=kwargs.get("name", "random"))
    self._tf_fn = tf_fn
    self._kwargs = kwargs
    self._outputs = [Tensor(self, shape, kwargs.get("dtype", tf.float32))]
    # Rerun to take the new output into account.
    self._splittable_dims, self._unsplittable_dims = (
        self._initialize_all_dimensions_as_splittable())

  def lower(self, lowering):
    mesh_impl = lowering.mesh_impl(self)
    output_shape = self.outputs[0].shape
    lowering.set_tensor_lowering(self.outputs[0], (
        mesh_impl.random(output_shape, self._tf_fn, self._kwargs)))


def random_uniform(mesh, shape, **kwargs):
  """Random uniform.

  Args:
    mesh: a Mesh
    shape: a Shape
    **kwargs: keyword args for tf.random.uniform, except seed

  Returns:
    a Tensor
  """
  shape = convert_to_shape(shape)
  return RandomOperation(mesh, shape, tf.random.uniform, **kwargs).outputs[0]


def random_normal(mesh, shape, **kwargs):
  """Random uniform.

  Args:
    mesh: a Mesh
    shape: a Shape
    **kwargs: keyword args for tf.random.normal, except seed

  Returns:
    a Tensor
  """
  shape = convert_to_shape(shape)
  return RandomOperation(mesh, shape, tf.random.normal, **kwargs).outputs[0]


def dropout(x, keep_prob=None, rate=None, noise_shape=None, name=None):
  """Randomly set some elements to 0 and scale up the rest.

  Dropout rate should be specified in exactly one of two ways:
    rate - the fraction to drop
    keep_prob - the fraction to keep

  If x has floating-point type, then kept values are scaled up by
  a factor of (1.0 / keep_prob).  If x is has integer type, the kept values
  are not scaled up.

  Args:
    x: a Tensor
    keep_prob: a float between 0.0 and 1.0
    rate: a float between 0.0 and 1.0
    noise_shape: an optional Shape (a subset of x.shape)
    name: an optional string

  Returns:
    a Tensor
  """
  if (keep_prob is None) == (rate is None):
    raise ValueError("exactly one of keep_prob and rate should be set")
  if keep_prob is None:
    keep_prob = 1.0 - rate
  noise_shape = convert_to_shape(noise_shape)
  if noise_shape is None:
    noise_shape = x.shape
  with tf.variable_scope(name, default_name="dropout"):
    if keep_prob == 1.0:
      return x
    noise = cast(less(random_uniform(
        x.mesh, noise_shape,
        dtype=(x.dtype if x.dtype.is_floating else tf.float32)),
                      keep_prob), x.dtype)
    if x.dtype.is_floating:
      noise /= keep_prob
    return x * noise


def _cumprod(l):
  """Cumulative product of a list.

  Args:
    l: a list of integers
  Returns:
    a list with one more element (starting with 1)
  """
  ret = [1]
  for item in l:
    ret.append(ret[-1] * item)
  return ret


def log_variable_sizes(var_list,
                       tag,
                       verbose=True,
                       mesh_to_impl=None,
                       log_file=None):
  """Log the sizes and shapes of variables, and the total size.

  Args:
    var_list: a list of variables; defaults to trainable_variables
    tag: a string; defaults to "Trainable Variables"
    verbose: bool, if True, log every weight; otherwise, log total size only.
    mesh_to_impl: an optional map from Mesh to MeshImpl
    log_file: an optional tf.io.gfile.GFile. If provided, information about
      the variables will also be logged to this file.
  """
  if not var_list:
    return

  name_to_var = {v.name: v for v in var_list}
  total_size = 0
  total_slice_size = 0
  for v_name in sorted(list(name_to_var)):
    v = name_to_var[v_name]
    v_size = v.shape.size
    if mesh_to_impl is not None:
      slice_size = mesh_to_impl[v.mesh].slice_size(v.shape)
    else:
      slice_size = 0
    total_slice_size += slice_size
    if verbose:
      _log_info_also_to_file(
          "Variable %s size %s slice_size %s %s",
          v.name.ljust(60),
          str(v_size).ljust(12),
          str(slice_size).ljust(12),
          str(v.shape).ljust(60),
          log_file=log_file)
      if isinstance(v, StackedVariable):
        for n in v.original_names:
          _log_info_also_to_file("    " + n, log_file=log_file)
    total_size += v_size
  _log_info_also_to_file(
      "%s count: %s  Total size: %s  Total slice_size: %s",
      tag.ljust(30),
      str(len(var_list)).ljust(6),
      str(total_size).ljust(15),
      str(total_slice_size).ljust(15),
      log_file=log_file)


def _log_info_also_to_file(format_str, *args, **kw_args):
  """Logs at the info level and writes to file if one is provided.

  Args:
    format_str: a string; will be logged and can contain things such as %s.
    *args: arguments to the format_str.
    **kw_args: keyword arguments. May contain optional tf.io.gfile.GFile keyed
      by "log_file", where the message will also be appended to this file. Other
      arguments will be ignored.
  """
  tf.logging.info(format_str, *args)
  log_file = kw_args.get("log_file", None)
  if log_file:
    log_file.write(format_str % args)
    log_file.write("\n")


class WhileLoopOperation(Operation):
  """While loop, like tf.while_loop."""

  def __init__(self, cond_fn, body_fn, inputs,
               tf_kwargs=None, has_accumulators=False, name="while_loop"):
    """Create a WhileLoopOperation.

    A few differences from tf.while_loop:

    - gradients are not yet supported

    - inputs must be a list of tensors, as opposed to an arbitrary nested
      structure.  cond_fn and body_fn take an argument list

    - we support optional "accumulators" which are additional outputs
      returned by body_fn.  These are summed across all iterations and
      retured as additional outputs of the while-loop.  To use accumulators,
      the has_accumulators argument must be True.  For better performance,
      we delay allreduce on the accumulators until after the loop, so that it
      only needs to happen once.  This is useful, for example, if the
      accumulators are summing gradients for many mini-batches.

    Args:
      cond_fn: a function from n mtf Tensors to mtf Scalar
      body_fn: a function from n mtf Tensors to sequence of mtf Tensors
      inputs: list of n mtf Tensors
      tf_kwargs: a dictionary of arguments for tf.while_loop
      has_accumulators: a boolean
      name: a string
    Returns:
      a WhileLoopOperation
    """

    super(WhileLoopOperation, self).__init__(
        inputs, mesh=inputs[0].mesh, name=name)
    self._cond_fn = cond_fn
    self._body_fn = body_fn
    self._tf_kwargs = tf_kwargs or {}
    assert not self._tf_kwargs.get("back_prop", False)
    ops = self.graph.operations
    # remove self from the graph's operations
    ops.pop()
    before = len(ops)
    def make_placeholders(name):
      return [Tensor(self, t.shape, t.dtype, name="%s:%d" % (name, i))
              for i, t in enumerate(inputs)]
    self._cond_inputs = make_placeholders("cond_input")
    self._cond_output = self._cond_fn(*self._cond_inputs)
    self._cond_ops = ops[before:]
    del ops[before:]
    self._body_inputs = make_placeholders("body_input")
    self._body_outputs = self._body_fn(*self._body_inputs)
    if len(self._body_outputs) < len(inputs):
      raise ValueError("body_fn produces fewer outputs than inputs")
    if len(self._body_outputs) > len(inputs) and not has_accumulators:
      raise ValueError("body_fn produces more outputs than inputs")
    for (i, (inp, body_out)) in enumerate(
        zip(inputs, self._body_outputs[:len(inputs)])):
      if inp.shape != body_out.shape:
        raise ValueError(
            "shape mismatch i=%d inp=%s body_out=%s" % (i, inp, body_out))
    # Pull new variables outside the loop.
    added_ops = ops[before:]
    del ops[before:]
    self._body_ops = []
    for op in added_ops:
      if isinstance(op, Variable):
        ops.append(op)
      else:
        self._body_ops.append(op)
    # re-add self to graph's operations
    ops.append(self)
    self._outputs = [
        Tensor(self, t.shape, t.dtype, name="output:%d" % i)
        for i, t in enumerate(self._body_outputs)]

    # Rerun to take the new output into account.
    self._splittable_dims, self._unsplittable_dims = (
        self._initialize_all_dimensions_as_splittable())

  def lower(self, lowering):
    mesh_impl = lowering.mesh_impl(self)
    def tf_cond_fn(*tf_inputs):
      for tf_inp, mtf_inp in zip(
          tf_inputs[:len(self._cond_inputs)], self._cond_inputs):
        lowering.tensors[mtf_inp] = mesh_impl.LaidOutTensor(tf_inp)
      for op in self._cond_ops:
        with tf.name_scope(op.name):
          op.lower(lowering)
      lowered_output = lowering.tensors[self._cond_output]
      ret = lowered_output.to_laid_out_tensor().tensor_list[0]
      return ret

    # This array keeps track of which lowered body-outputs have type
    # LazyAllreduceSum.  We treat these specially  - instead of
    # immediately converting to LaidOutTensor (executing the allreduce)
    # we sum across iterations first, then allreduce at the end.
    # When one of the body outputs is a LazyAllreduceSum, we put the
    #  LazyAllreduceSum object into this array for future reference.
    is_lazyallreducesum = [None] * len(self._outputs)
    def tf_body_fn(*tf_inputs):
      """Body function for tf.while_loop.

      Args:
        *tf_inputs: a list of tf.Tensor
      Returns:
        a list of tf.Tensor
      """
      for tf_inp, mtf_inp in zip(
          tf_inputs[:len(self._inputs)], self._body_inputs):
        lowering.tensors[mtf_inp] = mesh_impl.LaidOutTensor(tf_inp)
      for op in self._body_ops:
        with tf.name_scope(op.name):
          op.lower(lowering)
      ret = []
      for i, mtf_out in enumerate(self._body_outputs):
        lowered_out = lowering.tensors[mtf_out]
        if isinstance(lowered_out, LazyAllreduceSum):
          is_lazyallreducesum[i] = lowered_out
          ret.append(lowered_out.laid_out_input.tensor_list)
        else:
          ret.append(lowered_out.to_laid_out_tensor().tensor_list)
      # accumulators
      for i in range(len(self._inputs), len(self._outputs)):
        ret[i] = [x + y for x, y in zip(ret[i], tf_inputs[i])]
      return ret

    lowered_inputs = []
    for t in self.inputs:
      lowered_inputs.append(
          lowering.tensors[t].to_laid_out_tensor().tensor_list)
    # accumulators get initial value 0
    for t in self._body_outputs[len(self.inputs):]:
      def slice_fn():
        return tf.zeros(mesh_impl.slice_shape(t.shape), dtype=t.dtype)
      lowered_inputs.append(mesh_impl.slicewise(slice_fn).tensor_list)

    tf_outs = tf.while_loop(tf_cond_fn,
                            tf_body_fn,
                            lowered_inputs,
                            back_prop=False,
                            **self._tf_kwargs)
    for i, (tf_out, mtf_out) in enumerate(zip(tf_outs, self._outputs)):
      out = mesh_impl.LaidOutTensor(tf_out)
      lazy = is_lazyallreducesum[i]
      if lazy:
        out = LazyAllreduceSum(
            mesh_impl, out, lazy.mesh_axes, lazy.add_counter_fn)
      lowering.set_tensor_lowering(mtf_out, out)


def while_loop(cond_fn, body_fn, inputs, num_loop_vars=None,
               has_accumulators=False, **kwargs):
  """While Loop.

  See comments above for WhileLoopOperation

  num_loop_vars is a hack for the multi-gpu setup.  In this case, loops
  are generally slow, as all loop variables are placed on device.  By setting
  num_loop_vars=k, then all of the loop variables except for the first k
  are handled as mtf Variables instead of loop variables, using explicit
  updates and control dependencies.  In this case, we only return the
  first num_loop_vars outputs.  Do not use this option on TPU, since it
  is unnecessary and also produces incorrect results, since xla does not
  respect control dependencies.

  Args:
    cond_fn: a function from n Tensors to scalar boolean Tensor
    body_fn: a function from n Tensors to list of n Tensors
    inputs: a list of n Tensors
    num_loop_vars: an optional integer.
    has_accumulators: a boolean
    **kwargs: additional kwargs passed to tf.while_loop

  Returns:
    a list of n Tensors.
  """
  if num_loop_vars is None:
    return WhileLoopOperation(cond_fn, body_fn, inputs, tf_kwargs=kwargs,
                              has_accumulators=has_accumulators).outputs
  # Turn all loop vars except for the first ones into non-loop vars.
  # see comments in docstring.
  assert num_loop_vars > 0
  extra_inputs = inputs[num_loop_vars:]
  my_vars = []
  for i, x in enumerate(extra_inputs):
    my_vars.append(get_variable(
        x.mesh, "loop_var_%d" % i,
        x.shape, initializer=tf.zeros_initializer(),
        dtype=x.dtype,
        collections=[tf.GraphKeys.LOCAL_VARIABLES]))
  my_vars = tuple(my_vars)
  first_input = depend(
      inputs[0], [assign(var, x) for var, x in zip(my_vars, extra_inputs)])
  inputs = [first_input] + inputs[1:num_loop_vars]
  def my_cond_fn(*inputs):
    return cond_fn(*(inputs + my_vars))
  def my_body_fn(*inputs):
    outputs = tuple(body_fn(*(inputs + my_vars)))
    extra_outputs = outputs[num_loop_vars:]
    first_output = depend(
        outputs[0], [assign(var, x) for var, x in zip(my_vars, extra_outputs)])
    outputs = (first_output,) + outputs[1:num_loop_vars]
    return outputs
  return WhileLoopOperation(
      my_cond_fn, my_body_fn, inputs, tf_kwargs=kwargs,
      has_accumulators=has_accumulators).outputs


class CustomGradientOperation(Operation):
  """Operation to implement custom gradients.

  See comments on custom_gradient() below.
  """

  def __init__(self,
               explicit_inputs,
               all_inputs,
               fn_outputs,
               grad_fn,
               forward_operations,
               name=None):
    super(CustomGradientOperation, self).__init__(
        all_inputs + fn_outputs, name=name or "custom_gradient")
    self._explicit_inputs = explicit_inputs
    self._all_inputs = all_inputs
    self._grad_fn = grad_fn
    self._fn_outputs = fn_outputs
    self._outputs = [Tensor(self, x.shape, x.dtype, index=i)
                     for i, x in enumerate(fn_outputs)]
    self._forward_operations = forward_operations

  def lower(self, lowering):
    for fn_output, output in zip(
        self._fn_outputs, self._outputs):
      lowering.set_tensor_lowering(output,
                                   lowering.tensors[fn_output])

  def gradient(self, grad_ys):
    graph = self._inputs[0].graph
    old_num_vars = len(graph.all_variables)
    grads = self._grad_fn(
        explicit_inputs=self._explicit_inputs,
        all_inputs=self._all_inputs,
        forward_operations=self._forward_operations,
        outputs=self._fn_outputs,
        output_grads=grad_ys)
    new_num_vars = len(graph.all_variables)
    if new_num_vars != old_num_vars:
      raise ValueError(
          "new variables created by custom gradient."
          "Maybe a problem with scope. %s" % (
              graph.all_variables[old_num_vars:],))
    for g, t in zip(grads, self._all_inputs):
      if g is None:
        tf.logging.warning("No gradient on input %s" % t)
    return list(grads) + [None] * len(self._fn_outputs)


def custom_gradient(fn, grad_fn, explicit_inputs):
  """Execute a function and call a custom gradient fn on the backward pass.

  `fn` takes positional Tensor arguments and returns a Tensor or a tuple of
  Tensors.

  `explicit_inputs` is a list of tensors to be passed as positional arguments
  to the function `fn`.

  `grad_fn` has the following signature:
    Args:
      explicit_inputs: the list of Tensors passed to this function and to `fn`
      all_inputs: a list of tensors beginning with explicit_inputs, but also
        containing external Tensors used by fn.
      forward_operations: a list of Operation. (the operations created on the
        foward pass
      outputs: the outputs of `fn` from the forward pass
      output_grads: the gradient Tensors corresponding to those outputs.
    Returns
      a list of Tensor/None with the same length as `all_inputs`

  Args:
    fn: a function taking positional Tensor arguments
    grad_fn: a function (see above)
    explicit_inputs: list of Tensors
  Returns:
    a list of outputs
  """
  graph = explicit_inputs[0].graph
  outputs, forward_operations = graph.capture_operations(
      lambda: fn(*explicit_inputs))
  returns_tuple = isinstance(outputs, tuple)
  new_outputs = set()
  new_inputs = set()
  for op in forward_operations:
    new_inputs.update(set(op.inputs))
    if not isinstance(op, Variable):
      new_outputs.update(set(op.outputs))
  external_inputs = list(new_inputs - new_outputs - set(explicit_inputs))
  external_inputs = [t for t in external_inputs if t.dtype.is_floating]
  all_inputs = explicit_inputs + external_inputs
  if not returns_tuple:
    outputs = outputs,
  ret = CustomGradientOperation(explicit_inputs,
                                all_inputs,
                                list(outputs),
                                grad_fn,
                                forward_operations).outputs
  # Make sure no one uses the internals of this function, since the gradients
  #  will probably not work correctly.
  for t in new_outputs - set(outputs):
    t.usable = False
  return ret if returns_tuple else ret[0]


def _recompute_grad_grad(explicit_inputs,
                         all_inputs,
                         forward_operations,
                         outputs,
                         output_grads,
                         control_dependencies):
  """Gradient function used with recompute_grad."""
  graph = forward_operations[0].graph
  input_mapping = {t: t for t in all_inputs}
  if control_dependencies:
    # we need to outsmart XLA here to force a control dependency
    zero_with_control_dependency = reduce_sum(output_grads[0] * 1e-30)
    for t in explicit_inputs:
      if t.dtype.is_floating:
        input_mapping[t] += cast(zero_with_control_dependency, t.dtype)
  mapped_inputs = [input_mapping[t] for t in all_inputs]
  recomputed_operations, mapping = graph.clone_operations(
      forward_operations, input_mapping)
  recomputed_outputs = [mapping[t] for t in outputs]
  input_grads = gradients(
      ys=recomputed_outputs,
      xs=mapped_inputs,
      grad_ys=output_grads,
      operations=recomputed_operations)
  for x, g in zip(all_inputs, input_grads):
    if x.dtype.is_floating and g is None:
      raise ValueError("_recompute_grad_grad: no gradient for %s" % x)
  return input_grads


def recompute_grad(fn, explicit_inputs, control_dependencies=True):
  """Execute a function and recompute it on the backwards pass.

  Args:
    fn: a function taking positional arguments and returning a Tensor or tuple
      of Tensors.
    explicit_inputs: inputs to the function
    control_dependencies: a boolean - whether to force the recomputation to
      happen after the output gradients.
  Returns:
    a Tensor or tuple of Tensors
  """
  return custom_gradient(
      fn,
      functools.partial(_recompute_grad_grad,
                        control_dependencies=control_dependencies),
      explicit_inputs)


def where(condition, if_true, if_false, output_shape=None):
  dtype = if_true.dtype
  return (
      multiply(if_true, cast(condition, dtype), output_shape=output_shape) +
      multiply(if_false,
               cast(logical_not(condition), dtype), output_shape=output_shape))


def _shape_union(shapes):
  """A shape containing the union of all dimensions in the input shapes.

  Args:
    shapes: a list of Shapes

  Returns:
    a Shape
  """
  return Shape(sorted(list(set(sum([s.dims for s in shapes], [])))))


def _tf_flatten_batch_dims(x, num_nonbatch_dims):
  """Flatten all but last num_nonbatch_dims into one dimension.

  Args:
    x: a tf.Tensor:
    num_nonbatch_dims: an integer

  Returns:
    a tf.Tensor with 1 + num_nonbatch_dims dimensions.
  """
  shape = x.shape.as_list()
  assert None not in shape
  new_shape = ([list_product(shape[:-num_nonbatch_dims])]
               + shape[-num_nonbatch_dims:])
  if new_shape != shape:
    x = tf.reshape(x, new_shape)
  return x


def _tf_restore_batch_dims(x, num_nonbatch_dims, prototype):
  """Reverse op of _tf_flatten_batch_dims.

  Un-flatten the first dimension of x to match all but the last
  num_nonbatch_dims dimensions of prototype.

  Args:
    x: a tf.Tensor with 1 + num_nonbatch_dims dimensions
    num_nonbatch_dims: an integer
    prototype: a tf.Tensor

  Returns:
    a tf.Tensor
  """
  assert x.shape.ndims == 1 + num_nonbatch_dims
  new_shape = (
      prototype.shape.as_list()[:-num_nonbatch_dims] + x.shape.as_list()[1:])
  assert None not in new_shape
  if new_shape != x.shape.as_list():
    x = tf.reshape(x, new_shape)
  return x


def halo_exchange(x, blocks_dim, block_size_dim, halo_size, wrap=False):
  """Concat each block with the margins of adjacent blocks.

  Get left and right blocks_dim and concatenate along block_size_dim.

  Args:
    x: a Tensor.
    blocks_dim: a Dimension in x.shape
    block_size_dim: a Dimension in x.shape
    halo_size: an integer
    wrap: a boolean

  Returns:
    a Tensor with the same shape as x, other than in block_size_dim, whose
    size is increased by 2*halo_size.
  """
  if halo_size == 0:
    return x

  block_size = block_size_dim.size
  partial_size = halo_size % block_size
  num_complete_blocks = halo_size // block_size
  parts = [x]

  for i in xrange(1, num_complete_blocks + 1):
    parts = ([shift(x, i, blocks_dim, wrap)] + parts +
             [shift(x, -i, blocks_dim, wrap)])
  if partial_size > 0:
    left_margin = mtf_slice(x, 0, partial_size, block_size_dim.name)
    right_margin = mtf_slice(
        x, block_size_dim.size - partial_size, partial_size,
        block_size_dim.name)
    parts = (
        [shift(right_margin, num_complete_blocks + 1, blocks_dim, wrap)]
        + parts +
        [shift(left_margin, -(num_complete_blocks + 1), blocks_dim, wrap)])
  return concat(parts, block_size_dim.name)


def left_halo_exchange(x, blocks_dim, block_size_dim, halo_size, wrap=False):
  """Concat each block with the margins of adjacent blocks from the left.

  Get left blocks_dim and concatenate along block_size_dim.

  Args:
    x: a Tensor.
    blocks_dim: a Dimension in x.shape
    block_size_dim: a Dimension in x.shape
    halo_size: an integer
    wrap: a boolean

  Returns:
    a Tensor with the same shape as x, other than in block_size_dim, whose
    size is increased by halo_size.
  """
  if halo_size == 0:
    return x

  block_size = block_size_dim.size
  partial_size = halo_size % block_size
  num_complete_blocks = halo_size // block_size
  parts = [x]

  for i in xrange(1, num_complete_blocks + 1):
    parts = ([shift(x, i, blocks_dim, wrap)] + parts)
  if partial_size > 0:
    right_margin = mtf_slice(
        x, block_size_dim.size - partial_size, partial_size,
        block_size_dim.name)
    parts = ([shift(right_margin, num_complete_blocks + 1, blocks_dim, wrap)]
             + parts)
  return concat(parts, block_size_dim.name)


def tensor_dim_to_mesh_dim_size(layout, mesh_shape, tensor_dim):
  """How many ways does a tensor dimension get split.

  This is used to "cheat" when building the mtf graph and peek at how a
  tensor dimension will be split.  Returns 1 if the tensor dimension is not
  split.

  Args:
    layout: an input to convert_to_layout_rules
    mesh_shape: an input to convert_to_shape
    tensor_dim: a Dimension

  Returns:
    an integer
  """
  layout_rules = convert_to_layout_rules(layout)
  mesh_shape = convert_to_shape(mesh_shape)
  mesh_axis = layout_rules.tensor_dimension_to_mesh_axis(tensor_dim, mesh_shape)
  if mesh_axis is None:
    return 1
  else:
    return mesh_shape.dims[mesh_axis].size


def tensor_dim_to_size_per_split(layout, mesh_shape, tensor_dim):
  mesh_dim_size = tensor_dim_to_mesh_dim_size(layout, mesh_shape, tensor_dim)
  if tensor_dim.size % mesh_dim_size:
    raise ValueError("Mesh dimension (%s) must divide tensor dimension (%s)"
                     % (mesh_dim_size, tensor_dim))
  return tensor_dim.size // mesh_dim_size


def combined_dimension(dims, name=None):
  if not dims:
    raise ValueError("dims must be a list of one or more Dimensions")
  return Dimension(name or dims[0].name, Shape(dims).size)


def serialize_training_step(features, model_fn, batch_dim, num_splits):
  """Break the training batch into multiple microbatches.

  Returns two structures:

  grads - a list of Tensors corresponding to the gradients on
     graph.trainable_variables.  These are summed across all microbatches

  outputs - a dictionary of Tensors corresponding to the output dictionary of
     model_fn.   Each value is either summed across all microbatches (if it
     has no batch-dimension), or concatenated across all microbatches to
     represent the original batch (if it does have a batch-dimension).

  Args:
    features: a dictionary of Tensors, each with a batch_dim dimension
    model_fn: a function from feature dictionary to output dictionary
      output_dictionary must contain "loss"
    batch_dim: a Dimension
    num_splits: an integer dividing batch_dim.size

  Returns:
    grads: a list of Tensors corresponding to the gradients on
      graph.trainable_variables
    outputs: dictionary of output Tensors summed across microbatches
  """
  for v in features.values():
    mesh = v.mesh
    graph = v.graph
  microbatch_dim = Dimension("microbatch", num_splits)
  smaller_batch_dim = Dimension(batch_dim.name, batch_dim.size // num_splits)
  cache = {}
  def select(t, microbatch_num):
    return gather(
        replace_dimensions(t, batch_dim, [smaller_batch_dim, microbatch_dim]),
        microbatch_num, microbatch_dim)
  def cond_fn(microbatch_num):
    return less(microbatch_num, num_splits)
  def body_fn(microbatch_num):
    """Body function for mtf.while_loop.

    Args:
      microbatch_num: a mtf Scalar
    Returns:
      a list of mtf Tensors
    """
    my_features = {}
    for k, v in six.iteritems(features):
      my_features[k] = select(v, microbatch_num)
    outputs = model_fn(my_features)
    grads = gradients(
        [outputs["loss"]], [v.outputs[0] for v in graph.trainable_variables])
    if None in grads:
      for var, var_grad in zip(graph.trainable_variables, grads):
        if var_grad is None:
          tf.logging.error(
              "None gradient for trainable variable %s." % var.outputs[0])
      raise ValueError("Fond trainable variable(s) with None gradient. "
                       "Check if there are trainable variables(s) "
                       "disconnected from the graph.")
    output_keys = outputs.keys()
    cache["output_keys"] = output_keys
    ret = []
    ret.append(microbatch_num + 1)
    # The rest of the returned values are "accumulators" that get summed
    # across all microbatches.
    for t in outputs.values():
      if smaller_batch_dim in t.shape:
        # The output contains a batch dimension, so we want to concatenate
        # across microbatches.
        # Here we pad the tensor for each microbatch - summing will complete
        #  the concatenation.
        t = einsum(
            [t, one_hot(microbatch_num, microbatch_dim, dtype=t.dtype)],
            output_shape=replace_dimensions(
                t.shape, smaller_batch_dim,
                [smaller_batch_dim, microbatch_dim]))
        t = replace_dimensions(
            t, [smaller_batch_dim, microbatch_dim], batch_dim)
        ret.append(t)
      else:
        # There is no batch dimension.  Sum across all microbatches.
        ret.append(t)
    # we also want to sum the gradients.
    ret.extend(grads)
    return ret
  while_out = while_loop(
      cond_fn, body_fn, [constant(mesh, 0, dtype=tf.int32)],
      has_accumulators=True)
  num_outputs = len(cache["output_keys"])
  combined_outputs = {}
  for k, v in zip(cache["output_keys"], while_out[1:1 + num_outputs]):
    combined_outputs[k] = v
  combined_grads = while_out[1 + num_outputs:]
  return combined_grads, combined_outputs


def nth_largest_element(x, n, reduced_dim, name=None):
  """Nth-largest reduction on specified axis.

  Note that n is zero-indexed.

  Args:
    x: a Tensor
    n: an integer
    reduced_dim: a Dimension
    name: an optional string
  Returns:
    a Tensor
  """
  # Compute the top k=n+1 values, then take the last one.
  k_dim = Dimension("_top_k_", n + 1)
  values, _ = top_k(x, reduced_dim=reduced_dim, k_dim=k_dim, name=name)
  return gather(values, n, k_dim)


def nth_smallest_element(x, n, reduced_dim, name=None):
  return -nth_largest_element(-x, n, reduced_dim, name=name)
