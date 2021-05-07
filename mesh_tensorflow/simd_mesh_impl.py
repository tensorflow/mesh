# coding=utf-8
# Copyright 2021 The Mesh TensorFlow Authors.
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

"""SIMD Mesh implementation (for TPU/XLA)."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os

import gin
from mesh_tensorflow import ops_with_redefined_builtins as mtf
from mesh_tensorflow import tpu_variables
from mesh_tensorflow import utils
from six.moves import xrange  # pylint: disable=redefined-builtin

import tensorflow.compat.v1 as tf

from tensorflow.python.tpu.ops import tpu_ops  # pylint: disable=g-direct-tensorflow-import


@gin.configurable
class SimdMeshImpl(mtf.MeshImpl):
  """Mesh implementation for TPU using SIMD and MPI operations."""

  def __init__(self,
               shape,
               layout,
               devices=None,
               device_assignment=None,
               logical_to_physical=None,
               allreduce_in_bfloat16_max_group_size=8):
    """Create a SimdMeshImpl.

    Args:
      shape: an input to mtf.convert_to_shape()
      layout: an input to mtf.convert_to_layout_rules()
      devices: deprecated
      device_assignment: a tf.tpu.experimental.DeviceAssignment -
        devices must be asssigned in lexicographic order
      logical_to_physical: an optional permutation representing the mapping
        from logical cores to "physical" cores, where the physical cores are
        listed in lexicographic order in the physical mesh, and the logical
        cores are listed in lexicographic order in the logical mesh.
        Default is lexicographic order.
      allreduce_in_bfloat16_max_group_size: an integer.  Allreduces of bfloat16
        tensors are done in float32 if the group size exceeds this value.
    """
    super(SimdMeshImpl, self).__init__(shape, layout)
    if devices is not None:
      tf.logging.warning("SimdMeshImpl ignoring devices %s" % devices)
    for dim in shape.dims:
        assert dim.name in layout._mesh_dims, "dim " + dim.name + " not in " + str(layout._mesh_dims)
    self._devices = devices
    self._device_assignment = device_assignment
    tf.logging.info("SimdMeshImpl init: {0} {1}".format(shape, layout))
    tf.logging.info("Device Assignment: {0}".format(device_assignment))
    if logical_to_physical is None:
      # TODO(noam): maybe use auto_logical_to_physical_tpu() here
      logical_to_physical = list(range(self.size))
    if sorted(logical_to_physical) != list(range(self.size)):
      raise ValueError(
          "logical_to_physical must be a permutation on range(shape.size)"
          " shape=%s logical_to_physical=%s" % (shape, logical_to_physical))
    self._logical_to_physical = logical_to_physical
    self._physical_to_logical = [None] * self.size
    for logical, physical in enumerate(self._logical_to_physical):
      self._physical_to_logical[physical] = logical
    self._pnum_tensor = None
    self.graph_device_function_stacks = []
    self.copy_master_to_slice_ops = []
    self._allreduce_in_bfloat16_max_group_size = (
        allreduce_in_bfloat16_max_group_size)

  @property
  def pnum_tensor(self):
    if self._pnum_tensor is not None:
      return self._pnum_tensor
    with utils.outside_all_rewrites():
      tf.logging.info("Create pnum_tensor")
      self._pnum_tensor = tpu_ops.tpu_replicated_input(
          self._physical_to_logical, name="pnum_constants")
      return self._pnum_tensor

  def l2p(self, logical_pnum):
    return self._logical_to_physical[logical_pnum]

  def p2l(self, physical_pnum):
    return self._physical_to_logical[physical_pnum]

  class LaidOutTensor(object):
    """One Slice."""

    def __init__(self, tensor_list):
      assert isinstance(tensor_list, list)
      self._tensor_list = tensor_list

    def __repr__(self):
      return "[" + ",".join([str(t) for t in self._tensor_list]) + "]"

    @property
    def tensor_list(self):
      return self._tensor_list

    @property
    def one_slice(self):
      return self._tensor_list[0]

    @classmethod
    def from_tensor_list(cls, tensor_list):
      return cls(tensor_list)

    @property
    def all_slices(self):
      return self._tensor_list

    @property
    def slice_shape(self):
      return self.one_slice.shape.as_list()

    def to_laid_out_tensor(self):
      return self

  class LaidOutVariable(object):
    """Maintains slice-variables and copy operations."""

    def __init__(self, variable, mesh_impl):
      """Create a LaidOutVariable.

      Args:
        variable: a Variable (Operation)
        mesh_impl: a MeshImpl
      """
      self._variable = variable
      self._mesh_impl = mesh_impl
      shape = variable.outputs[0].shape
      slice_shape = mesh_impl.slice_shape(shape)
      base_name = variable.name
      slices = []
      slices_with_master_dtype = []
      with tf.device(variable.master_device), utils.outside_all_rewrites():
        zero_tensor = tf.zeros(slice_shape, dtype=variable.slice_dtype)

      # pylint: disable=protected-access
      init_device_stack = tf.get_default_graph()._device_function_stack

      if not mesh_impl.graph_device_function_stacks:
        for pnum in xrange(mesh_impl.size):
          tpu_device = mesh_impl.device_assignment.tpu_device(replica=pnum)
          with tf.device(tpu_device):
            mesh_impl.graph_device_function_stacks.append(
                tf.get_default_graph()._device_function_stack.copy())

      for physical_pnum in xrange(mesh_impl.size):
        slice_var_name = base_name + "_slice_%d" % physical_pnum
        # Use tf.Variable instead of tf.get_variable since latter adds lots of
        # useless operations to the TF graph. Use tf.get_variable only if
        # in a AUTO_REUSE scope.
        # Note: Repeatedly 'with tf.device():' slows down the graph
        # construction. Therefore we directly use the cached device_stack here.
        tf.get_default_graph()._device_function_stack = (
            mesh_impl.graph_device_function_stacks[physical_pnum])

        if tf.get_variable_scope().reuse == tf.AUTO_REUSE:
          slice_var = tf.get_variable(
              initializer=zero_tensor,
              trainable=self._variable.trainable,
              collections=["TPU_VAR"],
              dtype=variable.slice_dtype,
              name=slice_var_name)
        else:
          slice_var = tf.Variable(
              initial_value=zero_tensor,
              trainable=self._variable.trainable,
              collections=["TPU_VAR"],
              dtype=variable.slice_dtype,
              name=slice_var_name,
              expected_shape=slice_shape)

        slices.append(slice_var)

      # Restore the initial stack
      tf.get_default_graph()._device_function_stack = init_device_stack
      # pylint: enable=protected-access

      self._laid_out_tensor = mesh_impl.LaidOutTensor(
          [tpu_variables.ReplicatedVariable(base_name, slices)])
      with tf.device(variable.master_device), utils.outside_all_rewrites():
        if os.environ.get("MTF_SEQUENCE_MODE", "") == "1":
          if mesh_impl.copy_master_to_slice_ops:
            with tf.control_dependencies(
                [mesh_impl.copy_master_to_slice_ops[-1]]):
              self._copy_master_to_slices = self._gen_copy_master_to_slices_op(
                  variable.get_master(), shape, slices, slice_shape)
          else:
            self._copy_master_to_slices = self._gen_copy_master_to_slices_op(
                variable.get_master(), shape, slices, slice_shape)

          mesh_impl.copy_master_to_slice_ops.append(self._copy_master_to_slices)
        else:
          self._copy_master_to_slices = self._gen_copy_master_to_slices_op(
              variable.get_master(), shape, slices, slice_shape)
        slices_with_master_dtype = [
            tf.cast(s, variable.master_dtype) for s in slices]
        slices_with_master_dtype = [
            slices_with_master_dtype[mesh_impl.l2p(logical_pnum)]
            for logical_pnum in range(mesh_impl.size)]
        self._copy_slices_to_master = variable.assign_to_master(
            mesh_impl.combine_slices(slices_with_master_dtype, shape,
                                     device=variable.master_device))

    def _gen_copy_master_to_slices_op(self, master_variable, master_shape,
                                      slices, slice_shape):
      """Generate ops which slices master and assign to slices.

      Args:
        master_variable: The master variable.
        master_shape: The shape of master variable.
        slices: The list of slice-variables in physical order.
        slice_shape: The shape of the slice variable.
      Returns:
        A grouped tf.assign ops.
      """
      mesh_impl = self._mesh_impl
      master_layout = mesh_impl.tensor_layout(master_shape)
      # For handling case: master is float32 and slices are bfloat16.
      if master_variable.dtype != slices[0].dtype:
        master_variable = tf.cast(master_variable, slices[0].dtype)
      assign_ops = []
      if master_layout.is_fully_replicated:
        assign_ops = [tf.assign(t, master_variable) for t in slices]
      else:
        slice_dict = {}
        for logical_pnum in xrange(len(slices)):
          slice_begin = mesh_impl.slice_begin(master_shape, logical_pnum)
          slice_begin_tuple = tuple(slice_begin)
          # Reuse the same slice if slice_begin doesn't change.
          if slice_begin_tuple not in slice_dict:
            slice_dict[slice_begin_tuple] = tf.slice(master_variable,
                                                     slice_begin, slice_shape)
          physical_pnum = mesh_impl.l2p(logical_pnum)
          assign_ops.append(
              tf.assign(slices[physical_pnum], slice_dict[slice_begin_tuple]))
      return tf.group(assign_ops)

    def assign_to_slices(self, assign_fn, values, assign_to_tensor_list=None):
      """Assign to the slice variables.

      Args:
        assign_fn: a function from
          (mtf.Variable, tf.Variable, tf.Tensor) -> tf.Operation
        values: a list of tf.Tensor
        assign_to_tensor_list: an optional list of tf.Variable

      Returns:
        a tf.operation
      """
      if assign_to_tensor_list is None:
        assign_to_tensor_list = self._laid_out_tensor.all_slices
      # Handle both N -> 1 and N -> N cases.
      num_slices = min(len(assign_to_tensor_list), len(values))
      devices = [""] * num_slices
      return tf.group(
          mtf.parallel(devices, assign_fn,
                       [self._variable] * len(devices),
                       assign_to_tensor_list[:num_slices],
                       values[:num_slices]))

    @property
    def laid_out_tensor(self):
      return self._laid_out_tensor

    @property
    def copy_master_to_slices(self):
      return self._copy_master_to_slices

    @property
    def copy_slices_to_master(self):
      return self._copy_slices_to_master

  def laid_out_pnum(self):
    """Returns a LaidOutTensor containing the logical processor number.

    Returns:
      a LaidOutTensor where each slice is an integer scalar
    """
    return self.LaidOutTensor([self.pnum_tensor])

  def _create_group_assignment(self, mesh_axes):
    """Create group assignment for XLA cross replica ops (physical pnums)."""

    partitioning = {}
    for logical_pnum in xrange(self.size):
      group = mtf.pnum_to_group(self.shape, mesh_axes, logical_pnum)
      if group not in partitioning:
        partitioning[group] = []
      partitioning[group].append(self.l2p(logical_pnum))
    group_assignment = []
    for group, physical_pnums in partitioning.items():
      group_assignment.append(physical_pnums)
    return group_assignment

  def allreduce(self, x, mesh_axes, reduction_fn_string):
    """Grouped allreduce, (summed across the given dimensions).

    Args:
      x: a LaidOutTensor
      mesh_axes: a list of integers
      reduction_fn_string: "SUM"
    Returns:
      a LaidOutTensor
    Raises:
      ValueError: if the reduction is not yet implemented.
    """
    if not mesh_axes:
      return x
    x = x.to_laid_out_tensor()
    if reduction_fn_string == "SUM":
      group_assignment = self._create_group_assignment(mesh_axes)
      group_size = len(group_assignment[0])
      tf_in = x.one_slice
      dtype = tf_in.dtype
      if dtype == tf.float32:
        cast_to_float32 = False
      elif dtype == tf.bfloat16:
        cast_to_float32 = (
            group_size > self._allreduce_in_bfloat16_max_group_size)
      else:
        tf.logging.info("Casting %s to float32 for allreduce" % tf_in.dtype)
        cast_to_float32 = True
      if cast_to_float32:
        tf_in = tf.cast(tf_in, tf.float32)
      tf_out = tpu_ops.cross_replica_sum(tf_in, group_assignment)
      if cast_to_float32:
        tf_out = tf.cast(tf_out, dtype)
      return self.LaidOutTensor([tf_out])
    else:
      for axis in mesh_axes:
        x = self.allconcat(x, axis, 0, stack=True)
        x = self.LaidOutTensor(
            [mtf.reduction_fn(reduction_fn_string)(x.one_slice, 0)])
      return x

  def allconcat(self, x, mesh_axis, concat_axis, stack=False):
    """Grouped allconcat (like MPI allgather followed by concat).

    TODO(noam): inefficient - replace with a XLA allconcat when available

    Args:
      x: a LaidOutTensor
      mesh_axis: an integer - the mesh axis along which to group
      concat_axis: an integer (the Tensor axis along which to concatenate)
      stack: a boolean - whether to stack instead of concat
    Returns:
      a LaidOutTensor
    """
    x = x.to_laid_out_tensor()
    coord = self.laid_out_pcoord(mesh_axis)
    t = x.one_slice
    old_shape = t.shape.as_list()
    num_parts = self.shape[mesh_axis].size
    t = tf.expand_dims(t, concat_axis)
    t *= tf.reshape(
        tf.one_hot(coord.one_slice, num_parts, dtype=t.dtype),
        [num_parts if i == concat_axis else 1
         for i in xrange(len(old_shape) + 1)])
    if not stack:
      new_shape = old_shape[:]
      new_shape[concat_axis] *= num_parts
      t = tf.reshape(t, new_shape)
    return self.allreduce(self.LaidOutTensor([t]), [mesh_axis], "SUM")

  def alltoall(self, x, mesh_axis, split_axis, concat_axis):
    """Grouped alltoall (like MPI alltoall with splitting and concatenation).

    Args:
      x: a LaidOutTensor
      mesh_axis: an integer the mesh axis along which to group
      split_axis: an integer (the Tensor axis along which to split)
      concat_axis: an integer (the Tensor axis along which to concatenate)
    Returns:
      a LaidOutTensor
    """
    x = x.to_laid_out_tensor()
    t = x.one_slice
    group_assignment = self._create_group_assignment([mesh_axis])
    dtype = t.dtype
    if dtype == tf.float32:
      # There seems to be a bug with float32 alltoall.
      # Do it in bfloat16 until the bug is fixed.
      # TODO(noam): file a bug
      t = tf.to_bfloat16(t)
    t = tpu_ops.all_to_all(
        t,
        concat_dimension=concat_axis,
        split_dimension=split_axis,
        split_count=len(group_assignment[0]),
        group_assignment=group_assignment)
    t = tf.cast(t, dtype)
    x = self.LaidOutTensor([t])
    return x

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
    x = x.to_laid_out_tensor()
    t = x.one_slice
    source_target_pairs = []

    for pnum in xrange(self.size):
      coord = mtf.pnum_to_processor_coordinates(self.shape, pnum)
      k = coord[mesh_axis]
      if source_pcoord[k] is not None:
        coord[mesh_axis] = source_pcoord[k]
        source_pnum = mtf.processor_coordinates_to_pnum(self.shape, coord)
        source_target_pairs.append(
            [self.l2p(source_pnum),
             self.l2p(pnum)])

    if not source_target_pairs:
      ret = tf.zeros_like(t, t.dtype)
    elif t.dtype in [tf.float32, tf.bfloat16, tf.int32]:
      ret = tpu_ops.collective_permute(t, source_target_pairs)
    else:
      # If t is not one of the allowed types, cast and cast back.
      ret = tf.cast(tpu_ops.collective_permute(
          tf.cast(t, tf.float32), source_target_pairs), t.dtype)

    return self.LaidOutTensor([ret])

  def slice(self, tf_tensor, tensor_shape):
    """"Slice out the corresponding part of tensor given the pnum variable."""
    tensor_layout = self.tensor_layout(tensor_shape)

    if tensor_layout.is_fully_replicated:
      return self.LaidOutTensor([tf_tensor])
    else:
      slice_shape = self.slice_shape(tensor_shape)
      slice_begins = [
          self.slice_begin(tensor_shape, pnum) for pnum in xrange(self.size)
      ]
      slice_begins_tensor = tf.stack(slice_begins)
      # slice on source device
      selected_slice_begin = tf.gather(slice_begins_tensor, self.pnum_tensor)
      return self.LaidOutTensor(
          [tf.slice(tf_tensor, selected_slice_begin, slice_shape)])

  def slicewise(self, fn, *inputs):
    """Execute a function in parallel on all slices.

    Args:
      fn: a function from tf.Tensors to tf.Tensor or a tuple of tf.Tensors.
      *inputs: a list of inputs.  Each input is either a LaidOutTensor or
        is convertible to a tf.Tensor.
    Returns:
      a LaidOutTensor, or a tuple of LaidOutTensors if fn returns a tuple.
    """
    # convert all inputs to LaidOutTensor where possible
    inputs = mtf.convert_args_to_laid_out_tensors(inputs)
    ret = fn(*[
        x.one_slice if isinstance(x, self.LaidOutTensor) else x
        for x in inputs])
    if isinstance(ret, tuple):
      return tuple([self.LaidOutTensor([t]) for t in ret])
    else:
      return self.LaidOutTensor([ret])

  @property
  def device_assignment(self):
    return self._device_assignment

  @property
  def devices(self):
    return self._devices

  def random(self, shape, tf_fn, kwargs):
    """Call a random tf operation (e.g. random_uniform).

    Args:
      shape: a Shape
      tf_fn: a function such as tf.random.uniform
      kwargs: kwargs to pass to tf_fn, except for seed

    Returns:
      a LaidOutTensor
    """
    # TODO(noam): can we make things better with stateless_random?
    slice_shape = self.slice_shape(shape)
    x = tf_fn(slice_shape, **kwargs)
    # TPU does not have seeds enabled.  Sync up the
    # random choices by zeroing out all but the first core per group of
    # identical slices, then allreducing by group.
    layout = self.tensor_layout(shape)
    # we need to sync across these axes.
    mesh_axes = [i for i in xrange(self.ndims)
                 if i not in layout.tensor_axis_to_mesh_axis]
    multiplier = 1.0
    for axis in mesh_axes:
      multiplier *= tf.cast(
          tf.equal(self.laid_out_pcoord(axis).one_slice, 0), x.dtype)
    x *= multiplier
    x = self.LaidOutTensor([x])
    x = self.allreduce(x, mesh_axes, "SUM")
    return x

  def export_to_tf_tensor(self, x, laid_out_x):
    """Turn a Tensor into a tf.Tensor.

    Args:
      x: a Tensor
      laid_out_x: a LaidOutTensor
    Returns:
      a tf.Tensor
    """
    tensor_layout = self.tensor_layout(x.shape)
    if not tensor_layout.is_fully_replicated:
      raise NotImplementedError(
          "SimdMeshImpl only supports export_to_tf_tensor of fully-replicated "
          "Tensors.  Try reshaping to new dimension names. "
          " x.shape = %s tensor_layout=%s"
          % (x.shape, tensor_layout))
    return laid_out_x.one_slice

  def import_tf_tensor(self, x, tf_x):
    """Import a tf.Tensor, producing a LaidOutTensor.

    Args:
      x: a Tensor
      tf_x: a tf.Tensor
    Returns:
      a LaidOutTensor
    """
    return self.slice(tf_x, x.shape)

  @property
  def supports_control_dependencies(self):
    return False

  def einsum(self, equation, *slices):
    """Override this for custom einsum implementation.

    Args:
      equation: a string
      *slices: a list of tf.Tensor
    Returns:
      a tf.Tensor
    """
    return tf.einsum(equation, *slices)


def _ring_2d(m, n):
  """Ring-order of a mxn mesh.

  If m and n are both even, then we generate a ring like this:

     0 -- 1 -- 2 -- 3
     |    |    |    |
     15-- 6 -- 5 -- 4
     |    |    |    |
     14-- 7 -- 8 -- 9
     |    |    |    |
     13-- 12-- 11-- 10

  Args:
    m: an integer
    n: an integer
  Returns:
    a list of mxn pairs
  """
  if m == 1:
    return [(0, i) for i in range(n)]
  if n == 1:
    return [(i, 0) for i in range(m)]
  if m % 2 != 0:
    tf.logging.warning("Odd dimension")
    return [(i % m, i // m) for i in range(n * m)]
  ret = [(0, 0)]
  for i in range(m // 2):
    for j in range(1, n):
      ret.append((2 * i, j))
    for j in range(n-1, 0, -1):
      ret.append((2 * i + 1, j))
  for i in range(m-1, 0, -1):
    ret.append((i, 0))
  return ret


def _logical_1d_to_physical_subspace_auto(sizes_and_strides, physical_shape):
  """Maps logical 1d mesh to subspace of physical nd mesh.

  We are mapping a 1d logical mesh to a subspace (a strided slice containing the
  origin) of a n-dimensional physical mesh.

  output[i] contains the coordinate-tuple in the physical mesh for the i-th
  logical processor.

  sizes_and_strides is a list of (size, stride) pairs specifying the dimensions
  of the strided slice. For example,
    sizes_and_strides=[(2, 16), (4, 1)] would represent the slice containing
    [(0, 0), (0, 1), (0, 2), (0, 3),
     (16, 0), (16, 1), (16, 2), (16, 3)]

  This function heuristically picks an order, with the goal of optimizing
  allreduce performance.

  Args:
    sizes_and_strides: a list of n (size, stride) pairs
    physical_shape: ignored
  Returns:
    a list of coordinate-lists
  """
  del physical_shape
  ndims = len(sizes_and_strides)
  sizes = [p[0] for p in sizes_and_strides]
  strides = [p[1] for p in sizes_and_strides]
  n = mtf.list_product(sizes)
  if ndims >= 2 and sizes[0] > 1 and sizes[1] > 1:
    ring = _ring_2d(sizes[0], sizes[1])
    ret = []
    sizes_combined = [sizes[0] * sizes[1]] + sizes[2:]
    for logical_pnum in range(n):
      logical_coord = mtf.pnum_to_processor_coordinates(
          sizes_combined, logical_pnum)
      ret.append(list(ring[logical_coord[0]]) + logical_coord[1:])
  else:
    ret = [mtf.pnum_to_processor_coordinates(sizes, logical_pnum)
           for logical_pnum in range(n)]
  # multiply by strides
  ret = [[x * stride for x, stride in zip(pcoord, strides)] for pcoord in ret]
  return ret


def _logical_to_physical_v1(
    sizes_and_strides, physical_shape,
    fn_1d=_logical_1d_to_physical_subspace_auto):
  """Maps logical m-dimensional mesh to physical n-dimensional mesh.

  Also see comments to _logical_1d_to_physical_subspace_auto.

  We are mapping a m-dimensonal logical mesh to a n-dimensional physical mesh.

  output[i] contains the coordinate-tuple in the physical mesh for the i-th
  logical processor (if the logical processors are ordered lexicographically).

  sizes_and_strides is a list of m lists of n (size, stride) pairs.

  sizes_and_strides[i] specifies the subspace (strided slice containing the
  origin) of the physical mesh covered by axis i of the logical mesh.  See
  comments to _logical_1d_to_physical_subspace_auto for more detail.

  For example, say we have a physical mesh with shape [4, 4, 2] and a logical
  mesh with shape [4, 8].  We want to divide the physical mesh into 4 tiles,
  each with shape [2, 2, 2].  The first logical dimension corresponds to which
  tile, and the second logical dimension corresponds to position within a tile.
  This would correspond to:
     physical_shape=[4, 4, 2]
     sizes_and_strides=[[(2, 2), (2, 2), (1, 2)], [(2, 1), (2, 1), (2, 1)]]

  physical_shape can be inferred from sizes_and_strides, but is passed in for
  error checking.

  Args:
    sizes_and_strides: a list of m list of n (size, stride) pairs
    physical_shape: a list of integers
    fn_1d: a function like _logical_1d_to_physical_subspace_auto
  Returns:
    a list of coordinate-lists
  """
  pndims = len(physical_shape)
  logical_shape = [
      mtf.list_product([p[0] for p in l]) for l in sizes_and_strides]
  n = mtf.list_product(physical_shape)
  if n != mtf.list_product(logical_shape):
    raise ValueError(
        "logical size and physical size must match "
        "- got sizes_and_strides=%s physical_shape=%s"
        % (sizes_and_strides, physical_shape))
  dimension_layouts = [fn_1d(l, physical_shape) for l in sizes_and_strides]
  tf.logging.info("physical_shape: %s" % physical_shape)
  tf.logging.info("sizes_and_strides: %s" % sizes_and_strides)
  for i, l in enumerate(dimension_layouts):
    tf.logging.info("dimension_layout %s: %s" % (i, l))
  ret = []
  for logical_pnum in range(n):
    logical_coordinates = mtf.pnum_to_processor_coordinates(
        logical_shape, logical_pnum)
    physical_coordinates = [0] * pndims
    for logical_axis, logical_coord in enumerate(logical_coordinates):
      for physical_axis in range(pndims):
        physical_coordinates[physical_axis] += (
            dimension_layouts[logical_axis][logical_coord][physical_axis])
    ret.append(physical_coordinates)
  # verify that we have indeed covered all the processors
  l2p = [mtf.processor_coordinates_to_pnum(physical_shape, c) for c in ret]
  if sorted(l2p) != list(range(n)):
    raise ValueError(
        "logical_to_physical produced something that was not a permutation."
        " sizes_and_strides=%s physical_shape=%s ret=%s"
        % (sizes_and_strides, physical_shape, ret))
  return ret


class HierarchicalTiling(object):
  """One kind of mapping of a logical mesh to a physical mesh."""

  def __init__(self, spec, physical_shape):
    """Constructs a HierarchicalTiling.

    spec is a list corresponding to the logical dimensions.

    spec[i] corresponds to the i-th logical dimension and consists of a name
      and a list of integers, the list being the shape of logical axis i when
      it is physically projected to the physical mesh and then compacted.

    Striding information is omitted.  By convention, the earlier dimensions
      get more strided. so the axis corresponding to the last dimension always
      gets projected to the tile specified by its shape.

    Args:
      spec: a list of (string, list-of-integers) pairs
      physical_shape: a list of integers
    """
    self._names = [p[0] for p in spec]
    logical_ndims = len(spec)
    physical_ndims = len(physical_shape)
    projected_shapes = [p[1] for p in spec]
    if logical_ndims > 0 and projected_shapes[0] is None:
      # fill in missing value
      projected_shapes[0] = list(physical_shape)
      for s in projected_shapes[1:]:
        for i, x in enumerate(s):
          projected_shapes[0][i] //= x
    # compute strides, and verify that the spec is valid.
    products = [1] * physical_ndims
    sizes_and_strides = []
    for s in reversed(projected_shapes):
      sizes_and_strides.append(
          [(size, stride) for size, stride in zip(s, products)])
      for i, x in enumerate(s):
        products[i] *= x
    if products != physical_shape:
      raise ValueError("mesh spec multiplies to the wrong size"
                       "spec=%s physical_shape=%s products=%s" %
                       (spec, physical_shape, products))
    sizes_and_strides.reverse()
    self._physical_coordinates = _logical_to_physical_v1(
        sizes_and_strides, physical_shape)
    self._logical_to_physical = [
        mtf.processor_coordinates_to_pnum(physical_shape, c)
        for c in self._physical_coordinates]
    self._mesh_shape = mtf.Shape(
        [mtf.Dimension(name, mtf.list_product(s))
         for name, s in zip(self._names, projected_shapes)])

  @property
  def logical_to_physical(self):
    """List of physical processor numbers."""
    return list(self._logical_to_physical)

  @property
  def mesh_shape(self):
    return self._mesh_shape

  @classmethod
  def spec_to_mesh_shape(cls, spec, num_processors):
    """Compute mesh shape even without knowing the physical shape.

    This is useful in cases where the mesh shape must be computed before
    you know the physical_shape.

    Args:
      spec: a list of (string, list-of-integers) pairs
      num_processors: an integer
    Returns:
      a mtf.Shape
    """
    logical_ndims = len(spec)
    names = [p[0] for p in spec]
    sizes = [p[1] for p in spec]
    sizes = [None if s is None else mtf.list_product(s) for s in sizes]
    if logical_ndims > 0 and sizes[0] is None:
      sizes[0] = num_processors // mtf.list_product(sizes[1:])
    if mtf.list_product(sizes) != num_processors:
      raise ValueError("product of spec must be num_processors"
                       " spec=%s num_processors=%s"
                       % (spec, num_processors))
    return mtf.Shape(
        [mtf.Dimension(name, s) for name, s in zip(names, sizes)])


def physical_shape_3d_from_topology_proto_4d(mesh_shape):
  """Convert a 4d shape that we get from TPU estimator to a 3d shape.

  Args:
    mesh_shape: a list of length 4
  Returns:
    a list of length 3
  """
  if len(mesh_shape) != 4:
    raise ValueError("Expected a 4d shape [x, y, z, core]")
  return [mesh_shape[1]*mesh_shape[2], mesh_shape[0], mesh_shape[3]]


def auto_logical_to_physical_tpu(logical_shape,
                                 physical_shape,
                                 return_coordinates=False):
  """Set up a mapping from logical to physical cores for TPU.

  We will try to set up a mapping so that allreduce operations are relatively
  fast, prioritizing the later dimensions in the mesh_shape.

  Example:

  auto_logical_to_physical_tpu(
    logical_shape=[16, 8], physical_shape=[8, 8, 1, 2])

  Heuristics in this function subject to change.

  Args:
    logical_shape: a list of integers
    physical_shape: a list of integers - typically [X, Y, 1, cores]
    return_coordinates: a boolean - return a list of integer lists (coordinates)
       instead of a list of processor indices

  Returns:
    logical_to_physical: a permutation of range(product(physical_shape)))
  """
  tf.logging.info("auto_logical_to_physical_tpu "
                  "logical_shape=%s physical_shape=%s" %
                  (logical_shape, physical_shape))
  if mtf.list_product(logical_shape) != mtf.list_product(physical_shape):
    raise ValueError(
        "physical and logical shapes must have the same product "
        "physical_shape=%s logical_shape=%s" % (physical_shape, logical_shape))
  # drop logical dimensions of size 1
  logical_shape = [i for i in logical_shape if i != 1]
  num_cores = mtf.list_product(logical_shape)
  # For physical shapes different from what we are used to [2^a, 2^b, 2],
  #   return a simple default value (a lexicographic ordering)
  def _default_value():
    default = list(range(num_cores))
    if return_coordinates:
      default = [mtf.pnum_to_processor_coordinates(i) for i in default]
    return default
  if len(physical_shape) == 4 and physical_shape[2] == 1:
    physical_shape = physical_shape_3d_from_topology_proto_4d(physical_shape)
  elif len(physical_shape) != 3:
    tf.logging.warning("Unrecognized format for tpu physical shape")
    return _default_value()
  # physical_shape is a triple of rows, cols, cores
  p0, p1, p2 = physical_shape
  if p2 != 2:
    return _default_value
  for dimsize in [p0, p1]:
    # if dimsize not a power of 2, give up
    if dimsize & (dimsize - 1):
      return _default_value()
  # At this point, the physical shape has at least 1x1x2=2 cores, so there
  #   must be at least one logical dimension.
  assert logical_shape
  if len(logical_shape) == 1:
    # ring of p0 x p1 chips
    ring = _ring_2d(p0, p1)
    logical_to_physical = []
    for logical_pnum in range(num_cores):
      core_on_chip = logical_pnum % 2
      chip_num = logical_pnum // 2
      i, j = ring[chip_num]
      logical_to_physical.append((i, j, core_on_chip))
  else:
    # We have a p0 x p1 rectangle of chips, which we will tile with rectangular
    #   tiles.  The first logical dimension correspond to the number of tiles,
    #   and the other logical dimensions will correspond to position within a
    #   tile.
    num_tiles = logical_shape[0]
    tile_chips = num_cores // num_tiles // p2
    # If we can, we make each tile occupy exactly one row or column of chips.
    # Otherwise, we make each tile approximately square.
    if len(logical_shape) == 2 and tile_chips == p0:
      t0, t1 = [tile_chips, 1]
    elif len(logical_shape) == 2 and tile_chips == p1:
      t0, t1 = [1, tile_chips]
    else:
      # try to make the tile approximately square
      lg_tile_chips = int(math.log(tile_chips, 2))
      t0 = 2 ** (lg_tile_chips // 2)
      # make sure that the tile fits in the mesh - i.e.
      #   t0 <= p0
      #   t1 == tile_chips // t0 <= p1
      t0 = min(t0, p0)
      t0 = max(t0, tile_chips // p1)
      t1 = tile_chips // t0
    # recursive call to find mapping for one tile
    tile_logical_to_physical = auto_logical_to_physical_tpu(
        logical_shape[1:], [t0, t1, p2], return_coordinates=True)
    tiles_ring = _ring_2d(p0 // t0, p1 // t1)
    logical_to_physical = []
    for logical_pnum in range(num_cores):
      logical_tile_num = logical_pnum // (t0 * t1 * p2)
      logical_pos_in_tile = logical_pnum % (t0 * t1 * p2)
      logical_to_physical.append((
          tiles_ring[logical_tile_num][0] * t0 +
          tile_logical_to_physical[logical_pos_in_tile][0],
          tiles_ring[logical_tile_num][1] * t1 +
          tile_logical_to_physical[logical_pos_in_tile][1],
          tile_logical_to_physical[logical_pos_in_tile][2]))
  tf.logging.info("auto_logical_to_physical_tpu logical_to_physical = %s"
                  % logical_to_physical)
  if return_coordinates:
    return logical_to_physical
  else:
    return [mtf.processor_coordinates_to_pnum(physical_shape, coord)
            for coord in logical_to_physical]
