# coding=utf-8
# Copyright 2018 The Mesh TensorFlow Authors.
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

from mesh_tensorflow import ops_with_redefined_builtins as mtf
from mesh_tensorflow import tpu_variables
from mesh_tensorflow import utils
from six.moves import xrange  # pylint: disable=redefined-builtin

import tensorflow as tf

from tensorflow.contrib.tpu.python.ops import tpu_ops
from tensorflow.python.framework import ops


class SimdMeshImpl(mtf.MeshImpl):
  """Mesh implementation for TPU using SIMD and MPI operations."""

  def __init__(self, shape, layout, devices, device_assignment):
    super(SimdMeshImpl, self).__init__(shape, layout)
    self._devices = devices
    self._device_assignment = device_assignment
    tf.logging.info("SimdMeshImpl init: {0} {1}".format(shape, layout))
    self._pnum_tensor = None
    self.graph_device_function_stacks = []

  @property
  def pnum_tensor(self):
    if self._pnum_tensor is not None:
      return self._pnum_tensor
    with utils.outside_all_rewrites():
      tf.logging.info("Create pnum_tensor")
      self._pnum_tensor = tpu_ops.tpu_replicated_input(
          list(range(self.size)), name="pnum_constants")
      return self._pnum_tensor

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
          with ops.device(tpu_device):
            mesh_impl.graph_device_function_stacks.append(
                tf.get_default_graph()._device_function_stack.copy())

      for pnum in xrange(mesh_impl.size):
        slice_var_name = base_name + "_slice_%d" % pnum
        # Use tf.Variable instead of tf.get_variable since latter adds lots of
        # useless operations to the TF graph.
        # Note: Repeatedly 'with tf.device():' slows down the graph
        # construction. Therefore we directly use the cached device_stack here.
        tf.get_default_graph(
        )._device_function_stack = mesh_impl.graph_device_function_stacks[pnum]

        slices.append(
            tf.Variable(
                initial_value=zero_tensor,
                trainable=True,
                collections=[],
                dtype=variable.slice_dtype,
                name=slice_var_name,
                expected_shape=slice_shape))

      # Restore the initial stack
      tf.get_default_graph()._device_function_stack = init_device_stack
      # pylint: enable=protected-access

      self._laid_out_tensor = mesh_impl.LaidOutTensor(
          [tpu_variables.ReplicatedVariable(base_name, slices)])
      with tf.device(variable.master_device), utils.outside_all_rewrites():
        self._copy_master_to_slices = self._generate_copy_master_to_slices_op(
            variable.get_master(), shape, slices, slice_shape)
        slices_with_master_dtype = [
            tf.cast(s, variable.master_dtype) for s in slices]
        self._copy_slices_to_master = variable.assign_to_master(
            mesh_impl.combine_slices(slices_with_master_dtype, shape,
                                     device=variable.master_device))

    def _generate_copy_master_to_slices_op(self, master_variable, master_shape,
                                           slices, slice_shape):
      """Generate ops which slices master and assign to slices.

      Args:
        master_variable: The master variable.
        master_shape: The shape of master variable.
        slices: The list of sliced variables.
        slice_shape: The shape of the slice variable.
      Returns:
        A grouped tf.assign ops.
      """
      master_layout = self._mesh_impl.tensor_layout(master_shape)
      # For handling case: master is float32 and slices are bfloat16.
      if master_variable.dtype != slices[0].dtype:
        master_variable = tf.cast(master_variable, slices[0].dtype)
      assign_ops = []
      if master_layout.is_fully_replicated:
        assign_ops = [tf.assign(t, master_variable) for t in slices]
      else:
        slice_dict = {}
        for pnum in xrange(len(slices)):
          slice_begin = self._mesh_impl.slice_begin(master_shape, pnum)
          slice_begin_tuple = tuple(slice_begin)
          # Reuse the same slice if slice_begin doesn't change.
          if slice_begin_tuple not in slice_dict:
            slice_dict[slice_begin_tuple] = tf.slice(master_variable,
                                                     slice_begin, slice_shape)
          assign_ops.append(
              tf.assign(slices[pnum], slice_dict[slice_begin_tuple]))
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
    """Returns a LaidOutTensor containing the processor number.

    Returns:
      a LaidOutTensor where each slice is an integer scalar
    """
    return self.LaidOutTensor([self.pnum_tensor])

  def _create_group_assignment(self, mesh_axes):
    """Create group assignment for XLA cross replica ops."""

    partitioning = {}
    for pnum in xrange(self.size):
      group = mtf.pnum_to_group(self.shape, mesh_axes, pnum)
      if group not in partitioning:
        partitioning[group] = []
      partitioning[group].append(pnum)
    group_assignment = []
    for group, pnums in partitioning.items():
      group_assignment.append(pnums)
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
      tf_in = x.one_slice
      dtype = tf_in.dtype
      if not (dtype == tf.float32 or dtype == tf.bfloat16):
        tf.logging.info("Casting %s to float32 for allreduce" % tf_in.dtype)
        tf_in = tf.cast(tf_in, tf.float32)
      tf_out = tpu_ops.cross_replica_sum(tf_in, group_assignment)
      if tf_out.dtype != dtype:
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
      coord = self.pnum_to_processor_coordinates(self.shape, pnum)
      k = coord[mesh_axis]
      if source_pcoord[k] is not None:
        coord[mesh_axis] = source_pcoord[k]
        target_pnum = self.processor_coordinates_to_pnum(coord)
        source_target_pairs.append([pnum, target_pnum])

    return tpu_ops.collective_permute(t, source_target_pairs)

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
    if fn == tf.add:
      assert len(inputs) == 2
      if isinstance(inputs[0], mtf.LazyAllreduceSum):
        # sum of LazyAllreduceSum (keep delaying the allreduce)
        return inputs[0] + inputs[1]
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
      tf_fn: a function such as tf.random_uniform
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
