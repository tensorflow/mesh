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

"""Input pipeline for MeshTensorflow.

If you run MeshTensorflow models on TPUs, please use SimdMeshImplInputReader
as your input pipeline. Otherwise, please use PlacementMeshImplInputReader.

For SimdMeshImplInputReader, a user provides the following, and this set of APIs
will handle the input pipeline for MeshTensorflow.
  1. An instance of mtf.simd_mesh_impl.SimdMeshImpl.
  2. A function that creates a tf.data.Dataset.
     The Dataset returns single examples (no batch dimension).
  3. Shape (mtf.Shape) of each tensor given by tf.data.Dataset.
     Each of these shapes must begin with the same batch dimension.

Example of usage:
  simd_mesh_impl = mtf.simd_mesh_impl.SimdMeshImpl(...)
  simd_input_reader = SimdMeshImplInputReader(simd_mesh_impl,
                                              ds_creator,
                                              mtf_input_shapes)

  infeed_queue = simd_input_reader.infeed_queue
  tpu_train_computation = tpu.replicate(
      computation=model_fn,
      inputs=[[]] * num_cores,
      infeed_queue=infeed_queue, ...)

  # In model_fn, import the input tensors using mtf.import_laid_out_tensor.
  def model_fn(features, labels):
    ...
    laidout_features = mtf.simd_mesh_impl.SimdMeshImpl.LaidOutTensor([features])
    x = mtf.import_laid_out_tensor(mesh, laidout_features, mtf_io_shape)
    h = mtf.layers.dense(h, ...)
    ...

  # Start the infeed enqueue thread after you created a session:
  with tf.Session(...) as sess:
    simd_input_reader.start_infeed_thread(sess,
                                          number_steps=num_training_steps)
    for _ in range(num_training_steps):
      sess.run(tpu_train_computation)

Also check out SimdMeshImplInputReader.gen_infeed_queue().
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import threading
import time
import numpy as np
from six.moves import range
from six.moves import zip
import tensorflow.compat.v1 as tf

# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.framework import ops
from tensorflow.python.tpu import tpu_feed


_NONE_PNUM = None
_NO_DATA = None


def _host_device_to_id(device_str):
  assert isinstance(device_str, str)
  id_string = device_str.lower().split("/task:")[1].split("/device:")[0]
  id_int = int(id_string)
  assert str(id_int) == id_string
  return id_int


def _host_id_to_tf_device(host_id, external_worker):
  assert isinstance(host_id, int)
  if external_worker:
    return "/job:tpu_worker/task:{}/device:CPU:0".format(host_id)
  else:
    return "/task:{}/device:CPU:0".format(host_id)


class SubBatchSlicer(object):
  """Reads and distributes a sub-batch on a host."""

  def __init__(self, sub_batch_ds_creator, host_id, all_sub_batch_pnums,
               simd_mesh_impl, mtf_input_shapes, external_worker, global_batch):
    self._host_id = host_id
    self._all_sub_batch_pnums = all_sub_batch_pnums
    self._simd_mesh_impl = simd_mesh_impl
    self._mtf_input_shapes = mtf_input_shapes
    self._external_worker = external_worker
    self._global_batch = global_batch

    self._validate_args()

    with ops.device(_host_id_to_tf_device(self._host_id,
                                          self._external_worker)):
      self._ds_iterator = sub_batch_ds_creator().make_initializable_iterator()

  @property
  def initializer(self):
    return self._ds_iterator.initializer

  def get_slices(self):
    """Yields sliced tensors and which remote pnums they should go to.

    Yields:
      tf_tensor: The sliced tensor.
      pnum: Which process number the tf_tensor should to go.
      input_i: The input ordinal of the tf_tensor.
    """
    with ops.device(_host_id_to_tf_device(self._host_id,
                                          self._external_worker)):
      all_input_tensors = self._ds_iterator.get_next()
      if isinstance(all_input_tensors, tf.Tensor):
        all_input_tensors = [all_input_tensors]
      assert len(all_input_tensors) == len(self._all_sub_batch_pnums)

      for input_i in range(len(all_input_tensors)):
        input_tensor = all_input_tensors[input_i]
        sub_batch_pnums = self._all_sub_batch_pnums[input_i]
        mtf_input_shape = self._mtf_input_shapes[input_i]

        # Initialize the cache for each input_i
        self._init_slice_cache()

        for pnum in sub_batch_pnums:
          # TODO(lehou): tf.slice is kinda slow. Use tf.split instead.
          input_slice = self._slice_tensor(input_tensor, mtf_input_shape, pnum)
          yield input_slice, pnum, input_i

  def _validate_args(self):
    assert isinstance(self._all_sub_batch_pnums, list)
    assert isinstance(self._mtf_input_shapes, list)
    assert self._all_sub_batch_pnums
    assert self._mtf_input_shapes
    assert len(self._all_sub_batch_pnums) == len(self._mtf_input_shapes)

  def _init_slice_cache(self):
    # Cache for tensor slices
    self._slice_dict = collections.defaultdict(list)

  def _slice_tensor(self, input_tensor, mtf_input_shape, pnum):
    """Slice input_tensor according to mtf_input_shape and pnum."""
    s_begin = self._simd_mesh_impl.slice_begin(mtf_input_shape, pnum)
    if not self._global_batch:
      # Always slice from 0 in the first dimension (batch dimension), since
      # input_tensor a sub-batch tensor.
      s_begin[0] = 0
    if tuple(s_begin) in self._slice_dict:
      return self._slice_dict[tuple(s_begin)]

    s_shape = self._simd_mesh_impl.slice_shape(mtf_input_shape)
    input_slice = tf.slice(input_tensor, s_begin, s_shape)

    self._slice_dict[tuple(s_begin)] = input_slice
    return input_slice


class ProcessDevices(object):
  """An utility class that maps between pnum to devices."""

  def __init__(self, simd_mesh_impl):
    """Init tpu and host devices in logical order."""
    self._num_cores = simd_mesh_impl.device_assignment.num_replicas

    self._ordered_ordinals = []
    self._ordered_tpus = []
    self._ordered_hosts = []
    self._ordered_host_ids = []
    self._host_id_to_its_pnums = collections.defaultdict(list)
    d_assignment = simd_mesh_impl.device_assignment

    for pnum in range(self.num_cores):
      physical_pnum = simd_mesh_impl.l2p(pnum)

      # For MTF, there's always 1 core per replica. So logical_core=0.
      self._ordered_ordinals.append(
          d_assignment.tpu_ordinal(replica=physical_pnum, logical_core=0))
      tpu_device = d_assignment.tpu_device(replica=physical_pnum)
      host_device = d_assignment.host_device(replica=physical_pnum)
      host_id = _host_device_to_id(host_device)
      self._ordered_tpus.append(tpu_device)
      self._ordered_hosts.append(host_device)
      self._ordered_host_ids.append(host_id)
      self._host_id_to_its_pnums[host_id].append(pnum)

    self._num_hosts = len(set(self._ordered_hosts))
    self._num_cores_per_host = self.num_cores // self._num_hosts
    assert self.num_cores == self._num_hosts * self._num_cores_per_host

    tf.logging.info("Process Devices "
                    "ordered_ordinals: {}, "
                    "ordered_tpus: {}, "
                    "ordered_hosts: {}, "
                    "host_id_to_its_pnums: {}.".format(
                        self.ordered_ordinals,
                        self.ordered_tpus,
                        self.ordered_hosts,
                        self.host_id_to_its_pnums))

  @property
  def ordered_ordinals(self):
    return self._ordered_ordinals

  @property
  def ordered_tpus(self):
    return self._ordered_tpus

  @property
  def ordered_hosts(self):
    return self._ordered_hosts

  @property
  def ordered_host_ids(self):
    return self._ordered_host_ids

  @property
  def host_id_to_its_pnums(self):
    return self._host_id_to_its_pnums

  @property
  def num_cores(self):
    return self._num_cores

  @property
  def num_hosts(self):
    return self._num_hosts

  @property
  def num_cores_per_host(self):
    return self._num_cores_per_host


class SimdMeshImplInputReader(object):
  """Handles input pipeline for SimdMeshImpl."""

  def __init__(self,
               simd_mesh_impl,
               ds_creator,
               mtf_input_shapes,
               ds_prefetch_size=tf.data.experimental.AUTOTUNE,
               external_worker=True,
               is_eval_mode=False):
    """Input pipeline for the SIMD implementation of MeshTensorflow.

    Args:
      simd_mesh_impl: A mtf.simd_mesh_impl.SimdMeshImpl object.
      ds_creator: A function that creates a dataset.
      mtf_input_shapes: A list of mtf.Shape. Then length of it must be equal
        to the number of elements generated by the ds_creator. NOTE, we assume:
          1. The 0-th dimension is the batch dimension.
          2. The batch dimension is consistent across all input shapes in
             mtf_input_shapes.
      ds_prefetch_size: The buffer size for prefetching
        (default tf.data.experimental.AUTOTUNE).
      external_worker: Whether you have an external tpu_worker or not. Set it to
        False if you run the program locally, for example, during local unit
        test.
      is_eval_mode: In evaluation mode, only one dataset object will be created,
        as opposed to one dataset for each sub-batch. Default is False. Set it
        to True during evaluation, to ensure that one evaluation instance will
        be used once and only once.
    Note:
      1. The efficiency is optimized according to the shape of the 0-th tensor:
         mtf_input_shapes[0]. We recommand you to put the largest tensor as the
         0-th input.
      2. You need to call start_infeed_thread() before your train ops.
    Example:
        simd_mesh_impl = mtf.simd_mesh_impl.SimdMeshImpl(...)
        # ds_creator is function that creates a tf.data.Dataset.
        # This Dataset must return single examples (no batch dimension).
        def ds_creator():
          return tf.data.Dataset.from_tensors(x)
        # mtf_input_shapes is a list of Shapes of all input tensors given by the
        # dataset. All shapes must begin with the same batch dimension.
        simd_input_reader = SimdMeshImplInputReader(simd_mesh_impl,
                                                    ds_creator,
                                                    mtf_input_shapes)

        batch_dim = mtf.Dimension('batch', FLAGS.batch_size)
        io_dim = mtf.Dimension('io', FLAGS.io_size)
        mtf_input_shapes = [mtf.Shape([batch_dim, io_dim])]

        infeed_queue = simd_input_reader.infeed_queue
        tpu_train_computation = tpu.replicate(
            computation=model_fn,
            inputs=[[]] * num_cores,
            infeed_queue=infeed_queue, ...)

        with tf.Session() as sess:
          simd_input_reader.start_infeed_thread(sess,
                                                number_steps=num_training_steps)
          for _ in range(num_training_steps):
            sess.run(tpu_train_computation)
    """
    super(SimdMeshImplInputReader, self).__init__()
    assert mtf_input_shapes
    assert isinstance(mtf_input_shapes, list)

    # TODO(lehou): Support nested structures for ds_creator, mtf_input_shapes.
    self._simd_mesh_impl = simd_mesh_impl
    self._p_dev = ProcessDevices(simd_mesh_impl)
    self._ds_creator = ds_creator
    self._mtf_input_shapes = mtf_input_shapes
    self._ds_prefetch_size = ds_prefetch_size
    self._external_worker = external_worker
    self._is_eval_mode = is_eval_mode

    self._gen_infeed_queue()

  @property
  def infeed_queue(self):
    return self._infeed_queue

  def start_infeed_thread(self, sess, number_steps=-1, initial_wait_sec=0.5):
    """Start running enqueue ops in a thread.

    Args:
      sess: A tf.Session.
      number_steps: Number of times to call sess.run(enqueue_ops).
        default is -1 (forever).
      initial_wait_sec: Number of seconds to wait before starting the enqueue
        loop. Default is 0.5.
    """
    def _thread_fn():
      time.sleep(initial_wait_sec)
      if number_steps > 0:
        for _ in range(number_steps):
          sess.run(self._enqueue_ops)
      else:
        while True:
          sess.run(self._enqueue_ops)

    sess.run(self._input_initializers)
    self._infeed_thread = threading.Thread(target=_thread_fn)
    self._infeed_thread.start()

  def _gen_infeed_queue(self):
    """Generates _infeed_queue, _enqueue_ops, _input_initializers."""
    pnum_maps = []
    batch_size = self._mtf_input_shapes[0].to_integer_list[0]
    for mtf_shape in self._mtf_input_shapes:
      # Make sure that the batch size is the same across all input tensors.
      assert batch_size == mtf_shape.to_integer_list[0]
      pnum_maps.append(self._get_pnum_map(mtf_shape))

    # For each sub-batch, we need to know which host should read it.
    if self._is_eval_mode:
      # There should be just one dataset-holding host. Make the last host do it.
      hosts_to_hold_ds = [self._p_dev.num_hosts - 1]
    else:
      hosts_to_hold_ds = self._get_hosts_to_hold_ds(pnum_maps[0])
    sub_batch_size = batch_size // len(hosts_to_hold_ds)
    tf.logging.info("MTF sub_batch_size: {}".format(sub_batch_size))
    assert sub_batch_size * len(hosts_to_hold_ds) == batch_size

    def sub_batch_ds_creator():
      return self._ds_creator().batch(
          sub_batch_size, drop_remainder=True).prefetch(
              self._ds_prefetch_size)

    sub_batch_slicer_list = []
    # For each sub-batch, create a SubBatchSlicer object.
    for sub_batch_i, host_id in enumerate(hosts_to_hold_ds):
      # Get the list of pnums for each input.
      if self._is_eval_mode:
        all_sub_batch_pnums = [
            pnum_map.flatten().tolist() for pnum_map in pnum_maps]

        sub_batch_slicer_list.append(SubBatchSlicer(sub_batch_ds_creator,
                                                    host_id,
                                                    all_sub_batch_pnums,
                                                    self._simd_mesh_impl,
                                                    self._mtf_input_shapes,
                                                    self._external_worker,
                                                    global_batch=True))
      else:
        all_sub_batch_pnums = []
        for pnum_map in pnum_maps:
          sub_batch_pnums = pnum_map[sub_batch_i, ...].flatten().tolist()
          all_sub_batch_pnums.append(sub_batch_pnums)

        sub_batch_slicer_list.append(SubBatchSlicer(sub_batch_ds_creator,
                                                    host_id,
                                                    all_sub_batch_pnums,
                                                    self._simd_mesh_impl,
                                                    self._mtf_input_shapes,
                                                    self._external_worker,
                                                    global_batch=False))

    # Slots for all laidout tensors.
    all_laidout_tensors = [[_NO_DATA] * len(self._mtf_input_shapes) \
                           for _ in range(self._p_dev.num_cores)]

    # Read tf_tensors, put them in slots.
    for sub_batch_slicer in sub_batch_slicer_list:
      for tf_tensor, pnum, input_i in sub_batch_slicer.get_slices():
        all_laidout_tensors[pnum][input_i] = tf_tensor

    # Make sure that there are no Nones in all_laidout_tensors.
    for laidout_tensors in all_laidout_tensors:
      assert _NO_DATA not in laidout_tensors

    with ops.device(_host_id_to_tf_device(hosts_to_hold_ds[0],
                                          self._external_worker)):
      self._infeed_queue, self._enqueue_ops = self._enqueue_laidout_tensors(
          all_laidout_tensors)

    self._input_initializers = [s.initializer for s in sub_batch_slicer_list]

  def _get_pnum_map(self, mtf_shape):
    """Returns the pnum_map according to mtf_shape.

    Args:
      mtf_shape: A mtf.Shape object.
    Returns:
      A numpy array pnum_map. For the i-th sub-batch, pnum_map[i] is a numpy
      array containing all pnums that tensor slices of the i-th sub-batch
      will be send to.
    """
    s_shape = self._simd_mesh_impl.slice_shape(mtf_shape)
    shape_list = [dim_size // s_dim_size for dim_size, s_dim_size in zip(
        mtf_shape.to_integer_list, s_shape)]

    pnum_map_shape = shape_list + [
        self._p_dev.num_cores // np.prod(shape_list)]
    assert np.prod(pnum_map_shape) == self._p_dev.num_cores

    # Initialize the pnum_map to _NONE_PNUM.
    pnum_map = np.empty(pnum_map_shape, dtype=object)
    pnum_map[:] = _NONE_PNUM

    for pnum in range(self._p_dev.num_cores):
      s_begin = self._simd_mesh_impl.slice_begin(mtf_shape, pnum)
      coord = [dim_size // s_dim_size for dim_size, s_dim_size in zip(
          s_begin, s_shape)]
      # put pnum in pnum_map[coord]
      pnum_array_ref = pnum_map[tuple(coord)]
      for idx, value in enumerate(pnum_array_ref):
        if value is _NONE_PNUM:
          pnum_array_ref[idx] = pnum
          break

    tf.logging.info("MTF pnum_map: {}".format(pnum_map))
    assert _NONE_PNUM not in pnum_map
    return pnum_map

  def _get_hosts_to_hold_ds(self, pnum_map):
    """Finds which host should read which sub-batch."""
    assert _NONE_PNUM not in pnum_map

    # This records how many datasets (ds) are already stored on each host.
    num_dss_per_host = [0] * self._p_dev.num_hosts

    # A list of host_ids that holds datasets (ds).
    hosts_to_hold_ds = []

    def _get_num_pnums_per_host(sub_batch_pnum_map):
      num_pnums_per_host = [0] * self._p_dev.num_hosts
      for pnum in sub_batch_pnum_map.flatten():
        num_pnums_per_host[self._p_dev.ordered_host_ids[pnum]] += 1
      return num_pnums_per_host

    def _find_host_id_with_most_pnums_and_least_ds(num_pnums_per_host,
                                                   num_dss_per_host):
      host_metics = [(
          host_id, num_pnums_per_host[host_id],
          num_dss_per_host[host_id]) \
                     for host_id in range(self._p_dev.num_hosts)]
      # Major max key: num_pnums
      # Minor max key: -num_dss. We need to find a relatively spare host.
      host_id, _, _ = max(host_metics, key=lambda keys: (keys[1], -keys[2]))
      return host_id

    for sub_batch_pnum_map in pnum_map:
      num_pnums_per_host = _get_num_pnums_per_host(sub_batch_pnum_map)
      host_id = _find_host_id_with_most_pnums_and_least_ds(num_pnums_per_host,
                                                           num_dss_per_host)
      num_dss_per_host[host_id] += 1
      hosts_to_hold_ds.append(host_id)

    return hosts_to_hold_ds

  def _enqueue_laidout_tensors(self, all_laidout_tensors):
    """Generate enqueue ops to enqueue all_laidout_tensors."""

    def _tpu_ordinal_function_impl(pnum):
      return self._p_dev.ordered_ordinals[pnum]

    def _placement_function_impl(pnum):
      return self._p_dev.ordered_hosts[pnum]

    laidout_tensors0 = all_laidout_tensors[0]
    infeed_queue = tpu_feed.InfeedQueue(
        number_of_tuple_elements=len(laidout_tensors0),
        tuple_types=[x.dtype for x in laidout_tensors0],
        tuple_shapes=[x.shape for x in laidout_tensors0])
    enqueue_ops = infeed_queue.generate_enqueue_ops(
        all_laidout_tensors,
        tpu_ordinal_function=_tpu_ordinal_function_impl,
        placement_function=_placement_function_impl)

    return infeed_queue, enqueue_ops


class PlacementMeshImplInputReader(object):
  """Handles input pipeline for PlacementMeshImpl."""

  def __init__(self,
               placement_mesh_impl,
               ds_creator,
               mtf_input_shapes,
               ds_prefetch_size=tf.data.experimental.AUTOTUNE,
               is_eval_mode=False):

    self._placement_mesh_impl = placement_mesh_impl
    self._mtf_input_shapes = mtf_input_shapes

    batch_size = mtf_input_shapes[0].dims[0].size
    if is_eval_mode:
      ds = ds_creator().batch(
          batch_size, drop_remainder=False).prefetch(ds_prefetch_size)
    else:
      ds = ds_creator().batch(
          batch_size, drop_remainder=True).prefetch(ds_prefetch_size)
    self._ds_iterator = ds.make_initializable_iterator()
    self._input_initializers = [self._ds_iterator.initializer]

  def initialize(self, sess):
    sess.run(self._input_initializers)

  def gpu_placement(self, model_fn):
    image, label = self._ds_iterator.get_next()
    image_laid_out = self._placement_mesh_impl.make_slices(
        image, self._mtf_input_shapes[0])
    label_laid_out = self._placement_mesh_impl.make_slices(
        label, self._mtf_input_shapes[1])
    computation = model_fn(image_laid_out, label_laid_out)

    return computation
