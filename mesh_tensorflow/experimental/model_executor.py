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

"""A toy model using Mesh TensorFlow.

Using input_reader to handle the input pipeline.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import mesh_tensorflow as mtf
import numpy as np
import six
from six.moves import range
from six.moves import zip
import tensorflow.compat.v1 as tf

# pylint: disable=g-direct-tensorflow-import
# pylint: disable=g-direct-third-party-import
from mesh_tensorflow.experimental import input_reader
from mesh_tensorflow.experimental import unet
from tensorflow.contrib import summary as contrib_summary
from tensorflow.contrib import tpu
from tensorflow.contrib.tpu.python.tpu import device_assignment
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.platform import flags
from tensorflow.python.tpu.ops import tpu_ops


FLAGS = flags.FLAGS

flags.DEFINE_boolean('use_tpu', True, 'Use TPU or GPU.')
flags.DEFINE_float('lr', 0.003, 'Learning rate.')
flags.DEFINE_float('lr_drop_steps', 20000,
                   'Learning rate drops for every `lr_drop_steps` steps.')
flags.DEFINE_float('lr_drop_rate', 0.3,
                   'Learning rate drops by this amount.')
flags.DEFINE_integer('num_train_iterations_per_loop', 500,
                     'Number of training iterations per loop.')
flags.DEFINE_integer('num_eval_iterations_per_loop', 2,
                     'Number of eval iterations per loop.')
flags.DEFINE_integer('num_training_loops', 1000,
                     'Number of training loops.')

flags.DEFINE_string('mesh_shape', 'rows:4, columns:4, cores:2',
                    'mesh shape')
flags.DEFINE_string('master', '', 'Can be a headless master.')

flags.DEFINE_string('checkpoint_dir', '', 'Path to saved models.')
flags.DEFINE_integer('save_checkpoints_steps', 500,
                     'Frequency for saving models.')

flags.DEFINE_boolean('on_gcp', False, 'Assign true if running on google cloud.')
flags.DEFINE_boolean('write_summary', True, 'Whether to write summary.')
flags.DEFINE_string('summary_dir', '', 'Path to saved summaries.')
flags.DEFINE_string('pred_output_dir', '', 'Path to saved pred results.')


class _CapturedObject(object):
  """A placeholder to capture an object.

  This is useful when we need to capture a Python object in the Tensorflow
  control flow body function and use it outside the control flow.
  """

  def __init__(self):
    self._object = None
    self._captured = False

  def capture(self, o):
    if self._captured:
      raise RuntimeError(
          'InternalError: Object can capture only once. Please file bug.')

    self._captured = True
    self._object = o

  def get(self):
    if not self._captured:
      raise RuntimeError(
          'InternalError: Object is not captured properly before `get`. '
          'Please file bug.')
    return self._object


class _CkptLoaderHook(tf.estimator.SessionRunHook):
  """Load checkpoint right after the session started."""

  def after_create_session(self, session, coord):
    # pylint: disable=protected-access
    saver_collection = tf.get_collection(tf.GraphKeys.SAVERS)
    if saver_collection:
      saver = saver_collection[0]
      check_point = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
      if check_point:
        saver.restore(session, check_point)


class MeshContext(object):
  """Creates mtf graph, mesh, and mesh implementation."""

  def __init__(self, sess, use_tpu, mesh_shape, layout_rules):
    super(MeshContext, self).__init__()
    self._use_tpu = use_tpu
    self._mesh_shape = mtf.convert_to_shape(mesh_shape)
    self._layout_rules = layout_rules

    self._d_assignment = None
    self._num_hosts = None
    self._num_cores = None

    self._cpu_devices, self._gpu_devices = self._list_cpu_gpu_devices(sess)

    if self._use_tpu:
      topology = sess.run(tpu.initialize_system())
      topo_object = tpu.Topology(serialized=topology)
      self._num_cores = int(np.prod(topo_object.mesh_shape))
      self._num_hosts = int(topo_object.num_tasks)
      num_cores_per_host = int(self._num_cores // self._num_hosts)
      assert num_cores_per_host == int(topo_object.num_tpus_per_task)

      # Get a device_assignment object for mtf.
      self._d_assignment = device_assignment.device_assignment(
          topology,
          computation_shape=[1,] * mtf.utils.topology_rank(topology),
          num_replicas=self._num_cores)

      self._mesh_impl = mtf.simd_mesh_impl.SimdMeshImpl(
          self._mesh_shape, self._layout_rules, None, self._d_assignment)
    else:
      self._mesh_impl = mtf.placement_mesh_impl.PlacementMeshImpl(
          self._mesh_shape, self._layout_rules, self._gpu_devices)

  def create_graph_mesh_and_mesh_impl(self):
    """Creates mtf graph, mesh, and mesh impl.

    This function can be called inside model_fn, which might be tpu_rewritten.

    Returns:
      graph, mesh, mesh_impl
    """

    if self._use_tpu:
      assert self._d_assignment
      graph = mtf.Graph()

      # Worker 0 caches all the TPU binaries.
      replica_cache_size = 300 * 1024 * 1024  # 300M per replica.
      worker0_mem = replica_cache_size * 8 * self._num_hosts
      devices_memory_usage = [worker0_mem] + [0] * (self._num_hosts - 1)
      var_placer = mtf.utils.BalancedVariablePlacer(self._cpu_devices,
                                                    devices_memory_usage)
      mesh = mtf.Mesh(graph, 'my_mesh', var_placer)
      mesh_impl = mtf.simd_mesh_impl.SimdMeshImpl(
          self._mesh_shape, self._layout_rules, None, self._d_assignment)
      return graph, mesh, mesh_impl

    else:
      graph = mtf.Graph()
      mesh = mtf.Mesh(graph, 'my_mesh', None)
      mesh_impl = mtf.placement_mesh_impl.PlacementMeshImpl(
          self._mesh_shape, self._layout_rules, self._gpu_devices)
      return graph, mesh, mesh_impl

  @property
  def device_assignment(self):
    return self._d_assignment

  @property
  def num_hosts(self):
    return self._num_hosts

  @property
  def num_cores(self):
    return self._num_cores

  @property
  def num_cores_per_host(self):
    return self._num_cores // self._num_hosts

  @property
  def mesh_impl(self):
    return self._mesh_impl

  def _list_cpu_gpu_devices(self, sess):
    """Return the list of CPU and GPU (if any) devices in legacy name."""
    def _convert_to_legacy_name(n):
      n = re.sub('device:CPU', 'cpu', n)
      n = re.sub('device:GPU', 'gpu', n)
      return n

    def _sort_device_name(devices):
      parsed = []
      for d in devices:
        m = re.match('/job:(.*)/replica:(.*)/task:(.*)/.*', d)
        parsed.append((m.group(1), int(m.group(2)), int(m.group(3)), d))
      return [_[3] for _ in sorted(parsed)]

    all_devices = sess.list_devices()

    cpus = []
    for d in all_devices:
      if d.device_type == 'CPU':
        cpus += [_convert_to_legacy_name(d.name)]
    cpus = [n for n in _sort_device_name(cpus) if 'coordinator' not in n]

    gpus = []
    for d in all_devices:
      if d.device_type == 'GPU':
        gpus += [_convert_to_legacy_name(d.name)]
    gpus = _sort_device_name(gpus)

    return cpus, gpus


def _get_model_fn(train_or_eval, mesh_context):
  """Returns _model_fn."""
  captured_hooks = _CapturedObject()
  captured_output_dtypes_shapes = _CapturedObject()
  assert train_or_eval in ['train', 'eval']

  def _model_fn(input_fea, input_lab):
    """Creates a model, add summary, modes (train or eval), and hooks."""

    # input_fea and input_lab should be a list (laid_out_tensors).
    if not isinstance(input_fea, list):
      input_fea = [input_fea]
    if not isinstance(input_lab, list):
      input_lab = [input_lab]

    def _add_summary(lowering, train_or_eval, tf_loss, scalars, global_step):
      """Add all summaries."""
      for k in scalars.keys():
        if not isinstance(scalars[k], tf.Tensor):
          scalars[k] = tf.cast(
              lowering.export_to_tf_tensor(scalars[k]), tf.float32)

      def _host_loss_summary(global_step, tf_loss, **scalars):
        """Add summary.scalar in host side."""
        gs = tf.cast(global_step, tf.int64)
        sum_loss = contrib_summary.scalar(
            '{}_loss'.format(train_or_eval), tf_loss, step=gs)
        sum_ops = [sum_loss.op]
        for description, tf_metric in six.iteritems(scalars):
          sum_metric = contrib_summary.scalar(
              '{}_{}'.format(train_or_eval, description), tf_metric, step=gs)
          sum_ops.append(sum_metric)
        with tf.control_dependencies(sum_ops):
          return tf.identity(tf_loss)

      if FLAGS.use_tpu:
        # Cast the global step to tf.int32, since
        # outside_compilation does not support tf.int64.
        tf_loss = tpu.outside_compilation(
            _host_loss_summary,
            tf.cast(global_step, tf.int32),
            tf_loss,
            **scalars)
      else:
        tf_loss = _host_loss_summary(
            tf.cast(global_step, tf.int32),
            tf_loss,
            **scalars)

      return tf_loss

    global_step = tf.train.get_or_create_global_step()
    graph, mesh, mesh_impl = mesh_context.create_graph_mesh_and_mesh_impl()

    with mtf.utils.outside_all_rewrites():
      # Do not tpu_rewrite this part. Inside this unet, If you use Tensorflow,
      # instead of Mesh-Tensorflor, it will cause host to tpu send/rec.
      preds, loss, scalars, bn_update_ops = (
          unet.unet_with_spatial_partition(
              mesh, mesh_impl, train_or_eval, input_fea, input_lab))

    if train_or_eval == 'train':
      var_grads = mtf.gradients(
          [loss], [v.outputs[0] for v in graph.trainable_variables])

      lr = FLAGS.lr * tf.pow(
          FLAGS.lr_drop_rate,
          tf.floor(tf.cast(global_step, tf.float32) / FLAGS.lr_drop_steps))
      scalars['learning_rate'] = lr

      optimizer = mtf.optimize.AdafactorOptimizer(learning_rate=lr)
      update_ops = optimizer.apply_grads(var_grads, graph.trainable_variables)

      # This is where the actual tf graph got built.
      lowering = mtf.Lowering(graph, {mesh: mesh_impl})

      tf_update_ops = [lowering.lowered_operation(op) for op in update_ops]
      tf_update_ops.append(tf.assign_add(global_step, 1))
      tf_update_ops.extend(
          [lowering.lowered_operation(op) for op in bn_update_ops])

    else:  # train_or_eval == 'eval':
      preds = [mtf.anonymize(pred) for pred in preds]

      # This is where the actual tf graph got built.
      lowering = mtf.Lowering(graph, {mesh: mesh_impl})

      tf_preds = [tf.cast(
          lowering.export_to_tf_tensor(pred), tf.float32) for pred in preds]

    tf_loss = tf.cast(lowering.export_to_tf_tensor(loss), tf.float32)
    if FLAGS.write_summary:
      tf_loss = _add_summary(
          lowering, train_or_eval, tf_loss, scalars, global_step)
    master_to_slice_hook = mtf.MtfRestoreHook(lowering)

    if train_or_eval == 'train':
      with mtf.utils.outside_all_rewrites():
        saver = tf.train.Saver(tf.global_variables(),
                               save_relative_paths=True)
        tf.add_to_collection(tf.GraphKeys.SAVERS, saver)
        saver_listener = mtf.MtfCheckpointSaverListener(lowering)
        slice_to_master_hook = tf.train.CheckpointSaverHook(
            FLAGS.checkpoint_dir,
            save_steps=FLAGS.save_checkpoints_steps,
            saver=saver, listeners=[saver_listener])
        captured_hooks.capture([master_to_slice_hook, slice_to_master_hook])
        return tf.group([tf_loss] + tf_update_ops)

    else:  # train_or_eval == 'eval':
      if FLAGS.use_tpu:
        tf_preds.extend([tf_loss, global_step])
        tf_preds_dtypes = [tf_pred.dtype for tf_pred in tf_preds]
        tf_preds_shapes = [tf_pred.shape for tf_pred in tf_preds]
        captured_hooks.capture([master_to_slice_hook, None])
        captured_output_dtypes_shapes.capture(
            [tf_preds_dtypes, tf_preds_shapes])
        return tpu_ops.outfeed_enqueue_tuple(tf_preds)

      else:
        tf_preds.extend([tf_loss, global_step])
        captured_hooks.capture([master_to_slice_hook, None])
        return tf_preds

  return _model_fn, captured_hooks, captured_output_dtypes_shapes


def _get_scaffold(additional_initializers):
  return tf.train.Scaffold(
      init_op=control_flow_ops.group(
          tf.global_variables_initializer(),
          *additional_initializers),
      local_init_op=tf.group(
          tf.local_variables_initializer(),
          tf.train.Scaffold.default_local_init_op(),
          *additional_initializers))


def _print_variable_values(sess):
  """May give `Protocol buffer too large` error."""
  np.set_printoptions(precision=4, linewidth=1000)
  tf.logging.info('Printing variables.')
  tf.logging.info('===================')
  values = sess.run(tf.trainable_variables())
  for variable, value in zip(tf.trainable_variables(), values):
    tf.logging.info('{}, {}'.format(variable.name, value.shape))
    tf.logging.info('{}'.format(np.array(value).flatten()))


def _train_phase(mesh_context, config, master):
  """Handles input pipeline and trains the network."""
  if FLAGS.num_train_iterations_per_loop <= 0:
    return

  def _run_train_phase():
    """The real function that runs the training phase."""
    # Setup input pipeline.
    ds_creator = unet.get_dataset_creator('train')
    mtf_shapes = unet.get_input_mtf_shapes('train')

    model_train_fn, train_hooks, _ = _get_model_fn('train', mesh_context)

    if FLAGS.use_tpu:
      assert mesh_context.device_assignment
      assert mesh_context.num_cores
      simd_input_reader = input_reader.SimdMeshImplInputReader(
          mesh_context.mesh_impl, ds_creator, mtf_shapes,
          external_worker=(not FLAGS.on_gcp), is_eval_mode=False)
      train_computation = tpu.replicate(
          computation=model_train_fn,
          inputs=[[]] * mesh_context.num_cores,
          infeed_queue=simd_input_reader.infeed_queue,
          device_assignment=mesh_context.device_assignment)

    else:
      placement_input_reader = input_reader.PlacementMeshImplInputReader(
          mesh_context.mesh_impl, ds_creator, mtf_shapes, is_eval_mode=False)
      train_computation = placement_input_reader.gpu_placement(model_train_fn)

    ###########################################################
    # Training.
    master_to_slice_hook, slice_to_master_hook = train_hooks.get()
    ckpt_loader_hook = _CkptLoaderHook()
    step_counter_hook = tf.train.StepCounterHook(every_n_steps=10)
    all_hooks = [ckpt_loader_hook, master_to_slice_hook,
                 slice_to_master_hook, step_counter_hook]

    if FLAGS.write_summary:
      flush_summary = contrib_summary.flush()

    with tf.train.MonitoredTrainingSession(
        master=master,
        scaffold=_get_scaffold(additional_initializers=[]),
        hooks=all_hooks,
        config=config) as sess:

      if FLAGS.write_summary:
        contrib_summary.initialize(session=sess)

      if FLAGS.use_tpu:
        simd_input_reader.start_infeed_thread(
            sess, FLAGS.num_train_iterations_per_loop)
      else:
        placement_input_reader.initialize(sess)

      for step in range(FLAGS.num_train_iterations_per_loop):
        sess.run(train_computation)
        if FLAGS.write_summary:
          sess.run(flush_summary)
        tf.logging.info('train steps: {}'.format(step))

  with tf.Graph().as_default():
    if FLAGS.write_summary:
      summary_writer = contrib_summary.create_file_writer(FLAGS.summary_dir)
      with summary_writer.as_default(), (
          contrib_summary.always_record_summaries()):
        _run_train_phase()
    else:
      _run_train_phase()


def _eval_phase(mesh_context, config, master):
  """Handles input pipeline and evaluates the network."""
  if FLAGS.num_eval_iterations_per_loop <= 0:
    return

  def _run_eval_phase():
    """The real function that runs the evaluation phase."""
    # Setup input pipeline.
    ds_creator = unet.get_dataset_creator('eval')
    mtf_shapes = unet.get_input_mtf_shapes('eval')

    model_eval_fn, eval_hooks, output_dtypes_shapes = _get_model_fn(
        'eval', mesh_context)

    if FLAGS.use_tpu:
      assert mesh_context.device_assignment
      assert mesh_context.num_cores
      simd_input_reader = input_reader.SimdMeshImplInputReader(
          mesh_context.mesh_impl, ds_creator, mtf_shapes,
          external_worker=(not FLAGS.on_gcp), is_eval_mode=True)
      eval_computation = tpu.replicate(
          computation=model_eval_fn,
          inputs=[[]] * mesh_context.num_cores,
          infeed_queue=simd_input_reader.infeed_queue,
          device_assignment=mesh_context.device_assignment)

      output_dtypes, output_shapes = output_dtypes_shapes.get()
      outfeed_dequeue_ops = []

      # Create outfeed_dequeue_ops.
      for host_id in range(mesh_context.num_hosts):
        # pylint: disable=protected-access
        with ops.device(input_reader._host_id_to_tf_device(
            host_id, external_worker=(not FLAGS.on_gcp))):
          for device_ordinal in range(mesh_context.num_cores_per_host):
            outfeed_dequeue_op = tpu_ops.outfeed_dequeue_tuple(
                dtypes=output_dtypes,
                shapes=output_shapes,
                device_ordinal=device_ordinal)

            # We don't need output other than from core 0.
            if outfeed_dequeue_ops:
              outfeed_dequeue_ops.append(
                  [tf.reduce_mean(x) for x in outfeed_dequeue_op])
            else:
              outfeed_dequeue_ops.append(outfeed_dequeue_op)

    else:
      placement_input_reader = input_reader.PlacementMeshImplInputReader(
          mesh_context.mesh_impl, ds_creator, mtf_shapes, is_eval_mode=False)
      eval_computation = placement_input_reader.gpu_placement(model_eval_fn)

    ###########################################################
    # Evaluation.
    master_to_slice_hook, _ = eval_hooks.get()
    ckpt_loader_hook = _CkptLoaderHook()
    all_hooks = [ckpt_loader_hook, master_to_slice_hook]

    if FLAGS.write_summary:
      flush_summary = contrib_summary.flush()

    with tf.train.MonitoredSession(
        session_creator=tf.train.ChiefSessionCreator(
            master=master,
            config=config),
        hooks=all_hooks) as sess:

      if FLAGS.write_summary:
        contrib_summary.initialize(session=sess)

      if FLAGS.use_tpu:
        simd_input_reader.start_infeed_thread(
            sess, FLAGS.num_eval_iterations_per_loop)
      else:
        placement_input_reader.initialize(sess)

      pprocessor = unet.PostProcessor()
      for step in range(FLAGS.num_eval_iterations_per_loop):
        # Only get results from the 0-th core.
        if FLAGS.use_tpu:
          sess.run(eval_computation)
          results = sess.run(outfeed_dequeue_ops)[0]
        else:
          results = sess.run(eval_computation)
        pprocessor.record(results, FLAGS.pred_output_dir)

        if FLAGS.write_summary:
          sess.run(flush_summary)
        tf.logging.info('eval steps: {}'.format(step))
      pprocessor.finish()

  with tf.Graph().as_default():
    if FLAGS.write_summary:
      summary_writer = contrib_summary.create_file_writer(FLAGS.summary_dir)
      with summary_writer.as_default(), (
          contrib_summary.always_record_summaries()):
        _run_eval_phase()
    else:
      _run_eval_phase()


def train_and_eval():
  """Trains and evaluates MeshTensorflow model without TPUEstimator.

  TODO(lehou): Pack everything nicely as a set of APIs.
  """

  mesh_context = None
  tf.logging.info('FLAGS.master: {}'.format(FLAGS.master))
  resolver = tf.distribute.cluster_resolver.TPUClusterResolver(FLAGS.master)
  config = tf.ConfigProto()
  config.allow_soft_placement = True
  cluster_spec = resolver.cluster_spec()
  if cluster_spec:
    config.cluster_def.CopyFrom(cluster_spec.as_cluster_def())
  with tf.Session(target=resolver.master(), config=config) as sess:
    tf.tpu.experimental.initialize_tpu_system(resolver)
    mesh_context = MeshContext(
        sess, FLAGS.use_tpu, FLAGS.mesh_shape, unet.get_layout())

  for _ in range(FLAGS.num_training_loops):
    _train_phase(mesh_context, config, resolver.get_master())
    _eval_phase(mesh_context, config, resolver.get_master())

  if FLAGS.use_tpu:
    with tf.Session(target=resolver.get_master(), config=config) as sess:
      sess.run(tpu.shutdown_system())

  tf.logging.info('finished.')


def main(_):
  train_and_eval()


if __name__ == '__main__':
  tf.app.run()
