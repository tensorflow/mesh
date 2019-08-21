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

"""A toy model using Mesh TensorFlow.

Using mtf_input_reader to handle the input pipeline.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import mesh_tensorflow as mtf
import numpy as np
import tensorflow as tf

# pylint: disable=g-direct-tensorflow-import
# pylint: disable=g-direct-third-party-import
from mesh_tensorflow.experimental import mtf_input_reader
from mesh_tensorflow.experimental import mtf_unet
from tensorflow.contrib import tpu
from tensorflow.contrib.tpu.python.tpu import device_assignment
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.platform import flags


FLAGS = flags.FLAGS

tf.flags.DEFINE_float('lr', 3e-3, 'Learning rate.')
tf.flags.DEFINE_integer('num_train_iterations_per_loop', 200,
                        'Number of training iterations per loop.')
tf.flags.DEFINE_integer('num_eval_iterations_per_loop', 2,
                        'Number of eval iterations per loop.')
tf.flags.DEFINE_integer('num_training_loops', 1000,
                        'Number of training loops.')

tf.flags.DEFINE_string('mesh_shape', 'rows:4, columns:4, cores:2',
                       'mesh shape')
tf.flags.DEFINE_string('master', '', 'Can be a headless master.')

tf.flags.DEFINE_string('checkpoint_dir', '', 'Path to saved models.')
tf.flags.DEFINE_integer('save_checkpoints_steps', 100,
                        'Frequency for saving models.')

tf.flags.DEFINE_string('summary_dir', '', 'Path to saved summaries.')


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


def _list_cpu_devices(sess):
  """Return the list of CPU devices in legacy name."""
  def _convert_to_legacy_name(n):
    n = re.sub('device:CPU', 'cpu', n)
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

  return [n for n in _sort_device_name(cpus) if 'coordinator' not in n]


def _get_model_fn(train_or_eval, cpu_devices, d_assignment, num_hosts):
  """Returns _model_fn."""
  captured_hooks = _CapturedObject()
  assert train_or_eval in ['train', 'eval']

  def _model_fn(input_fea, input_lab):
    """Creates a model, add summary, modes (train or eval), and hooks."""

    def _add_summary(lowering, train_or_eval, loss, scalars, global_step):
      """Add all summaries."""
      tf_loss = tf.to_float(lowering.export_to_tf_tensor(loss))
      for k in scalars.keys():
        scalars[k] = tf.to_float(
            lowering.export_to_tf_tensor(scalars[k]))

      def _host_loss_summary(global_step, tf_loss, **scalars):
        """Add summary.scalar in host side."""
        gs = tf.cast(global_step, tf.int64)
        sum_loss = tf.contrib.summary.scalar(
            '{}_loss'.format(train_or_eval), tf_loss, step=gs)
        sum_ops = [sum_loss.op]
        for description, tf_metric in scalars.iteritems():
          sum_metric = tf.contrib.summary.scalar(
              '{}_{}'.format(train_or_eval, description), tf_metric, step=gs)
          sum_ops.append(sum_metric)
        with tf.control_dependencies(sum_ops):
          return tf.identity(tf_loss)

      # Cast the global step to tf.int32, since
      # outside_compilation does not support tf.int64.
      tf_loss = tpu.outside_compilation(
          _host_loss_summary,
          tf.cast(global_step, tf.int32),
          tf_loss,
          **scalars)

      return tf_loss

    global_step = tf.train.get_or_create_global_step()
    graph = mtf.Graph()

    # Worker 0 caches all the TPU binaries.
    replica_cache_size = 300 * 1024 * 1024  # 300M per replica.
    worker0_mem = replica_cache_size * 8 * num_hosts
    devices_memory_usage = [worker0_mem] + [0] * (num_hosts - 1)

    tf.logging.info('cpu_devices: {}, devices_mem: {}'.format(
        cpu_devices, devices_memory_usage))
    var_placer = mtf.utils.BalancedVariablePlacer(cpu_devices,
                                                  devices_memory_usage)

    mesh = mtf.Mesh(graph, 'my_mesh', var_placer)

    mesh_shape = mtf.convert_to_shape(FLAGS.mesh_shape)
    layout_rules = mtf_unet.get_layout()
    mesh_impl = mtf.simd_mesh_impl.SimdMeshImpl(
        mesh_shape, layout_rules, None, d_assignment)

    with mtf.utils.outside_all_rewrites():  # Do not tpu_rewrite this part.
      # Not using the logits output for now.
      _, loss, scalars, bn_update_ops = (
          mtf_unet.unet_with_spatial_partition(
              mesh, train_or_eval, input_fea, input_lab))

    if train_or_eval == 'train':
      var_grads = mtf.gradients(
          [loss], [v.outputs[0] for v in graph.trainable_variables])
      optimizer = mtf.optimize.AdafactorOptimizer(learning_rate=FLAGS.lr)
      update_ops = optimizer.apply_grads(var_grads, graph.trainable_variables)

      # This is where the actual tf graph got built.
      lowering = mtf.Lowering(graph, {mesh: mesh_impl})

      tf_update_ops = [lowering.lowered_operation(op) for op in update_ops]
      tf_update_ops.append(tf.assign_add(global_step, 1))
      tf_update_ops.extend(
          [lowering.lowered_operation(op) for op in bn_update_ops])
      tf_update_ops_group = tf.group(tf_update_ops)

    else:  # train_or_eval == 'eval':
      # This is where the actual tf graph got built.
      lowering = mtf.Lowering(graph, {mesh: mesh_impl})

    tf_loss = _add_summary(lowering, train_or_eval, loss, scalars, global_step)

    with mtf.utils.outside_all_rewrites():
      master_to_slice_hook = mtf.MtfRestoreHook(lowering)
      if train_or_eval == 'train':
        saver = tf.train.Saver(tf.global_variables(),
                               save_relative_paths=True)
        tf.add_to_collection(tf.GraphKeys.SAVERS, saver)
        saver_listener = mtf.MtfCheckpointSaverListener(lowering)
        slice_to_master_hook = tf.train.CheckpointSaverHook(
            FLAGS.checkpoint_dir,
            save_steps=FLAGS.save_checkpoints_steps,
            saver=saver, listeners=[saver_listener])
        captured_hooks.capture([master_to_slice_hook, slice_to_master_hook])
        return tf_update_ops_group

      else:  # train_or_eval == 'eval':
        captured_hooks.capture([master_to_slice_hook, None])
        return tf_loss

  return _model_fn, captured_hooks


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


def _train_phase(mesh_impl, cpu_devices, d_assignment, num_hosts, num_cores):
  """Train network."""
  with tf.Graph().as_default():
    summary_writer = tf.contrib.summary.create_file_writer(FLAGS.summary_dir)
    with summary_writer.as_default(), (
        tf.contrib.summary.always_record_summaries()):
      # Setup input pipeline.
      ds_creator = mtf_unet.get_dataset_creator('train')
      mtf_shapes = mtf_unet.get_input_mtf_shapes('train')
      simd_input_reader = mtf_input_reader.SimdMeshImplInputReader(
          mesh_impl, ds_creator, mtf_shapes)

      model_train_fn, train_hooks = _get_model_fn(
          'train', cpu_devices, d_assignment, num_hosts)
      tpu_train_computation = tpu.replicate(
          computation=model_train_fn,
          inputs=[[]] * num_cores,
          infeed_queue=simd_input_reader.infeed_queue,
          device_assignment=d_assignment)

      ###########################################################
      # Training.
      master_to_slice_hook, slice_to_master_hook = train_hooks.get()
      ckpt_loader_hook = _CkptLoaderHook()
      flush_summary = tf.contrib.summary.flush()

      with tf.train.MonitoredTrainingSession(
          master=FLAGS.master,
          scaffold=_get_scaffold(additional_initializers=[]),
          hooks=[ckpt_loader_hook, master_to_slice_hook, slice_to_master_hook],
          config=tf.ConfigProto(allow_soft_placement=True)) as sess:

        tf.contrib.summary.initialize(session=sess)
        simd_input_reader.start_infeed_thread(
            sess, FLAGS.num_train_iterations_per_loop)

        for step in range(FLAGS.num_train_iterations_per_loop):
          sess.run([tpu_train_computation, flush_summary])
          tf.logging.info('train steps: {}'.format(step))


def _eval_phase(mesh_impl, cpu_devices, d_assignment, num_hosts, num_cores):
  """Evaluate network and write summary."""
  with tf.Graph().as_default():
    summary_writer = tf.contrib.summary.create_file_writer(FLAGS.summary_dir)
    with summary_writer.as_default(), (
        tf.contrib.summary.always_record_summaries()):
      # Setup input pipeline.
      ds_creator = mtf_unet.get_dataset_creator('eval')
      mtf_shapes = mtf_unet.get_input_mtf_shapes('eval')
      simd_input_reader = mtf_input_reader.SimdMeshImplInputReader(
          mesh_impl, ds_creator, mtf_shapes)

      model_eval_fn, eval_hooks = _get_model_fn(
          'eval', cpu_devices, d_assignment, num_hosts)
      tpu_eval_computation = tpu.replicate(
          computation=model_eval_fn,
          inputs=[[]] * num_cores,
          infeed_queue=simd_input_reader.infeed_queue,
          device_assignment=d_assignment)

      ###########################################################
      # Evaluation.
      master_to_slice_hook, _ = eval_hooks.get()
      ckpt_loader_hook = _CkptLoaderHook()
      flush_summary = tf.contrib.summary.flush()

      with tf.train.MonitoredSession(
          session_creator=tf.train.ChiefSessionCreator(
              master=FLAGS.master,
              config=tf.ConfigProto(allow_soft_placement=True)),
          hooks=[ckpt_loader_hook, master_to_slice_hook]) as sess:

        simd_input_reader.start_infeed_thread(
            sess, FLAGS.num_eval_iterations_per_loop)
        for step in range(FLAGS.num_eval_iterations_per_loop):
          sess.run([tpu_eval_computation, flush_summary])
          tf.logging.info('eval steps: {}'.format(step))


def _shutdown():
  with tf.Session(target=FLAGS.master,
                  config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    sess.run(tpu.shutdown_system())


def train_and_eval():
  """Trains and evaluates MeshTensorflow model without TPUEstimator.

  TODO(lehou): Pack everything nicely as a set of APIs.
  """

  # Open a session to get the list of CPU devices to hold master variables.
  with tf.Session(target=FLAGS.master,
                  config=tf.ConfigProto(allow_soft_placement=True)) as sess:
    topology = sess.run(tpu.initialize_system())
    cpu_devices = _list_cpu_devices(sess)

  topo_object = tf.contrib.tpu.Topology(serialized=topology)
  num_cores = int(np.prod(topo_object.mesh_shape))
  num_hosts = int(topo_object.num_tasks)
  num_cores_per_host = int(num_cores // num_hosts)
  assert num_cores_per_host == int(topo_object.num_tpus_per_task)

  # Get a device_assignment object for mtf.
  d_assignment = device_assignment.device_assignment(
      topology, computation_shape=[1, 1, 1],
      num_replicas=num_cores)

  # Get mesh_impl.
  mesh_shape = mtf.convert_to_shape(FLAGS.mesh_shape)
  layout_rules = mtf_unet.get_layout()
  mesh_impl = mtf.simd_mesh_impl.SimdMeshImpl(
      mesh_shape, layout_rules, None, d_assignment)

  for _ in range(FLAGS.num_training_loops):
    _train_phase(mesh_impl, cpu_devices, d_assignment, num_hosts, num_cores)
    _eval_phase(mesh_impl, cpu_devices, d_assignment, num_hosts, num_cores)

  _shutdown()

  tf.logging.info('finished.')


def main(_):
  train_and_eval()


if __name__ == '__main__':
  tf.compat.v1.app.run()
