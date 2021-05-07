"""
Benchmark script for studying the scaling of distributed FFTs on Mesh Tensorflow
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
import tensorflow.compat.v1 as tf
import mesh_tensorflow as mtf

from tensorflow.python.tpu import tpu_config  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.tpu import tpu_estimator  # pylint: disable=g-direct-tensorflow-import
from tensorflow_estimator.python.estimator import estimator as estimator_lib

# Cloud TPU Cluster Resolver flags
tf.flags.DEFINE_string(
    "tpu", default=None,
    help="The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")
tf.flags.DEFINE_string(
    "tpu_zone", default=None,
    help="[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")
tf.flags.DEFINE_string(
    "gcp_project", default=None,
    help="[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("model_dir", None, "Estimator model_dir")

tf.flags.DEFINE_integer("cube_size", 512, "Size of the 3D volume.")
tf.flags.DEFINE_integer("batch_size", 128,
                        "Mini-batch size for the training. Note that this "
                        "is the global batch size and not the per-shard batch.")

tf.flags.DEFINE_string("mesh_shape", "b1:32", "mesh shape")
tf.flags.DEFINE_string("layout", "nx:b1,tny:b1", "layout rules")

FLAGS = tf.flags.FLAGS

def benchmark_model(mesh):
  """
  Initializes a 3D volume with random noise, and execute a forward FFT
  """
  batch_dim = mtf.Dimension("batch", FLAGS.batch_size)

  # Declares real space dimensions
  x_dim = mtf.Dimension("nx", FLAGS.cube_size)
  y_dim = mtf.Dimension("ny", FLAGS.cube_size)
  z_dim = mtf.Dimension("nz", FLAGS.cube_size)

  # Declares Fourier space dimensions
  tx_dim = mtf.Dimension("tnx", FLAGS.cube_size)
  ty_dim = mtf.Dimension("tny", FLAGS.cube_size)
  tz_dim = mtf.Dimension("tnz", FLAGS.cube_size)

  # Create field
  field = mtf.random_uniform(mesh, [batch_dim, x_dim, y_dim, z_dim])

  # Apply FFT
  fft_field = mtf.signal.fft3d(mtf.cast(field, tf.complex64), [tx_dim, ty_dim, tz_dim])

  # Inverse FFT
  rfield = mtf.cast(mtf.signal.ifft3d(fft_field, [x_dim, y_dim, z_dim]), tf.float32)

  # Compute errors
  err = mtf.reduce_max(mtf.abs(field - rfield))
  return err

def model_fn(features, labels, mode, params):
  """A model is called by TpuEstimator."""
  del labels
  del features

  mesh_shape = mtf.convert_to_shape(FLAGS.mesh_shape)
  layout_rules = mtf.convert_to_layout_rules(FLAGS.layout)

  ctx = params['context']
  num_hosts = ctx.num_hosts
  host_placement_fn = ctx.tpu_host_placement_function
  device_list = [host_placement_fn(host_id=t) for t in range(num_hosts)]
  tf.logging.info('device_list = %s' % device_list,)

  mesh_devices = [''] * mesh_shape.size
  mesh_impl = mtf.simd_mesh_impl.SimdMeshImpl(
      mesh_shape, layout_rules, mesh_devices, ctx.device_assignment)

  graph = mtf.Graph()
  mesh = mtf.Mesh(graph, "fft_mesh")

  with mtf.utils.outside_all_rewrites():
    err = benchmark_model(mesh)

  lowering = mtf.Lowering(graph, {mesh: mesh_impl})

  tf_err = tf.to_float(lowering.export_to_tf_tensor(err))

  with mtf.utils.outside_all_rewrites():
    return tpu_estimator.TPUEstimatorSpec(mode, loss=tf_err)


def main(_):

  tf.logging.set_verbosity(tf.logging.INFO)
  mesh_shape = mtf.convert_to_shape(FLAGS.mesh_shape)

  # Resolve the TPU environment
  tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
      FLAGS.tpu,
      zone=FLAGS.tpu_zone,
      project=FLAGS.gcp_project
  )

  run_config = tf.estimator.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      model_dir=FLAGS.model_dir,
      save_checkpoints_steps=None,  # Disable the default saver
      save_checkpoints_secs=None,  # Disable the default saver
      log_step_count_steps=100,
      save_summary_steps=100,
      tpu_config=tpu_config.TPUConfig(
          num_shards=mesh_shape.size,
          iterations_per_loop=100,
          num_cores_per_replica=1,
          per_host_input_for_training=tpu_config.InputPipelineConfig.BROADCAST))

  model = tpu_estimator.TPUEstimator(
      use_tpu=True,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.batch_size,
      eval_batch_size=FLAGS.batch_size)

  def dummy_input_fn(params):
    """Dummy input function """
    return tf.zeros(shape=[params['batch_size']], dtype=tf.float32), tf.zeros(shape=[params['batch_size']], dtype=tf.float32)

  # Run evaluate loop for ever, we will be connecting to this process using a profiler
  model.evaluate(input_fn=dummy_input_fn, steps=100000)

if __name__ == "__main__":
  tf.disable_v2_behavior()
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
