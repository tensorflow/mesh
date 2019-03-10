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

r"""Transformer using Mesh-TensorFlow.

Training/Eval/Inference of a transformer machine-translation model.

Data comes from TensorFlow Datasets.

The core transformer model code is in the mesh_tensorflow/transformer/
directory of this repository.

Instructions for running this on cloud TPU are in the README .
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mesh_tensorflow as mtf
import transformer_dataset as transformer_dataset  # local file import
from mesh_tensorflow.transformer import transformer
from mesh_tensorflow.transformer import transformer_layers
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tpu.python.tpu import tpu_config
from tensorflow.contrib.tpu.python.tpu import tpu_estimator

tf.flags.DEFINE_string("model_dir", "/tmp/mnist_model", "Estimator model_dir")
tf.flags.DEFINE_string("dataset", "wmt_translate_ende/ende_subwords8k_t2t",
                       "TensorFlow Datasets dataset name")
tf.flags.DEFINE_string("data_dir", "", "data_dir for TensorFlow Datasets")

# MODEL HYPERPARAMETERS
tf.flags.DEFINE_integer("max_length", 256,
                        "maximum sequence length (checkpoints depend on this)")
tf.flags.DEFINE_integer("length", 0,
                        "actual sequence length - defaults to FLAGS.max_length")
tf.flags.DEFINE_integer("num_layers", 6, "number of transformer layers")
tf.flags.DEFINE_integer("d_model", 512, "size of hidden state")
tf.flags.DEFINE_integer("d_ff", 2048, "size of feed-forward hidden layers")
tf.flags.DEFINE_integer("d_kv", 128, "size of attention keys/values")
tf.flags.DEFINE_integer("num_heads", 8, "heads per attention layer")

# DATA TYPES (each should be float32 or bfloat16)
# master_dtype must be the same between training and eval/inference
# slice_dtype should be float32 for training (otherwise bad quality)
tf.flags.DEFINE_string("master_dtype", "bfloat16", "datatype for checkpoints")
tf.flags.DEFINE_string(
    "slice_dtype", "",
    "datatype for variables in memory. "
    "Defaults to float32 during training and on non-TPU. "
    "Defaults to bfloat16 during non-training on TPU. ")
tf.flags.DEFINE_string(
    "activation_dtype", "",
    "datatype for activations.  Defaults to bfloat16 on TPU else float32")

# TRAINING HYPERPARAMETERS
tf.flags.DEFINE_integer("batch_size", 64,
                        "Mini-batch size for the training. Note that this "
                        "is the global batch size and not the per-shard batch.")
tf.flags.DEFINE_float("dropout", 0.1, "dropout")
tf.flags.DEFINE_float("label_smoothing", 0.1, "label smoothing")
tf.flags.DEFINE_integer(
    "train_steps", 100000, "Total number of training steps.")

# DISTRIBUTED LAYOUT
# When running on TPU, make sure that the size of the mesh
# (the product of all dimension sizes) equals the number of TPU cores.
#
# The layout specifies a partial mapping from tensor-dimension-names to
# mesh-dimension names over which those tensor-dimensions are split.
#
# For our Transformer implementation, the reasonable model-dimensions
# to split are:
#   - "d_ff" (feed-forward hidden-layer size)
#   - "heads" (number of attention heads)
#   - "vocab" (vocabulary size)
# For a model-parallel layout, all three of these dimensions should be split
#   across the same mesh dimension - i.e. layout=d_ff:all,heads:all,vocab:all
# For a data-parallel and model-parallel layout then split the batch along
#   one mesh dimension and the model dimensions along the other:
#   mesh_shape=rows:2,cols:4 layout=batch:rows,d_ff:cols,heads:cols,vocab:cols
tf.flags.DEFINE_string("mesh_shape", "all:8", "mesh shape")
tf.flags.DEFINE_string("layout", "batch:all", "layout")

# INFERENCE HYPERPARAMETERS
tf.flags.DEFINE_float("temperature", 0.0, "sampling temperature for inference")
tf.flags.DEFINE_float("alpha", 0.6, "length adjustment for beam search")
tf.flags.DEFINE_integer("beam_size", 1, "use a value >1 for beam search")
tf.flags.DEFINE_string("input_file", "", "Where to read decoding prompts")
tf.flags.DEFINE_string("output_file", "", "Where to write decoding outputs")

# MISC
tf.flags.DEFINE_integer("iterations_per_loop", 100, "steps per train loop")
tf.flags.DEFINE_integer("save_checkpoints_steps", 1000, "steps per checkpoint")
tf.flags.DEFINE_integer("eval_steps", 10, "Number of evaluation steps.")

tf.flags.DEFINE_string(
    "tpu",
    default=None,
    help="The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.")
tf.flags.DEFINE_string("mode", "train", "train/evaluate/infer")

tf.flags.DEFINE_string(
    "gcp_project",
    default=None,
    help="Project name for the Cloud TPU-enabled project. If not specified, we "
    "will attempt to automatically detect the GCE project from metadata.")

tf.flags.DEFINE_string(
    "tpu_zone",
    default=None,
    help="GCE zone where the Cloud TPU is located in. If not specified, we "
    "will attempt to automatically detect the GCE project from metadata.")

# Enable this to speed up compilation on large clusters.
tf.flags.DEFINE_boolean("autostack", True, "Internally combine variables")

FLAGS = tf.flags.FLAGS


def length_from_flags():
  return FLAGS.length or FLAGS.max_length


def import_feature(features, mesh, key):
  """Import a feature from the features dictionary into a mtf.Tensor.

  Args:
    features: a features dictionary
    mesh: a Mesh
    key: a string

  Returns:
    a mtf.Tensor with dtype int32 and shape [batch_dim, length_dim]
  """
  batch_dim = mtf.Dimension("batch", FLAGS.batch_size)
  length_dim = mtf.Dimension("length", length_from_flags())
  mtf_shape = mtf.Shape([batch_dim, length_dim])
  if key not in features:
    return None
  x = tf.to_int32(features[key])
  if not FLAGS.tpu:
    x = tf.Print(x, [x], "import feature %s" % key, summarize=1000, first_n=1)
  return mtf.import_fully_replicated(mesh, x, mtf_shape, name=key)


def layer_stack(include_encdec_attention):
  """Create a layer stack.

  Args:
    include_encdec_attention: a boolean
  Returns:
    a LayerStack
  """
  ret = []
  for _ in xrange(FLAGS.num_layers):
    ret.append(
        transformer_layers.SelfAttention(
            num_heads=FLAGS.num_heads,
            key_value_size=FLAGS.d_kv,
            attention_kwargs={"dropout_rate": FLAGS.dropout}))
    if include_encdec_attention:
      ret.append(
          transformer_layers.EncDecAttention(
              num_heads=FLAGS.num_heads,
              key_value_size=FLAGS.d_kv,
              attention_kwargs={"dropout_rate": FLAGS.dropout}))
    ret.append(
        transformer_layers.DenseReluDense(
            hidden_size=FLAGS.d_ff,
            dropout_rate=FLAGS.dropout))
  return transformer.LayerStack(ret)


def my_model_fn(features,
                labels,
                mode,
                params=None,
                config=None):
  """Estimator model function.

  Args:
    features: input features dictionary
    labels: ignored
    mode: a tf.estimator.ModeKeys
    params: something
    config: something
  Returns:
    something
  """
  del labels, config
  use_tpu = FLAGS.tpu
  global_step = tf.train.get_global_step()

  mesh_shape = mtf.convert_to_shape(FLAGS.mesh_shape)
  layout_rules = mtf.convert_to_layout_rules(FLAGS.layout)
  if use_tpu:
    ctx = params["context"]
    num_hosts = ctx.num_hosts
    host_placement_fn = ctx.tpu_host_placement_function
    device_list = [host_placement_fn(host_id=t) for t in range(num_hosts)]
    # TODO(ylc): Better estimation of replica cache size?
    replica_cache_size = 300 * 1000000  # 300M per replica
    # Worker 0 caches all the TPU binaries.
    worker0_mem = replica_cache_size * ctx.num_replicas
    devices_memeory_usage = [worker0_mem] + [0] * (num_hosts - 1)
    var_placer = mtf.utils.BalancedVariablePlacer(device_list,
                                                  devices_memeory_usage)
    mesh_devices = [""] * mesh_shape.size
    mesh_impl = mtf.simd_mesh_impl.SimdMeshImpl(
        mesh_shape, layout_rules, mesh_devices, ctx.device_assignment)
  else:
    var_placer = None
    mesh_devices = [""] * mesh_shape.size
    mesh_impl = mtf.placement_mesh_impl.PlacementMeshImpl(
        mesh_shape, layout_rules, mesh_devices)

  graph = mtf.Graph()
  mesh = mtf.Mesh(graph, "my_mesh", var_placer)

  model = transformer.Bitransformer(
      encoder_layer_stack=layer_stack(include_encdec_attention=False),
      decoder_layer_stack=layer_stack(include_encdec_attention=True),
      encoder_d_model=FLAGS.d_model,
      decoder_d_model=FLAGS.d_model,
      input_vocab_size=transformer_dataset.padded_vocab_size(
          transformer_dataset.inputs_vocab_size(FLAGS.dataset)),
      output_vocab_size=transformer_dataset.padded_vocab_size(
          transformer_dataset.targets_vocab_size(FLAGS.dataset)),
      max_length=FLAGS.max_length,
      shared_embedding=False,
      shared_embedding_and_softmax_weights=True,
      label_smoothing=FLAGS.label_smoothing,
      layout=FLAGS.layout,
      mesh_shape=FLAGS.mesh_shape)

  inputs = import_feature(features, mesh, "inputs")

  # Data-types used for variables and activations
  # See comments in the FLAGS
  master_dtype = tf.as_dtype(FLAGS.master_dtype)
  if FLAGS.slice_dtype:
    slice_dtype = tf.as_dtype(FLAGS.slice_dtype)
  elif not FLAGS.tpu or FLAGS.mode == "train":
    slice_dtype = tf.float32
  else:
    slice_dtype = tf.bfloat16
  if FLAGS.activation_dtype:
    activation_dtype = tf.as_dtype(FLAGS.activation_dtype)
  else:
    activation_dtype = tf.bfloat16 if FLAGS.tpu else tf.float32
  variable_dtype = mtf.VariableDType(master_dtype=master_dtype,
                                     slice_dtype=slice_dtype,
                                     activation_dtype=activation_dtype)

  # PREDICT mode
  if mode == tf.estimator.ModeKeys.PREDICT:
    mtf_samples = model.decode(
        inputs,
        variable_dtype=variable_dtype,
        beam_size=FLAGS.beam_size,
        alpha=FLAGS.alpha,
        temperature=FLAGS.temperature)
    mtf_samples = mtf.anonymize(mtf_samples)
    lowering = mtf.Lowering(graph, {mesh: mesh_impl}, autostack=FLAGS.autostack)
    outputs = lowering.export_to_tf_tensor(mtf_samples)
    predictions = {
        "outputs": outputs
    }
    return tpu_estimator.TPUEstimatorSpec(
        mode=tf.estimator.ModeKeys.PREDICT,
        predictions=predictions,
        prediction_hooks=[mtf.MtfRestoreHook(lowering)])

  targets = import_feature(features, mesh, "targets")
  anon_targets = mtf.anonymize(targets)
  logits, loss = model.call_simple(
      inputs=inputs,
      targets=targets,
      compute_loss=True,
      mode=mode,
      variable_dtype=variable_dtype,
      encoder_sequence_id=import_feature(features, mesh, "inputs_segmentation"),
      decoder_sequence_id=import_feature(
          features, mesh, "targets_segmentation"),
      encoder_position=import_feature(features, mesh, "inputs_position"),
      decoder_position=import_feature(features, mesh, "targets_position")
  )

  if use_tpu and logits is not None:
    logits = mtf.anonymize(logits)

  # TRAIN mode
  if mode == tf.estimator.ModeKeys.TRAIN:
    var_grads = mtf.gradients(
        [loss], [v.outputs[0] for v in graph.trainable_variables])
    optimizer = mtf.optimize.AdafactorOptimizer()
    update_ops = optimizer.apply_grads(var_grads, graph.trainable_variables)

  lowering = mtf.Lowering(graph, {mesh: mesh_impl}, autostack=FLAGS.autostack)

  tf_loss = lowering.export_to_tf_tensor(loss)
  tf_loss = tf.to_float(tf_loss)
  if not use_tpu:
    tf_loss = tf.Print(
        tf_loss, [tf_loss, tf.train.get_global_step()], "step, tf_loss")
  if logits and mode != tf.estimator.ModeKeys.TRAIN:
    tf_logits = lowering.export_to_tf_tensor(logits)

  if mode == tf.estimator.ModeKeys.TRAIN:
    tf_update_ops = [lowering.lowered_operation(op) for op in update_ops]
    tf_update_ops.append(tf.assign_add(global_step, 1))
    train_op = tf.group(tf_update_ops)

  with mtf.utils.outside_all_rewrites():
    # Copy master variables to slices. Must be called first.
    restore_hook = mtf.MtfRestoreHook(lowering)
    saver = tf.train.Saver(
        tf.global_variables(),
        sharded=True,
        max_to_keep=10,
        keep_checkpoint_every_n_hours=2,
        defer_build=False,
        save_relative_paths=True)
    tf.add_to_collection(tf.GraphKeys.SAVERS, saver)
    saver_listener = mtf.MtfCheckpointSaverListener(lowering)
    saver_hook = tf.train.CheckpointSaverHook(
        FLAGS.model_dir,
        save_steps=1000,
        saver=saver,
        listeners=[saver_listener])

    if mode == tf.estimator.ModeKeys.TRAIN:
      if use_tpu:
        return tpu_estimator.TPUEstimatorSpec(
            mode=tf.estimator.ModeKeys.TRAIN,
            loss=tf_loss,
            train_op=train_op,
            training_hooks=[restore_hook, saver_hook])
      else:
        return tf.estimator.EstimatorSpec(
            tf.estimator.ModeKeys.TRAIN, loss=tf_loss, train_op=train_op,
            training_chief_hooks=[restore_hook, saver_hook])
    elif mode == tf.estimator.ModeKeys.EVAL:
      def padded_neg_log_perplexity(logits, labels):
        weights = tf.to_float(tf.not_equal(labels, 0))
        xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logits)
        return {"neg_log_perplexity": tf.metrics.mean(-xent, weights)}
      labels = lowering.export_to_tf_tensor(anon_targets)
      eval_metrics = (padded_neg_log_perplexity, [tf_logits, labels])
      return tpu_estimator.TPUEstimatorSpec(
          tf.estimator.ModeKeys.EVAL,
          evaluation_hooks=[restore_hook],
          loss=tf_loss,
          eval_metrics=eval_metrics)


def decode_from_file(estimator):
  """Decode from a text file.

  Args:
    estimator: a TPUEstimator
  """
  with tf.gfile.Open(FLAGS.input_file) as f:
    text = f.read()
  records = text.split("\n")
  inputs = [record.strip() for record in records]
  # Strip the last empty line.
  if not inputs[-1]:
    inputs.pop()
  n = len(inputs)
  # encode all inputs
  encoder = transformer_dataset.inputs_encoder(FLAGS.dataset)
  all_input_ids = []
  length = FLAGS.length or FLAGS.max_length
  for line in inputs:
    ids = encoder.encode(line.strip())
    ids = transformer_dataset.add_eos(ids)
    if len(ids) > length:
      ids = ids[:length]
    else:
      ids.extend([0] * (length - len(ids)))
    all_input_ids.append(ids)
  # pad to make an integral number of batches
  all_input_ids.extend([all_input_ids[0]] * (-n % FLAGS.batch_size))
  padded_n = len(all_input_ids)
  all_input_ids = np.array(all_input_ids, dtype=np.int32)
  def input_fn(params):
    del params
    dataset = tf.data.Dataset.from_tensor_slices({"inputs": all_input_ids})
    dataset = dataset.batch(FLAGS.batch_size, drop_remainder=True)
    return dataset
  result_iter = estimator.predict(input_fn)
  targets_encoder = transformer_dataset.targets_encoder(FLAGS.dataset)
  output_file = tf.gfile.Open(FLAGS.output_file, "w")
  vocab_size = transformer_dataset.targets_vocab_size(FLAGS.dataset)
  decodes = []
  for i, result in enumerate(result_iter):
    output_ids = transformer_dataset.clean_output(
        list(result["outputs"]), vocab_size)
    output_string = targets_encoder.decode(output_ids)
    decodes.append(output_string)
    if i < 3:
      # LOG THE FIRST FEW DECODES
      tf.logging.info(inputs[i])
      tf.logging.info(output_string)
  # BUG WORKAROUND - on TF1.13 and earlier, the output for each batch is
  # repeated a number of times equal to the number of cores.
  num_cores = mtf.convert_to_shape(FLAGS.mesh_shape).size
  if len(decodes) == padded_n:
    tf.logging.info("number of decodes matches number of inputs")
  elif len(decodes) == padded_n * num_cores:
    tf.logging.info("output is repeated num_cores times - removing extras")
    def keep(i):
      return i % (FLAGS.batch_size * num_cores) < FLAGS.batch_size
    decodes = [d for i, d in enumerate(decodes) if keep(i)]
  else:
    raise ValueError("unexpected number of outputs")
  output_file = tf.gfile.Open(FLAGS.output_file, "w")
  decodes = decodes[:n]
  for d in decodes:
    output_file.write(d)
    output_file.write("\n")
  output_file.close()


def main(_):
  """Run training/eval/inference."""
  cluster = tf.contrib.cluster_resolver.TPUClusterResolver(
      tpu=[FLAGS.tpu], zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  my_tpu_config = tpu_config.TPUConfig(
      iterations_per_loop=FLAGS.iterations_per_loop,
      num_cores_per_replica=1,
      per_host_input_for_training=tpu_config.InputPipelineConfig.BROADCAST,
  )

  run_config = tpu_config.RunConfig(
      cluster=cluster,
      model_dir=FLAGS.model_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=my_tpu_config)

  estimator = tpu_estimator.TPUEstimator(
      model_fn=my_model_fn,
      config=run_config,
      train_batch_size=FLAGS.batch_size,
      eval_batch_size=FLAGS.batch_size,
      predict_batch_size=FLAGS.batch_size,
      use_tpu=FLAGS.tpu,
      export_to_tpu=False)

  def input_fn(params):
    del params
    return transformer_dataset.get_dataset(
        FLAGS.dataset,
        FLAGS.data_dir or None,
        train=(FLAGS.mode == "train"),
        batch_size=FLAGS.batch_size,
        length=length_from_flags())

  if FLAGS.mode == "train":
    estimator.train(
        input_fn=input_fn,
        max_steps=FLAGS.train_steps
    )
  elif FLAGS.mode == "evaluate":
    estimator.evaluate(
        input_fn=input_fn,
        steps=FLAGS.eval_steps,
    )
  elif FLAGS.mode == "infer":
    decode_from_file(estimator)
  else:
    raise ValueError(
        "unknown mode %s - must be train/evaluate/infer" % FLAGS.mode)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
