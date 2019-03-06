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

Everything needed to train and infer from a transformer in one file.
TODO(noam): factor and document this better.
TODO(noam): add instructions for running on cloud TPU
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import mesh_tensorflow as mtf
from mesh_tensorflow.transformer import transformer
from mesh_tensorflow.transformer import transformer_layers
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.contrib.tpu.python.tpu import tpu_config
from tensorflow.contrib.tpu.python.tpu import tpu_estimator

tf.flags.DEFINE_string("model_dir", "/tmp/mnist_model", "Estimator model_dir")
tf.flags.DEFINE_string("dataset", "wmt_translate_ende/ende_subwords8k_t2t",
                       "dataset")
tf.flags.DEFINE_string("inputs_feature", "en", "input feature")
tf.flags.DEFINE_string("targets_feature", "de", "target feature")
tf.flags.DEFINE_integer("batch_size", 64,
                        "Mini-batch size for the training. Note that this "
                        "is the global batch size and not the per-shard batch.")
tf.flags.DEFINE_integer("max_length", 256,
                        "maximum sequence length (checkpoints depend on this)")
tf.flags.DEFINE_integer("length", 0,
                        "actual sequence length - defaults to FLAGS.max_length")
tf.flags.DEFINE_integer("num_layers", 6, "number of transformer layers")
tf.flags.DEFINE_integer("d_model", 512, "size of hidden state")
tf.flags.DEFINE_integer("d_ff", 2048, "size of feed-forward hidden layers")
tf.flags.DEFINE_integer("d_kv", 128, "size of attention keys/values")
tf.flags.DEFINE_integer("num_heads", 8, "heads per attention layer")
tf.flags.DEFINE_integer(
    "train_steps", 100000, "Total number of training steps.")

tf.flags.DEFINE_integer("iterations_per_loop", 100, "steps per train loop")
tf.flags.DEFINE_integer("save_checkpoints_steps", 1000, "steps per checkpoint")
tf.flags.DEFINE_integer("steps_between_evals", 1000,
                        "# of epochs between evaluations.")
tf.flags.DEFINE_integer("eval_steps", 0,
                        "Total number of evaluation steps. If `0`, evaluation "
                        "after training is skipped.")
tf.flags.DEFINE_string("mesh_shape", "all:8", "mesh shape")
tf.flags.DEFINE_string("layout", "batch:all", "layout")
tf.flags.DEFINE_float("label_smoothing", 0.1, "label smoothing")
tf.flags.DEFINE_float("dropout", 0.1, "dropout")

# Inference params
tf.flags.DEFINE_float("temperature", 0.0, "sampling temperature for inference")
tf.flags.DEFINE_float("alpha", 0.6, "length adjustment for beam search")
tf.flags.DEFINE_integer("beam_size", 1, "use a value >1 for beam search")
tf.flags.DEFINE_string("input_file", "", "Where to read decoding prompts")
tf.flags.DEFINE_string("output_file", "", "Where to write decoding outputs")

tf.flags.DEFINE_string(
    "tpu",
    default=None,
    help="The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.")
tf.flags.DEFINE_string("mode", "train", "train/evaluate/infer")

FLAGS = tf.flags.FLAGS


def pack_dataset(dataset, length, keys=None):
  """Creates a 'packed' version of a dataset on-the-fly.

  Borrowed from the tensor2tensor library.
  TODO(noam): make this faster
  TODO(noam): move to another file.

  This is meant to replace the irritation of having to create a separate
  "packed" version of a dataset to train efficiently on TPU.

  Each example in the output dataset represents several examples in the
  input dataset.

  For each key in the input dataset, two additional keys are created:
  <key>_segmentation: an int32 tensor identifying the parts
     representing the original example.
  <key>_position: an int32 tensor identifying the position within the original
     example.

  Example:
  Two input examples get combined to form an output example.
  The input examples are:
  {"inputs": [8, 7, 1, 0], "targets":[4, 1, 0]}
  {"inputs": [2, 3, 4, 1], "targets":[5, 6, 1]}
  The output example is:
  {
                 "inputs": [8, 7, 1, 2, 3, 4, 1, 0, 0, 0]
    "inputs_segmentation": [1, 1, 1, 2, 2, 2, 2, 0, 0, 0]
        "inputs_position": [0, 1, 2, 0, 1, 2, 3, 0, 0, 0]
                "targets": [4, 1, 5, 6, 1, 0, 0, 0, 0, 0]
   "targets_segmentation": [1, 1, 2, 2, 2, 0, 0, 0, 0, 0]
       "targets_position": [0, 1, 0, 1, 2, 0, 0, 0, 0, 0]
  }

  0 represents padding in both the inputs and the outputs.

  Sequences in the incoming examples are truncated to length "length", and the
  sequences in the output examples all have fixed (padded) length "length".

  Args:
    dataset: a tf.data.Dataset
    length: an integer
    keys: a list of strings (e.g. ["inputs", "targets"])

  Returns:
    a tf.data.Dataset
  """
  shapes = dataset.output_shapes
  if keys is None:
    keys = shapes.keys()
  for k in keys:
    if k not in shapes:
      raise ValueError("Key %s not found in dataset.  Available keys are %s"
                       % (k, shapes.keys()))
    if not shapes[k].is_compatible_with(tf.TensorShape([None])):
      raise ValueError("Tensors to be packed must be one-dimensional.")

  # trim to length
  dataset = dataset.map(lambda x: {k: x[k][:length] for k in keys})
  # Setting batch_size=length ensures that the concatenated sequences (if they
  # have length >=1) are sufficient to fill at least one packed example.
  batch_size = length
  dataset = dataset.padded_batch(
      batch_size, padded_shapes={k: [-1] for k in keys})
  return _pack_with_tf_ops(dataset, keys, length)


def _pack_with_tf_ops(dataset, keys, length):
  """Helper-function for packing a dataset which has already been batched.

  See pack_dataset()

  Uses tf.while_loop.  Slow.

  Args:
    dataset: a dataset containing padded batches of examples.
    keys: a list of strings
    length: an integer

  Returns:
    a dataset.
  """
  empty_example = {}
  for k in keys:
    empty_example[k] = tf.zeros([0], dtype=tf.int64)
    empty_example[k + "_position"] = tf.zeros([0], dtype=tf.int32)
  keys_etc = empty_example.keys()

  def write_packed_example(partial, outputs):
    new_partial = empty_example.copy()
    new_outputs = {}
    for k in keys_etc:
      new_outputs[k] = outputs[k].write(
          outputs[k].size(),
          tf.pad(partial[k], [[0, length - tf.size(partial[k])]]))
    return new_partial, new_outputs

  def map_fn(x):
    """Internal function to flat_map over.

    Consumes a batch of input examples and produces a variable number of output
    examples.

    Args:
      x: a single example
    Returns:
      a tf.data.Dataset
    """
    partial = empty_example.copy()
    i = tf.zeros([], dtype=tf.int32)
    dynamic_batch_size = tf.shape(x[keys[0]])[0]
    outputs = {}
    for k in keys:
      outputs[k] = tf.TensorArray(
          tf.int64, size=0, dynamic_size=True, element_shape=[length])
      outputs[k + "_position"] = tf.TensorArray(
          tf.int32, size=0, dynamic_size=True, element_shape=[length])
    def cond_fn(i, partial, outputs):
      del partial, outputs
      return i < dynamic_batch_size
    def body_fn(i, partial, outputs):
      """Body function for while_loop.

      Args:
        i: integer scalar
        partial: dictionary of Tensor (partially-constructed example)
        outputs: dictionary of TensorArray
      Returns:
        A triple containing the new values of the inputs.
      """
      can_append = True
      one_example = {}
      for k in keys:
        val = x[k][i]
        val = val[:tf.reduce_sum(tf.to_int32(tf.not_equal(val, 0)))]
        one_example[k] = val
      for k in keys:
        can_append = tf.logical_and(
            can_append,
            tf.less_equal(
                tf.size(partial[k]) + tf.size(one_example[k]), length))
      def false_fn():
        return write_packed_example(partial, outputs)
      def true_fn():
        return partial, outputs
      partial, outputs = tf.cond(can_append, true_fn, false_fn)
      new_partial = {}
      for k in keys:
        new_seq = one_example[k][:length]
        new_seq_len = tf.size(new_seq)
        new_partial[k] = tf.concat([partial[k], new_seq], 0)
        new_partial[k + "_position"] = tf.concat(
            [partial[k + "_position"],
             tf.range(new_seq_len, dtype=tf.int32)], 0)
      partial = new_partial
      return i+1, partial, outputs

    i, partial, outputs = tf.while_loop(
        cond_fn, body_fn, (i, partial, outputs),
        back_prop=False,
        shape_invariants=(
            tf.TensorShape([]),
            {k: tf.TensorShape([None]) for k in keys_etc},
            {k: tf.TensorShape(None) for k in keys_etc},
            ))
    partial, outputs = write_packed_example(partial, outputs)
    packed = {k: outputs[k].stack() for k in keys_etc}
    for k in keys:
      packed[k + "_segmentation"] = (
          tf.cumsum(tf.to_int32(tf.equal(packed[k + "_position"], 0)), axis=1) *
          tf.to_int32(tf.not_equal(packed[k], 0)))

    return tf.data.Dataset.from_tensor_slices(packed)
  dataset = dataset.flat_map(map_fn)
  return dataset


def _trim_and_pad(t, batch_size, length):
  """Trim/pad to get a tf.Tensor with shape [batch_size, length].

  Args:
    t: a 2d tf.Tensor
    batch_size: an integer
    length: an integer
  Returns:
    a 2d Tensor
  """
  t = t[:batch_size, :length]
  paddings = [
      [0, batch_size - tf.shape(t)[0]], [0, length - tf.shape(t)[1]]]
  t = tf.pad(t, paddings)
  return tf.reshape(t, [batch_size, length])


def trim_and_pad_all_features(features, batch_size, length):
  """Trim and pad all features."""
  return {k: _trim_and_pad(v, batch_size, length) for k, v in features.items()}


def add_eos(x):
  """Increase all ids by 1 and append EOS=1.

  Args:
    x: an unpadded 1d tensor of token ids, or a python list
  Returns:
    the same type as x
  """
  if isinstance(x, tf.Tensor):
    return tf.concat([x + 1, [1]], 0)
  elif isinstance(x, list):
    return [i + 1 for i in x] + [1]
  else:
    raise ValueError("unsupported type for x=%s" % (x,))


def clean_output(ids, vocab_size):
  """Decrease all ids by 1, stop at EOS or padding or OOV.

  Args:
    ids: a list of integers
    vocab_size: an integer
  Returns:
    a list of integers
  """
  ret = []
  for i in ids:
    i -= 1
    if i <= 0 or i >= vocab_size:
      break
    else:
      ret.append(i)
  return ret


def get_dataset(train=True):
  """Get a tf.data.Dataset. for training/eval.

  Args:
    train: a boolean
  Returns:
    a tf.data.Dataset
  """
  length = FLAGS.length or FLAGS.max_length
  dataset = tfds.load(
      FLAGS.dataset,
      split=tfds.Split.TRAIN if train else tfds.Split.VALIDATION)
  if train:
    dataset = dataset.repeat()
  def my_fn(x):
    return {"inputs": add_eos(x[FLAGS.inputs_feature]),
            "targets": add_eos(x[FLAGS.targets_feature])}
  dataset = dataset.map(my_fn)
  dataset = pack_dataset(dataset, length=length)
  dataset = dataset.batch(FLAGS.batch_size, drop_remainder=False)
  dataset = dataset.map(
      functools.partial(trim_and_pad_all_features,
                        batch_size=FLAGS.batch_size,
                        length=length))
  return dataset


def padded_vocab_size(vocab_size):
  # shift to make room for EOS=1
  vocab_size += 1
  # pad to multiple of 128
  return vocab_size + (-vocab_size % 128)


def inputs_vocab_size():
  return (tfds.builder(FLAGS.dataset)
          .info.features[FLAGS.inputs_feature].encoder.vocab_size)


def targets_vocab_size():
  return (tfds.builder(FLAGS.dataset)
          .info.features[FLAGS.targets_feature].encoder.vocab_size)


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
  length_dim = mtf.Dimension("length", FLAGS.length or FLAGS.max_length)
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
      input_vocab_size=padded_vocab_size(inputs_vocab_size()),
      output_vocab_size=padded_vocab_size(targets_vocab_size()),
      max_length=FLAGS.max_length,
      shared_embedding=False,
      shared_embedding_and_softmax_weights=True,
      label_smoothing=FLAGS.label_smoothing,
      layout=FLAGS.layout,
      mesh_shape=FLAGS.mesh_shape)

  inputs = import_feature(features, mesh, "inputs")

  # PREDICT mode
  if mode == tf.estimator.ModeKeys.PREDICT:
    mtf_samples = model.decode(
        inputs,
        # variable_dtype=None,  # TODO(noam)
        beam_size=FLAGS.beam_size,
        alpha=FLAGS.alpha,
        temperature=FLAGS.temperature)
    mtf_samples = mtf.anonymize(mtf_samples)
    lowering = mtf.Lowering(graph, {mesh: mesh_impl})
    outputs = lowering.export_to_tf_tensor(mtf_samples)
    ndims = len(outputs.shape.as_list())
    actual_batch_size = tf.shape(features["inputs"])[0]
    outputs = tf.slice(
        outputs, [0] * ndims, [actual_batch_size] + [-1] * (ndims - 1))
    predictions = {
        "outputs": outputs
    }
    if features.get("inputs") is not None:
      predictions["inputs"] = features["inputs"]
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

  lowering = mtf.Lowering(graph, {mesh: mesh_impl})

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
  encoder = (tfds.builder(FLAGS.dataset).info.features[FLAGS.inputs_feature]
             .encoder)
  all_input_ids = []
  length = FLAGS.length or FLAGS.max_length
  for line in inputs:
    ids = encoder.encode(line.strip())
    ids = add_eos(ids)
    if len(ids) > length:
      ids = ids[:length]
    else:
      ids.extend([0] * (length - len(ids)))
    all_input_ids.append(ids)
  # pad to make an integral number of batches
  all_input_ids.extend([all_input_ids[0]] * (-n % FLAGS.batch_size))
  all_input_ids = np.array(all_input_ids, dtype=np.int32)
  def input_fn(params):
    del params
    dataset = tf.data.Dataset.from_tensor_slices({"inputs": all_input_ids})
    dataset = dataset.batch(FLAGS.batch_size, drop_remainder=True)
    return dataset
  # decodes = []
  result_iter = estimator.predict(input_fn)
  targets_encoder = (
      tfds.builder(FLAGS.dataset).info.features[FLAGS.targets_feature]
      .encoder)
  output_file = tf.gfile.Open(FLAGS.output_file, "w")
  vocab_size = targets_vocab_size()
  for i, result in enumerate(result_iter):
    if i >= n:
      break
    output_ids = clean_output(list(result["outputs"]), vocab_size)
    output_string = targets_encoder.decode(output_ids)
    tf.logging.info(inputs[i])
    tf.logging.info(output_string)
    output_file.write(output_string)
    output_file.write("\n")
  output_file.close()


def main(_):
  """Run training/eval/inference."""
  cluster = tf.contrib.cluster_resolver.TPUClusterResolver(
      tpu=[FLAGS.tpu], zone=None)

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

  if FLAGS.mode == "train":
    estimator.train(
        input_fn=lambda(params): get_dataset(train=True),
        max_steps=FLAGS.train_steps
    )
  elif FLAGS.mode == "evaluate":
    estimator.evaluate(
        input_fn=lambda(params): get_dataset(train=False),
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
