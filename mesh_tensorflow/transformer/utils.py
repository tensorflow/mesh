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

r"""Utilities for running training and inference."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin

import mesh_tensorflow as mtf
from mesh_tensorflow.transformer import transformer
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tpu.python.tpu import tpu_config
from tensorflow.contrib.tpu.python.tpu import tpu_estimator


# TODO(noam): maybe add gin-config to mtf.get_variable so we can delete
#  this stupid VariableDtype class and stop passing it all over creation.
@gin.configurable
def get_variable_dtype(
    master_dtype=tf.bfloat16,
    slice_dtype=tf.float32,
    activation_dtype=tf.float32):
  """Datatypes to use for the run.

  Args:
    master_dtype: string, datatype for checkpoints
      keep this the same between training and eval/inference
    slice_dtype: string, datatype for variables in memory
      must be tf.float32 for training
    activation_dtype: string, datatype for activations
      less memory usage if tf.bfloat16 but possible numerical issues
  Returns:
    a mtf.VariableDtype
  """
  return mtf.VariableDType(
      master_dtype=tf.as_dtype(master_dtype),
      slice_dtype=tf.as_dtype(slice_dtype),
      activation_dtype=tf.as_dtype(activation_dtype))


def build_model(model_type="bitransformer",
                vocab_size=gin.REQUIRED):
  """Build a transformer model.

  Currently, three types of models are supported:

  "bitransformer": The traditional encoder-decoder architecture from
     "attention is all you need".  Requires a non-text2self dataset.

  "lm": an autoregressive language model (one layer stack).  This is similar
     to the decoder part of a bitransformer, but with no attention over an
     encoder, since there is no encoder.  Requires a text2self dataset,
     with targets, but no inputs.

  "aligned": a non-autoregressive single-stack model (like BERT).  Requires
     a non-text2self dataset with inputs and targets.  The targets are
     aligned with the inputs.

  Args:
    model_type: a string - "bitransformer", "lm" or "aligned"
    vocab_size: an integer
  Returns:
    a Unitransformer or Bitransformer
  """
  if model_type == "bitransformer":
    return transformer.make_bitransformer(
        input_vocab_size=vocab_size,
        output_vocab_size=vocab_size)
  elif model_type == "lm" or model_type == "aligned":
    return transformer.Unitransformer(
        autoregressive=model_type == "lm",
        layer_stack=transformer.make_layer_stack(),
        input_vocab_size=vocab_size,
        output_vocab_size=vocab_size)
  else:
    raise ValueError("unknown model_type")


def tpu_estimator_model_fn(model_type,
                           transformer_model,
                           model_dir,
                           use_tpu,
                           mesh_shape,
                           layout_rules,
                           batch_size,
                           sequence_length,
                           autostack):
  """Create a TPUEstimator model function.

  Args:
    model_type: a string
    transformer_model: a transformer.Unitransformer or transformer.Bitransformer
    model_dir: a string
    use_tpu: a boolean
    mesh_shape: a mtf.Shape
    layout_rules: a mtf.LayoutRules
    batch_size: an integer
    sequence_length: an integer
    autostack: a boolean

  Returns:
    a function to be passed to TPUEstimator
  """

  def my_model_fn(features, labels, mode, params=None, config=None):
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
    global_step = tf.train.get_global_step()
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

    def _import_feature(key, allow_missing=False):
      """Import a feature from the features dictionary into a mtf.Tensor.

      Args:
        key: a string
        allow_missing: a boolean

      Returns:
        a mtf.Tensor with dtype int32 and shape [batch_dim, length_dim]
      """
      batch_dim = mtf.Dimension("batch", batch_size)
      length_dim = mtf.Dimension("length", sequence_length)
      mtf_shape = mtf.Shape([batch_dim, length_dim])
      if key not in features:
        if allow_missing:
          return None
        else:
          raise ValueError(
              "feature not found %s - features %s = " % (key, features))
      tf.logging.info("Import feature %s: %s" % (key, features[key]))

      x = tf.to_int32(features[key])
      if not use_tpu:
        x = tf.Print(
            x, [x], "import feature %s" % key, summarize=1000, first_n=1)
      return mtf.import_fully_replicated(mesh, x, mtf_shape, name=key)

    # PREDICT mode
    if mode == tf.estimator.ModeKeys.PREDICT:
      inputs = _import_feature("inputs")
      if isinstance(transformer_model, transformer.Unitransformer):
        mtf_samples = transformer_model.sample_autoregressive(
            inputs, variable_dtype=get_variable_dtype())
      elif isinstance(transformer_model, transformer.Bitransformer):
        mtf_samples = transformer_model.decode(
            inputs, variable_dtype=get_variable_dtype())
      else:
        raise ValueError("unrecognized class")
      mtf_samples = mtf.anonymize(mtf_samples)
      lowering = mtf.Lowering(graph, {mesh: mesh_impl}, autostack=autostack)
      outputs = lowering.export_to_tf_tensor(mtf_samples)
      predictions = {"outputs": outputs}
      return tpu_estimator.TPUEstimatorSpec(
          mode=tf.estimator.ModeKeys.PREDICT,
          predictions=predictions,
          prediction_hooks=[mtf.MtfRestoreHook(lowering)])

    targets = _import_feature("targets")
    anon_targets = mtf.anonymize(targets)
    if model_type == "lm":
      _, length_dim = targets.shape
      inputs = mtf.shift(targets, offset=1, dim=length_dim, wrap=False)
    else:
      inputs = _import_feature("inputs")

    if isinstance(transformer_model, transformer.Unitransformer):
      position_kwargs = dict(
          sequence_id=_import_feature("targets_segmentation", True),
          position=_import_feature("targets_position", True),
      )
    elif isinstance(transformer_model, transformer.Bitransformer):
      position_kwargs = dict(
          encoder_sequence_id=_import_feature("inputs_segmentation", True),
          decoder_sequence_id=_import_feature("targets_segmentation", True),
          encoder_position=_import_feature("inputs_position", True),
          decoder_position=_import_feature("targets_position", True),
      )
    else:
      raise ValueError("unrecognized class")

    logits, loss = transformer_model.call_simple(
        inputs=inputs,
        targets=targets,
        compute_loss=True,
        mode=mode,
        variable_dtype=get_variable_dtype(),
        **position_kwargs)

    if use_tpu and logits is not None:
      logits = mtf.anonymize(logits)

    # TRAIN mode
    if mode == tf.estimator.ModeKeys.TRAIN:
      var_grads = mtf.gradients(
          [loss], [v.outputs[0] for v in graph.trainable_variables])
      optimizer = mtf.optimize.AdafactorOptimizer()
      update_ops = optimizer.apply_grads(var_grads, graph.trainable_variables)

    lowering = mtf.Lowering(graph, {mesh: mesh_impl}, autostack=autostack)

    tf_loss = lowering.export_to_tf_tensor(loss)
    tf_loss = tf.to_float(tf_loss)
    if not use_tpu:
      tf_loss = tf.Print(tf_loss, [tf_loss, tf.train.get_global_step()],
                         "step, tf_loss")
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
          model_dir, save_steps=1000, saver=saver, listeners=[saver_listener])

      if mode == tf.estimator.ModeKeys.TRAIN:
        if use_tpu:
          return tpu_estimator.TPUEstimatorSpec(
              mode=tf.estimator.ModeKeys.TRAIN,
              loss=tf_loss,
              train_op=train_op,
              training_hooks=[restore_hook, saver_hook])
        else:
          return tf.estimator.EstimatorSpec(
              tf.estimator.ModeKeys.TRAIN,
              loss=tf_loss,
              train_op=train_op,
              training_chief_hooks=[restore_hook, saver_hook])
      elif mode == tf.estimator.ModeKeys.EVAL:
        # TODO(katherinelee): specify eval_metrics.
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

  return my_model_fn


@gin.configurable
def decode_from_file(estimator,
                     vocabulary,
                     model_type,
                     batch_size,
                     sequence_length,
                     checkpoint_path="",
                     input_filename=gin.REQUIRED,
                     output_filename=gin.REQUIRED):
  """Decode from a text file.

  Args:
    estimator: a TPUEstimator
    vocabulary: a mtf.transformer.vocabulary.Vocabulary
    model_type: a string
    batch_size: an integer
    sequence_length: an integer (maximum decode length)
    checkpoint_path: an optional string
    input_filename: a string
    output_filename: a string
  """
  with tf.gfile.Open(input_filename) as f:
    text = f.read()
  records = text.split("\n")
  inputs = [record.strip() for record in records]
  # Strip the last empty line.
  if not inputs[-1]:
    inputs.pop()
  n = len(inputs)
  # encode all inputs
  all_input_ids = []
  for line in inputs:
    ids = vocabulary.encode(line.strip())
    if model_type != "lm":
      # for text2self problems, the inputs represent a partial sequence
      # to be continued, and should not be terminated by EOS.
      # for sequence-to-sequence problems, the input needs to be EOS-terminated
      ids += [1]
    if len(ids) > sequence_length:
      ids = ids[:sequence_length]
    else:
      ids.extend([0] * (sequence_length - len(ids)))
      all_input_ids.append(ids)
      # pad to make an integral number of batches
  all_input_ids.extend([all_input_ids[0]] * (-n % batch_size))
  padded_n = len(all_input_ids)
  all_input_ids = np.array(all_input_ids, dtype=np.int32)

  def input_fn(params):
    del params
    dataset = tf.data.Dataset.from_tensor_slices({"inputs": all_input_ids})
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset

  result_iter = estimator.predict(input_fn, checkpoint_path=checkpoint_path)
  vocab_size = vocabulary.vocab_size
  decodes = []
  for i, result in enumerate(result_iter):
    output_ids = clean_decodes(list(result["outputs"]), vocab_size)
    output_string = vocabulary.decode([int(x) for x in output_ids])
    decodes.append(output_string)
    if i & (i - 1) == 0:
      # LOG every power of 2
      tf.logging.info("decode %d input = %s" % (i, inputs[i]))
      tf.logging.info("          output = %s" % output_string)
  # BUG WORKAROUND - on TF1.13 and earlier, the output for each batch is
  # repeated a number of times equal to the number of cores.
  if len(decodes) == padded_n:
    tf.logging.info("number of decodes matches number of inputs")
  elif len(decodes) % padded_n == 0:
    num_cores = len(decodes) // padded_n
    tf.logging.info("output is repeated num_cores times - removing extras")

    def keep(i):
      return i % (batch_size * num_cores) < batch_size

    decodes = [d for i, d in enumerate(decodes) if keep(i)]
  else:
    raise ValueError("unexpected number of outputs")
  output_file = tf.gfile.Open(output_filename, "w")
  decodes = decodes[:n]
  for d in decodes:
    output_file.write(d)
    output_file.write("\n")
  output_file.close()


def clean_decodes(ids, vocab_size):
  """Stop at EOS or padding or OOV.

  Args:
    ids: a list of integers
    vocab_size: an integer

  Returns:
    a list of integers
  """
  ret = []
  for i in ids:
    if i <= 1 or i >= vocab_size:
      break
    else:
      ret.append(int(i))
  return ret


@gin.configurable
def auto_batch_size(sequence_length,
                    mesh_shape,
                    layout_rules,
                    tokens_per_split=2048):
  """Automatically compute batch size.

  Args:
    sequence_length: an integer
    mesh_shape: an input to mtf.convert_to_shape()
    layout_rules: an input to mtf.convert_to_layout_rules()
    tokens_per_split: an integer
  Returns:
    an integer
  """
  num_splits = mtf.tensor_dim_to_mesh_dim_size(
      layout_rules, mesh_shape, mtf.Dimension("batch", 0))
  ret = (tokens_per_split // sequence_length) * num_splits
  tf.logging.info(
      "AUTO_BATCH_SIZE tokens_per_split=%s num_splits=%s"
      " sequence_length=%s batch_size=%s"
      % (tokens_per_split, num_splits, sequence_length, ret))
  return ret


@gin.configurable
def run(tpu_job_name,
        tpu,
        gcp_project,
        tpu_zone,
        model_dir,
        model_type="bitransformer",
        vocabulary=gin.REQUIRED,
        dataset_fn=gin.REQUIRED,
        dataset_split="train",
        autostack=True,
        checkpoint_path="",
        mode="train",
        iterations_per_loop=100,
        save_checkpoints_steps=1000,
        eval_steps=10,
        train_steps=1000000,
        batch_size=auto_batch_size,
        sequence_length=gin.REQUIRED,
        mesh_shape=gin.REQUIRED,
        layout_rules=gin.REQUIRED):
  """Run training/eval/inference.

  Args:
    tpu_job_name: string, name of TPU worker binary
    tpu: string, the Cloud TPU to use for training
    gcp_project: string, project name for the Cloud TPU-enabled project
    tpu_zone: string, GCE zone where the Cloud TPU is located in
    model_dir: string, estimator model_dir
    model_type: a string - either "bitransformer", "lm" or "aligned"
    vocabulary: a vocabulary.Vocabulary
    dataset_fn: A function returning a tf.data.Dataset
    dataset_split: a string
    autostack: boolean, internally combine variables
    checkpoint_path: a string - which checkpoint to load for inference
    mode: string, train/evaluate/infer
    iterations_per_loop: integer, steps per train loop
    save_checkpoints_steps: integer, steps per checkpoint
    eval_steps: integer, number of evaluation steps
    train_steps: Total number of training steps.
    batch_size: An integer or a function with the same signature as
      auto_batch_size().  Mini-batch size for the training. Note that this is
      the global batch size and not the per-shard batch size.
    sequence_length: an integer
    mesh_shape: an input to mtf.convert_to_shape()
    layout_rules: an input to mtf.convert_to_layout_rules()
  """
  if not isinstance(batch_size, int):
    batch_size = batch_size(sequence_length, mesh_shape, layout_rules)

  tf.logging.info("mode=%s" % mode,)
  tf.logging.info("batch_size=%s" % batch_size,)
  tf.logging.info("sequence_length=%s" % sequence_length,)
  tf.logging.info("mesh_shape=%s" % mesh_shape,)
  tf.logging.info("layout_rules=%s" % layout_rules,)

  if mode == "train" and dataset_split != "train":
    raise ValueError("mode==\"train\" requires dataset_split==\"train\"")

  mesh_shape = mtf.convert_to_shape(mesh_shape)
  layout_rules = mtf.convert_to_layout_rules(layout_rules)

  cluster = tf.contrib.cluster_resolver.TPUClusterResolver(
      tpu if (tpu) else "", zone=tpu_zone, project=gcp_project)

  my_tpu_config = tpu_config.TPUConfig(
      tpu_job_name=tpu_job_name,
      iterations_per_loop=iterations_per_loop,
      num_cores_per_replica=1,
      per_host_input_for_training=tpu_config.InputPipelineConfig.BROADCAST,
  )

  run_config = tpu_config.RunConfig(
      cluster=cluster,
      model_dir=model_dir,
      save_checkpoints_steps=save_checkpoints_steps,
      tpu_config=my_tpu_config)

  transformer_model = build_model(
      model_type=model_type,
      vocab_size=vocabulary.vocab_size)

  model_fn = tpu_estimator_model_fn(
      model_type=model_type,
      transformer_model=transformer_model,
      model_dir=model_dir,
      use_tpu=tpu,
      mesh_shape=mesh_shape,
      layout_rules=layout_rules,
      batch_size=batch_size,
      sequence_length=sequence_length,
      autostack=autostack)

  estimator = tpu_estimator.TPUEstimator(
      model_fn=model_fn,
      config=run_config,
      train_batch_size=batch_size,
      eval_batch_size=batch_size,
      predict_batch_size=batch_size,
      use_tpu=tpu,
      export_to_tpu=False,
      params={})

  def input_fn(params):
    del params
    dataset = dataset_fn(batch_size=batch_size,
                         sequence_length=sequence_length,
                         vocabulary=vocabulary,
                         dataset_split=dataset_split)
    return dataset

  if mode == "train":
    estimator.train(input_fn=input_fn, max_steps=train_steps)
  elif mode == "evaluate":
    estimator.evaluate(
        input_fn=input_fn,
        steps=eval_steps,
    )
  elif mode == "continuous_eval":
    eval_args = {"eval": (input_fn, eval_steps)}
    continuous_eval(estimator, eval_args)
  elif mode == "infer":
    decode_from_file(
        estimator,
        vocabulary=vocabulary,
        model_type=model_type,
        batch_size=batch_size,
        sequence_length=sequence_length,
        checkpoint_path=checkpoint_path)
  else:
    raise ValueError(
        "unknown mode %s - must be train/evaluate/continuous_eval/infer" % mode)


def continuous_eval(estimator, eval_args):
  """Runs evaluation whenever there's a new checkpoint & logs to tensorboard.

  Args:
    estimator: A tf.Estimator object.
    eval_args: Dictionary of {eval_name: (input_fn, eval_steps)} where eval_name
      is the name of the evaluation set, e.g. "train" or "val", input_fn is an
      input function returning a tuple (features, labels), and eval_steps is the
      number of steps for which to evaluate the model. If None, evaluates until
      input_fn raises an end-of-input exception.
  """
  for _ in tf.contrib.training.checkpoints_iterator(estimator.model_dir):
    _ = evaluate(estimator, eval_args)


def evaluate(estimator, eval_args):
  """Runs evaluation on the latest model checkpoint & logs to tensorboard.

  Args:
    estimator: A tf.Estimator object.
    eval_args: Dictionary of {eval_name: (input_fn, eval_steps)} where eval_name
      is the name of the evaluation set, e.g. "train" or "val", input_fn is an
      input function returning a tuple (features, labels), and eval_steps is the
      number of steps for which to evaluate the model. If None, evaluates until
      input_fn raises an end-of-input exception.

  Returns:
    A dict of metric values from the evaluation. May be empty, e.g. if the
    training job has not yet saved a checkpoint or the checkpoint is deleted by
    the time the TPU worker initializes.
  """
  values = {}  # Default return value if evaluation fails.

  checkpoint_path = estimator.latest_checkpoint()
  if not checkpoint_path:
    # This is expected if the training job has not yet saved a checkpoint.
    return values

  tf.logging.info("Starting evaluation on checkpoint %s", checkpoint_path)
  for eval_name in eval_args:
    input_fn, eval_steps = eval_args[eval_name]
    metric_values = estimator.evaluate(
        input_fn,
        steps=eval_steps,
        name=eval_name,
        checkpoint_path=checkpoint_path)
    for key, val in metric_values.iteritems():
      values[eval_name + "/" + key] = val

  tf.logging.info(values)
  return values
