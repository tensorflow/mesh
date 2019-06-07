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

import functools
import math
import os

import gin
import gin.tf

import mesh_tensorflow as mtf
from mesh_tensorflow.transformer import transformer
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from tensorflow.contrib.tpu.python.tpu import tpu_config
from tensorflow.contrib.tpu.python.tpu import tpu_estimator

tf.flags.DEFINE_multi_string("gin_file", None,
                             "List of paths to the config files.")
tf.flags.DEFINE_multi_string(
    "gin_param", None, "Newline separated list of Gin parameter bindings.")
FLAGS = tf.flags.FLAGS

_DEFAULT_CONFIG_FILE = "./gin/defaults.gin"


def parse_gin_defaults_and_flags():
  """Parses all default gin files and those provides via flags."""
  # Set up the default values for the configurable parameters. These values will
  # be overridden by any user provided gin files/parameters.
  gin.parse_config_file(
      os.path.join(os.path.dirname(__file__), _DEFAULT_CONFIG_FILE))
  gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)


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


def inputs_vocabulary(vocabulary):
  """Inputs vocabulary.

  Args:
    vocabulary: Vocabulary or (inputs_vocabulary, targets_vocabulary) tuple.

  Returns:
    a Vocabulary
  """
  if isinstance(vocabulary, tuple):
    vocabulary = vocabulary[0]
  return vocabulary


def targets_vocabulary(vocabulary):
  """Targets vocabulary.

  Args:
    vocabulary: Vocabulary or (inputs_vocabulary, targets_vocabulary) tuple.

  Returns:
    a Vocabulary
  """
  if isinstance(vocabulary, tuple):
    vocabulary = vocabulary[1]
  return vocabulary


@gin.configurable
def separate_vocabularies(inputs=gin.REQUIRED, targets=gin.REQUIRED):
  """Gin-configurable helper function."""
  return (inputs, targets)


def build_model(model_type="bitransformer",
                input_vocab_size=gin.REQUIRED,
                output_vocab_size=gin.REQUIRED,
                layout_rules=None,
                mesh_shape=None):
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
    input_vocab_size: an integer
    output_vocab_size: an integer
    layout_rules: optional - an input to mtf.convert_to_layout_rules
    mesh_shape: optional - an input to mtf.convert_to_shape
  Returns:
    a Unitransformer or Bitransformer
  """
  if model_type == "bitransformer":
    return transformer.make_bitransformer(
        input_vocab_size=input_vocab_size,
        output_vocab_size=output_vocab_size,
        mesh_shape=mesh_shape,
        layout=layout_rules)
  elif model_type == "lm" or model_type == "aligned":
    return transformer.Unitransformer(
        autoregressive=model_type == "lm",
        layer_stack=transformer.make_layer_stack(),
        input_vocab_size=input_vocab_size,
        output_vocab_size=output_vocab_size,
        mesh_shape=mesh_shape,
        layout=layout_rules)
  else:
    raise ValueError("unknown model_type")


@gin.configurable
def tpu_mesh_shape(tpu_topology=gin.REQUIRED,
                   model_parallelism=gin.REQUIRED):
  """Create a mesh_shape for data-parallelism and model-parallelism on TPU.

  Example: tpu_mesh_shape("4x4", 8) -> mtf.Shape(("batch", 4), ("model", 8))
  Since there are 4x4x2=32 total cores, and we want 8-way model paralleism.

  Args:
    tpu_topology: a string - e.g. "2x2"
    model_parallelism: an integer - the number of cores per model replica
  Returns:
    a mtf.Shape
  """
  x, y = tpu_topology.split("x")
  num_cores = int(x) * int(y) * 2
  data_parallelism = num_cores // model_parallelism
  dims = []
  if data_parallelism > 1:
    dims.append(mtf.Dimension("batch", data_parallelism))
  if model_parallelism > 1:
    dims.append(mtf.Dimension("model", model_parallelism))
  return mtf.Shape(dims)


def _logical_to_physical(physical_shape, mesh_shape):
  """Mapping from logical mesh to physical TPU cores.

  This is to create the logical_to_physical mapping for SimdMeshImpl.  For 2d
  meshes, we use a tiled layout with the second logical mesh-dimension
  corresponding to position within a tile.  This tends to give better
  performance.  For non-2d meshes, we use the default mapping.

  Args:
    physical_shape: a list of integers - the physical mesh shape
    mesh_shape: a mtf.Shape
  Returns:
    a permutation of range(mesh_shape.size) or None
  """
  mesh_shape = mesh_shape.to_integer_list
  if len(mesh_shape) != 2:
    return None
  # Use "tiled" mapping of logical mesh to physical mesh.
  # The first logical-mesh dimension corresponds to which phyiscal tile
  # and the second logical-mesh dimension corresponds to location within
  # a tile.
  tile_size = mesh_shape[1] // 2  # size in chips (each with 2 cores)
  lg_tile_size = int(math.log(tile_size, 2))
  t0 = 2 ** (lg_tile_size // 2)
  t1 = tile_size // t0
  tile_shape = [t0, t1]
  tf.logging.info("Mesh shape = %s" % mesh_shape)
  tf.logging.info("Physical shape = %s" % physical_shape)
  tf.logging.info("Tile shape = %s" % tile_shape)
  _, logical_to_physical = mtf.simd_mesh_impl.tile_2d(
      physical_shape, tile_shape)
  tf.logging.info("logical_to_physical = %s" % (logical_to_physical,))
  return logical_to_physical


@gin.configurable
def tpu_estimator_model_fn(model_type,
                           transformer_model,
                           model_dir,
                           use_tpu,
                           mesh_shape,
                           layout_rules,
                           batch_size,
                           sequence_length,
                           autostack,
                           checkpoints_to_keep,
                           save_steps,
                           learning_rate_schedule=None,
                           optimizer=None,
                           outer_batch_size=1,
                           tpu_summaries=False):
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
    checkpoints_to_keep: an integer
    save_steps: an integer
    learning_rate_schedule: an optional function taking the scalar named
      argument `step` and return the scalar learning rate.
      Alternatively, a constant.
    optimizer: a class extending optimize.Optimizer, required for training
    outer_batch_size: outer batch dimension that could be used to enable the mix
      of data-parallel and model-parallel training of MoE models
    tpu_summaries: a boolean - if True, then use rewrites to make summaries
      work on TPU.  This may be slow, since it uses a host call hack.

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
      physical_shape = list(
          params["context"].device_assignment.topology.mesh_shape)
      logical_to_physical = _logical_to_physical(physical_shape, mesh_shape)
      mesh_impl = mtf.simd_mesh_impl.SimdMeshImpl(
          mesh_shape, layout_rules, mesh_devices, ctx.device_assignment,
          logical_to_physical=logical_to_physical)
    else:
      var_placer = None
      mesh_devices = [""] * mesh_shape.size
      mesh_impl = mtf.placement_mesh_impl.PlacementMeshImpl(
          mesh_shape, layout_rules, mesh_devices)

    graph = mtf.Graph()
    mesh = mtf.Mesh(graph, "my_mesh", var_placer)

    outer_batch_dim = mtf.Dimension("outer_batch", outer_batch_size)
    batch_dim = mtf.Dimension("batch", batch_size // outer_batch_size)
    length_dim = mtf.Dimension("length", sequence_length)
    feature_shape = mtf.Shape([outer_batch_dim, batch_dim, length_dim])

    mtf_features = {}
    for key, x in features.items():
      x = tf.to_int32(features[key])
      x = tf.reshape(x, [outer_batch_size, batch_size // outer_batch_size, -1])
      if not use_tpu:
        x = tf.Print(
            x, [x], "import feature %s" % key, summarize=1000, first_n=1)
      mtf_features[key] = mtf.import_fully_replicated(
          mesh, x, feature_shape, name=key)

    if mode == tf.estimator.ModeKeys.PREDICT:
      inputs = mtf_features["inputs"]
      inputs = mtf.reshape(
          inputs,
          mtf.Shape([
              mtf.Dimension("batch", batch_size),
              mtf.Dimension("length", sequence_length)
          ]))
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

    elif mode == tf.estimator.ModeKeys.EVAL:
      raise NotImplementedError("We don't expect to use mode == eval.")

    else:
      assert mode == tf.estimator.ModeKeys.TRAIN
      num_microbatches = serialize_num_microbatches(batch_dim,
                                                    length_dim,
                                                    mesh_shape,
                                                    layout_rules)
      def model_fn(mtf_features):
        """The kind of function we need for mtf.serialize_training_step.

        Args:
          mtf_features: a dictionary
        Returns:
          a dictionary
        """
        targets = mtf_features["targets"]
        if model_type == "lm":
          _, length_dim = targets.shape
          inputs = mtf.shift(targets, offset=1, dim=length_dim, wrap=False)
        else:
          inputs = mtf_features["inputs"]

        if isinstance(transformer_model, transformer.Unitransformer):
          position_kwargs = dict(
              sequence_id=mtf_features.get("targets_segmentation", None),
              position=mtf_features("targets_position", None),
          )
        elif isinstance(transformer_model, transformer.Bitransformer):
          position_kwargs = dict(
              encoder_sequence_id=mtf_features.get(
                  "inputs_segmentation", None),
              decoder_sequence_id=mtf_features.get(
                  "targets_segmentation", None),
              encoder_position=mtf_features.get(
                  "inputs_position", None),
              decoder_position=mtf_features.get(
                  "targets_position", None),
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
        if num_microbatches > 1:
          loss /= float(num_microbatches)
        del logits
        return {"loss": loss}

      if num_microbatches > 1:
        var_grads, loss_dict = mtf.serialize_training_step(
            mtf_features, model_fn, batch_dim, num_microbatches)
      else:
        loss_dict = model_fn(mtf_features)
        var_grads = mtf.gradients(
            [loss_dict["loss"]],
            [v.outputs[0] for v in graph.trainable_variables])

      loss = loss_dict["loss"]

      if callable(learning_rate_schedule):
        # the following happens on CPU since TPU can't handle summaries.
        with mtf.utils.outside_all_rewrites():
          learning_rate = learning_rate_schedule(
              step=tf.train.get_global_step())
          tf.summary.scalar("learning_rate", learning_rate)
      else:
        learning_rate = learning_rate_schedule

      update_ops = optimizer(learning_rate=learning_rate).apply_grads(
          var_grads, graph.trainable_variables)

      lowering = mtf.Lowering(graph, {mesh: mesh_impl}, autostack=autostack)

      tf_loss = lowering.export_to_tf_tensor(loss)
      tf_loss = tf.to_float(tf_loss)
      if not use_tpu:
        tf_loss = tf.Print(tf_loss, [tf_loss, tf.train.get_global_step()],
                           "step, tf_loss")

      tf_update_ops = [lowering.lowered_operation(op) for op in update_ops]
      tf_update_ops.append(tf.assign_add(global_step, 1))
      train_op = tf.group(tf_update_ops)

      with mtf.utils.outside_all_rewrites():
        # Copy master variables to slices. Must be called first.
        restore_hook = mtf.MtfRestoreHook(lowering)
        saver = tf.train.Saver(
            tf.global_variables(),
            sharded=True,
            max_to_keep=checkpoints_to_keep,
            keep_checkpoint_every_n_hours=2,
            defer_build=False,
            save_relative_paths=True)
        tf.add_to_collection(tf.GraphKeys.SAVERS, saver)
        saver_listener = mtf.MtfCheckpointSaverListener(lowering)
        saver_hook = tf.train.CheckpointSaverHook(
            model_dir,
            save_steps=save_steps,
            saver=saver,
            listeners=[saver_listener])
        gin_config_saver_hook = gin.tf.GinConfigSaverHook(
            model_dir, summarize_config=True)

        if use_tpu:
          if tpu_summaries:
            tf.summary.scalar("loss", tf_loss)
            host_call = mtf.utils.create_host_call(model_dir)
            mtf.utils.remove_summaries()
          else:
            host_call = None
          return tpu_estimator.TPUEstimatorSpec(
              mode=tf.estimator.ModeKeys.TRAIN,
              loss=tf_loss,
              train_op=train_op,
              host_call=host_call,
              training_hooks=[
                  restore_hook,
                  saver_hook,
                  gin_config_saver_hook,
              ])
        else:
          return tf.estimator.EstimatorSpec(
              tf.estimator.ModeKeys.TRAIN,
              loss=tf_loss,
              train_op=train_op,
              training_chief_hooks=[
                  restore_hook,
                  saver_hook,
                  gin_config_saver_hook,
              ])

  return my_model_fn


def get_inputs_from_file(input_filename):
  """Read data from file and strip new lines."""
  inputs = [line.rstrip() for line in tf.io.gfile.GFile(input_filename)]

  # Strip the last empty line.
  if not inputs[-1]:
    inputs.pop()
  return inputs


def encode_inputs(inputs,
                  vocabulary,
                  model_type,
                  batch_size,
                  sequence_length,
                  eos_id=1):
  """Encode inputs.

  Args:
    inputs: list of strings
    vocabulary: a mtf.transformer.vocabulary.Vocabulary
    model_type: a string
    batch_size: an integer
    sequence_length: an integer (maximum decode length)
    eos_id: EOS id

  Returns:
    all_input_ids: encoded inputs
  """
  n = len(inputs)
  all_input_ids = []
  for line in inputs:
    ids = inputs_vocabulary(vocabulary).encode(line.strip())
    if model_type != "lm":
      # for text2self problems, the inputs represent a partial sequence
      # to be continued, and should not be terminated by EOS.
      # for sequence-to-sequence problems, the input needs to be EOS-terminated
      ids += [eos_id]
    if len(ids) > sequence_length:
      ids = ids[:sequence_length]
    else:
      ids.extend([0] * (sequence_length - len(ids)))
    all_input_ids.append(ids)
  # pad to make an integral number of batches
  all_input_ids.extend([all_input_ids[0]] * (-n % batch_size))
  all_input_ids = np.array(all_input_ids, dtype=np.int32)

  return all_input_ids


@gin.configurable
def decode(estimator,
           input_fn,
           dataset_size,
           padded_dataset_size,
           batch_size,
           vocabulary,
           checkpoint_path=""):
  """Decode from an input_fn.

  Args:
    estimator: a TPUEstimator
    input_fn: function that returns a tf.Dataset
    dataset_size: number of examples in the dataset
    padded_dataset_size: number of examples in the padded dataset
    batch_size: an integer
    vocabulary: a mtf.transformer.vocabulary.Vocabulary
    checkpoint_path: an optional string

  Returns:
    list of decoded strings
  """
  result_iter = estimator.predict(input_fn,
                                  checkpoint_path=checkpoint_path)
  vocab_size = targets_vocabulary(vocabulary).vocab_size
  decodes = []
  for i, result in enumerate(result_iter):
    output_ids = clean_decodes(list(result["outputs"]), vocab_size)
    output_string = targets_vocabulary(vocabulary).decode(
        [int(x) for x in output_ids])
    decodes.append(output_string)
    if i & (i - 1) == 0:
      # LOG every power of 2.
      tf.logging.info("decoded {}: {}".format(i, output_string))

  # BUG WORKAROUND - on TF1.13 and earlier, the output for each batch is
  # repeated a number of times equal to the number of cores.
  tf.logging.info(
      "num examples in dataset (dataset_size): %d\n"
      "num_examples in padded dataset (padded_dataset_size): %d"
      "len(decodes): %d",
      dataset_size, padded_dataset_size, len(decodes))
  if len(decodes) == padded_dataset_size:
    tf.logging.info("number of decodes matches number of inputs")
  elif len(decodes) % padded_dataset_size == 0:
    num_cores = len(decodes) // padded_dataset_size
    tf.logging.info("output is repeated num_cores times - removing extras")

    def keep(i):
      return i % (batch_size * num_cores) < batch_size

    decodes = [d for i, d in enumerate(decodes) if keep(i)]

  # Since we replicate a batch enough times to fill the min_dataset_size, this
  # might not be an integer number of repeats. So we take the first dataset_size
  # examples.
  if len(decodes) != dataset_size:
    tf.logging.info("Taking the first %d examples of %d",
                    dataset_size, len(decodes))
    decodes = decodes[:dataset_size]
  return decodes


def log_pred_target(decodes, dataset, dataset_size, vocabulary,
                    pred_output_filename=None,
                    target_output_filename=None):
  """Log predictions and targets."""

  vocab_size = targets_vocabulary(vocabulary).vocab_size

  # Write out predicted strings
  if pred_output_filename is not None:
    if tf.io.gfile.exists(pred_output_filename):
      tf.io.gfile.remove(pred_output_filename)
    with tf.io.gfile.GFile(pred_output_filename, "w") as output_file:
      for d in decodes:
        output_file.write(d + "\n")

  # Write out targets
  if target_output_filename is not None:
    if dataset is None:
      raise ValueError(
          "Must provide a dataset if target_output_filename is set.")
    dataset = tfds.as_numpy(dataset)
    count = 0
    # Break through both for-loops by setting broken to be True.
    broken = False
    if tf.io.gfile.exists(target_output_filename):
      tf.io.gfile.remove(target_output_filename)
    with tf.io.gfile.GFile(target_output_filename, "w") as output_file:
      for input_target in dataset:
        if "targets" not in input_target:
          raise ValueError("The input_fn provided to decode does not have a "
                           "value for \"targets\"")
        for ids in input_target["targets"]:
          output_ids = clean_decodes(ids, vocab_size)
          output_string = targets_vocabulary(vocabulary).decode(
              [int(x) for x in output_ids])
          output_file.write(output_string + "\n")
          count += 1
          if count >= dataset_size:
            broken = True
            break
        if broken:
          break

  return decodes


@gin.configurable
def decode_from_file(estimator,
                     vocabulary,
                     model_type,
                     batch_size,
                     sequence_length,
                     checkpoint_path="",
                     input_filename=gin.REQUIRED,
                     output_filename=gin.REQUIRED,
                     eos_id=1):
  """Decode from a text file and write to output_filename.

  Args:
    estimator: a TPUEstimator
    vocabulary: a mtf.transformer.vocabulary.Vocabulary
    model_type: a string
    batch_size: an integer
    sequence_length: an integer (maximum decode length)
    checkpoint_path: an optional string
    input_filename: a string
    output_filename: a string
    eos_id: EOS id
  """
  inputs = get_inputs_from_file(input_filename)

  all_input_ids = encode_inputs(inputs, vocabulary, model_type, batch_size,
                                sequence_length, eos_id=eos_id)
  def input_fn(params):
    del params
    dataset = tf.data.Dataset.from_tensor_slices({"inputs": all_input_ids})
    dataset = dataset.batch(batch_size, drop_remainder=True)
    return dataset

  dataset_size = len(inputs)
  padded_dataset_size = len(all_input_ids)

  decodes = decode(estimator, input_fn, dataset_size, padded_dataset_size,
                   batch_size, vocabulary, checkpoint_path=checkpoint_path)
  log_pred_target(decodes, None, dataset_size, vocabulary,
                  pred_output_filename=output_filename)


@gin.configurable
def clean_decodes(ids, vocab_size, eos_id=1):
  """Stop at EOS or padding or OOV.

  Args:
    ids: a list of integers
    vocab_size: an integer
    eos_id: EOS id

  Returns:
    a list of integers
  """
  ret = []
  for i in ids:
    if i == eos_id:
      break
    if i >= vocab_size:
      break
    ret.append(int(i))
  return ret


def compute_batch_size(sequence_length,
                       mesh_shape,
                       layout_rules,
                       method_and_value):
  """Compute the total batch size in sequences.

  method_and_value is a (string, int) pair.
  The method string is one of the following four options:

  "sequences_per_batch"
  "tokens_per_batch"
  "sequences_per_replica"
  "tokens_per_replica"

  According to the method string, the value represents either a number of
  sequences or a number of tokens, and represents either the size of the total
  batch or the fraction of the batch assigned to each model replica.

  For example ("tokens_per_replica", 2048) means that the batch size should be
  set so that the number of tokens per model replica is 2048.  So if the
  sequence length is 1024 and there is 16-way data-parallelism, then the number
  of sequences per batch would be 2048 * 16 / 1024 = 32.

  The "per_batch" versions are useful for ensuring indentical overall batch
  sizes across different mesh shapes/layouts.  The "per_replica" versions are
  useful for scaling up the total batch size relative to the degree of
  data-parallelism

  Args:
    sequence_length: an integer
    mesh_shape: an input to mtf.convert_to_shape()
    layout_rules: an input to mtf.convert_to_layout_rules()
    method_and_value: a pair
  Returns:
    an integer - the number of sequences per batch
  """
  def checkdiv(a, b):
    if a % b:
      raise ValueError("%d is not divisible by %d" % (a, b))
    return a // b
  num_replicas = (
      mtf.tensor_dim_to_mesh_dim_size(
          layout_rules, mesh_shape, mtf.Dimension("batch", 0)) *
      mtf.tensor_dim_to_mesh_dim_size(
          layout_rules, mesh_shape, mtf.Dimension("outer_batch", 0)))
  method, value = method_and_value
  if method == "sequences_per_batch":
    return value
  elif method == "tokens_per_batch":
    return checkdiv(value, sequence_length)
  elif method == "sequences_per_replica":
    return value * num_replicas
  elif method == "tokens_per_replica":
    return checkdiv(value, sequence_length) * num_replicas
  else:
    raise ValueError("unknown method %s" % method,)


@gin.configurable
def serialize_num_microbatches(batch_dim,
                               length_dim,
                               mesh_shape,
                               layout_rules,
                               tokens_per_microbatch_per_replica=None):
  """Number of microbatches per batch for serialized training.

  We want to split each training step into multiple sequential steps
  to limit memory usage.  Gradients are accumulated locally and reduced once.

  This function determines the number of microbatches per batch.
  If tokens_per_microbatch_per_replica=None, then the batch is not split.

  Args:
    batch_dim: a mtf.Dimension
    length_dim: a mtf.Dimension
    mesh_shape: an input to mtf.convert_to_shape()
    layout_rules: an input to mtf.convert_to_layout_rules()
    tokens_per_microbatch_per_replica: an optional integer, e.g. 2048
  Returns:
    an integer
  """
  if not tokens_per_microbatch_per_replica:
    return 1
  batch_per_replica = mtf.tensor_dim_to_size_per_split(
      layout_rules, mesh_shape, batch_dim)
  # number of sequences per microbatch
  microbatch_size = max(1, tokens_per_microbatch_per_replica // length_dim.size)
  # decrease microbatch_size until it is a divisor of batch_per_replica
  # This is guaranteed to stop at microbatch_size=1 if not earlier.
  while batch_per_replica % microbatch_size:
    microbatch_size -= 1
  num_microbatches = batch_per_replica // microbatch_size
  tf.logging.info(
      "serialize_num_microbatches: "
      "tokens_per_microbatch_per_replica=%d "
      "batch_dim=%s "
      "length_dim=%s "
      "batch_per_replica=%d "
      "num_microbatches=%d",
      tokens_per_microbatch_per_replica,
      batch_dim,
      length_dim,
      batch_per_replica,
      num_microbatches)
  return num_microbatches


@gin.configurable
def auto_train_steps(batch_size,
                     sequence_length,
                     train_tokens=2 ** 36):
  """Automatically compute number of training steps.

  Since the batch size and sequence length can vary across experiments, we
  specify the amount of training in terms of (non-unique) input tokens processed
  over the course of training the model.  The number of steps is computed as

    train_steps = train_tokens // (batch_size * sequence_length)

  Args:
    batch_size: an integer
    sequence_length: an integer
    train_tokens: an integer (train_steps * batch_size * sequence_length)
  Returns:
    an integer
  """
  return train_tokens // (batch_size * sequence_length)


@gin.configurable
def run(tpu_job_name,
        tpu,
        gcp_project,
        tpu_zone,
        model_dir,
        model_type="bitransformer",
        vocabulary=gin.REQUIRED,
        train_dataset_fn=None,
        eval_dataset_fn=None,
        dataset_split="train",
        autostack=True,
        checkpoint_path="",
        mode="train",
        iterations_per_loop=100,
        save_checkpoints_steps=1000,
        batch_size=("tokens_per_replica", 2048),
        train_steps=auto_train_steps,
        sequence_length=gin.REQUIRED,
        mesh_shape=gin.REQUIRED,
        layout_rules=gin.REQUIRED,
        get_components_fn=None,
        compute_metrics=None,
        checkpoints_to_keep=10,
        save_steps=1000,
        learning_rate_schedule=None,
        optimizer=None):
  """Run training/eval/inference.

  Args:
    tpu_job_name: string, name of TPU worker binary
    tpu: string, the Cloud TPU to use for training
    gcp_project: string, project name for the Cloud TPU-enabled project
    tpu_zone: string, GCE zone where the Cloud TPU is located in
    model_dir: string, estimator model_dir
    model_type: a string - either "bitransformer", "lm" or "aligned"
    vocabulary: a vocabulary.Vocabulary or
      (inputs_vocabulary, targets_vocabulary) tuple.
    train_dataset_fn: A function returning a tf.data.Dataset. Must be provided
      for mode=train
    eval_dataset_fn: A function returning a tf.data.Dataset. Must be provided
      for model=eval
    dataset_split: a string
    autostack: boolean, internally combine variables
    checkpoint_path: a string - which checkpoint to load for inference
    mode: string, train/evaluate/infer
    iterations_per_loop: integer, steps per train loop
    save_checkpoints_steps: integer, steps per checkpoint
    batch_size: An integer or a (method, value) pair to pass to
      compute_batch_size(). Note that this is
      the global batch size and not the per-shard batch size.
    train_steps: An integer or a function with the same signature as
      auto_train_steps().  Total number of training steps.
    sequence_length: an integer
    mesh_shape: an input to mtf.convert_to_shape()
    layout_rules: an input to mtf.convert_to_layout_rules()
    get_components_fn: an optional function that takes in a component and
      returns a list of tuples of (metric_names, component) for each component.
      Required if mode is "continuous_eval."
    compute_metrics: an optional function that takes in: metric names (list of
      strs), pred_output_filename (str), target_output_filename (str), dataset
      split (str), and tb_summary_dir (str), runs metrics on the outputs in
      output_filename, and returns a dictionary of metrics and their computed
      values. Required if mode is "continuous_eval."
    checkpoints_to_keep: an integer, keep up to this many checkpoints
    save_steps: an integer, save every this many steps
    learning_rate_schedule: an optional function taking the scalar name
      argument `step` and the numeric argument `total_train_steps` and return
      the scalar learning rate
    optimizer: a class extending optimize.Optimizer, required for training
  """
  if not isinstance(batch_size, int):
    batch_size = compute_batch_size(
        sequence_length, mesh_shape, layout_rules, batch_size)

  if not isinstance(train_steps, int):
    train_steps = train_steps(batch_size, sequence_length)

  if callable(learning_rate_schedule):
    learning_rate_schedule = functools.partial(
        learning_rate_schedule, total_train_steps=train_steps)

  tf.logging.info("mode=%s" % mode,)
  tf.logging.info("sequence_length=%s" % sequence_length,)
  tf.logging.info("batch_size=%s" % batch_size,)
  tf.logging.info("train_steps=%s" % train_steps,)
  tf.logging.info("mesh_shape=%s" % mesh_shape,)
  tf.logging.info("layout_rules=%s" % layout_rules,)

  if mode == "train" and dataset_split != "train":
    raise ValueError("mode==\"train\" requires dataset_split==\"train\"")

  mesh_shape = mtf.convert_to_shape(mesh_shape)
  layout_rules = mtf.convert_to_layout_rules(layout_rules)

  cluster = tf.contrib.cluster_resolver.TPUClusterResolver(
      tpu if (tpu) else "", zone=tpu_zone, project=gcp_project)

  tf.logging.info(
      "Building TPUConfig with tpu_job_name={}".format(tpu_job_name)
  )
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
      input_vocab_size=inputs_vocabulary(vocabulary).vocab_size,
      output_vocab_size=targets_vocabulary(vocabulary).vocab_size,
      layout_rules=layout_rules,
      mesh_shape=mesh_shape)

  model_fn = tpu_estimator_model_fn(
      model_type=model_type,
      transformer_model=transformer_model,
      model_dir=model_dir,
      use_tpu=tpu,
      mesh_shape=mesh_shape,
      layout_rules=layout_rules,
      batch_size=batch_size,
      sequence_length=sequence_length,
      autostack=autostack,
      learning_rate_schedule=learning_rate_schedule,
      checkpoints_to_keep=checkpoints_to_keep,
      save_steps=save_steps,
      optimizer=optimizer)

  estimator = tpu_estimator.TPUEstimator(
      model_fn=model_fn,
      config=run_config,
      train_batch_size=batch_size,
      eval_batch_size=batch_size,
      predict_batch_size=batch_size,
      use_tpu=tpu,
      export_to_tpu=False,
      params={})

  if mode == "train":
    if train_dataset_fn is None:
      raise ValueError("Must provide train_dataset_fn through gin for train.")
    def input_fn(params):
      del params
      dataset = train_dataset_fn(batch_size=batch_size,
                                 sequence_length=sequence_length,
                                 vocabulary=vocabulary,
                                 dataset_split=dataset_split)
      return dataset

    estimator.train(input_fn=input_fn, max_steps=train_steps)
  elif mode == "continuous_eval":
    if eval_dataset_fn is None:
      raise ValueError("Must provide eval_dataset_fn through gin for eval.")
    if get_components_fn is None:
      raise ValueError("Must provide get_components_fn through gin for eval.")
    if compute_metrics is None:
      raise ValueError(
          "Must provide compute_metrics through gin for eval.")

    metrics_inputs = get_components_fn()
    for ckpt in tf.contrib.training.checkpoints_iterator(estimator.model_dir):
      for metric_names, component in metrics_inputs:
        if not metric_names:
          tf.logging.info("Skipping %s", component.__dict__)
          continue
        tf.logging.info("Evaluating %s on metrics %s",
                        component.tfds_name, component.metric_names)
        tf.logging.info("on split %s", dataset_split)

        # Regenerate the estimator
        model_fn = tpu_estimator_model_fn(
            model_type=model_type,
            transformer_model=transformer_model,
            model_dir=model_dir,
            use_tpu=tpu,
            mesh_shape=mesh_shape,
            layout_rules=layout_rules,
            batch_size=batch_size,
            sequence_length=sequence_length,
            autostack=autostack,
            checkpoints_to_keep=checkpoints_to_keep,
            save_steps=save_steps)
        estimator = tpu_estimator.TPUEstimator(
            model_fn=model_fn,
            config=run_config,
            train_batch_size=batch_size,
            eval_batch_size=batch_size,
            predict_batch_size=batch_size,
            use_tpu=tpu,
            export_to_tpu=False,
            params={})

        # Extra eval_dataset_fn call to get the dataset_size and an extra
        # dataset object to write out targets. We need to use a separate graph
        # because estimator finalizes the default graph after iterating over the
        # dataset.
        dataset_graph = tf.Graph()
        with dataset_graph.as_default():
          dataset, dataset_size, padded_dataset_size = eval_dataset_fn(
              component,  # pylint: disable=cell-var-from-loop
              batch_size=batch_size, sequence_length=sequence_length,
              vocabulary=vocabulary, dataset_split=dataset_split, pack=False)

        def input_fn(params):
          del params
          dataset, _, _ = eval_dataset_fn(component,  # pylint: disable=cell-var-from-loop
                                          batch_size=batch_size,
                                          sequence_length=sequence_length,
                                          vocabulary=vocabulary,
                                          dataset_split=dataset_split,
                                          pack=False)
          return dataset

        dataset_name = component.tfds_name.replace("/", "-").replace(":", "-")
        output_filename = os.path.join(model_dir, "{}-{}-decoded".format(
            dataset_name, dataset_split))
        pred_output_filename = output_filename + "-preds-test"
        target_output_filename = output_filename + "-targets-test"
        decodes = decode(estimator, input_fn, dataset_size, padded_dataset_size,
                         batch_size, vocabulary,
                         checkpoint_path=checkpoint_path)
        with dataset_graph.as_default():
          log_pred_target(decodes, dataset, dataset_size, vocabulary,
                          pred_output_filename=pred_output_filename,
                          target_output_filename=target_output_filename)
        tf.logging.info("Evaluating metrics: {}".format(
            metric_names))
        tb_summary_dir = os.path.join(model_dir, "{}_eval".format(
            "eval" if dataset_split == "validation" else dataset_split))
        _ = compute_metrics(
            metric_names, pred_output_filename,
            target_output_filename, dataset_split, tb_summary_dir, ckpt)

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
        "unknown mode %s - must be train/continuous_eval/infer" % mode)
