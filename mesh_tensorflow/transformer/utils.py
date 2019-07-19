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
import re

import gin
import gin.tf

import mesh_tensorflow as mtf
from mesh_tensorflow.transformer import dataset as transformer_dataset
from mesh_tensorflow.transformer import transformer
import numpy as np
import six
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

# List of features used by model.
_INPUT_FEATURES = [
    "inputs", "inputs_position", "inputs_segmentation", "targets",
    "targets_position", "targets_segmentation", "targets_subsegmentation"
]


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

  Currently, four types of models are supported:

  "bitransformer": The traditional encoder-decoder architecture from
     "attention is all you need".  Requires a non-text2self dataset.

  "lm": an autoregressive language model (one layer stack).  This is similar
     to the decoder part of a bitransformer, but with no attention over an
     encoder, since there is no encoder.  Requires a text2self dataset,
     with targets, but no inputs.

  "aligned": a non-autoregressive single-stack model (like BERT).  Requires
     a non-text2self dataset with inputs and targets.  The targets are
     aligned with the inputs.

  "bi_teacher_student": a teacher-student model where both the student and
    teacher are bitransformers. Requires a non-text2self dataset.

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
  elif model_type == "bi_student_teacher":
    return transformer.make_bi_student_teacher(
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
                           keep_checkpoint_max,
                           save_checkpoints_steps,
                           learning_rate_schedule=None,
                           optimizer=None,
                           outer_batch_size=1,
                           tpu_summaries=False,
                           predict_fn=None):
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
    keep_checkpoint_max: an integer
    save_checkpoints_steps: an integer
    learning_rate_schedule: an optional function taking the scalar named
      argument `step` and return the scalar learning rate. Alternatively, a
      constant.
    optimizer: a class extending optimize.Optimizer, required for training
    outer_batch_size: outer batch dimension that could be used to enable the mix
      of data-parallel and model-parallel training of MoE models
    tpu_summaries: a boolean - if True, then use rewrites to make summaries work
      on TPU.  This may be slow, since it uses a host call hack.
    predict_fn: an optional function, see docs for run for more information

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
      x = tf.reshape(
          x, [outer_batch_size, batch_size // outer_batch_size, sequence_length]
      )
      if not use_tpu:
        x = tf.Print(
            x, [x], "import feature %s" % key, summarize=1000, first_n=1)
      mtf_features[key] = mtf.import_fully_replicated(
          mesh, x, feature_shape, name=key)

    if mode == tf.estimator.ModeKeys.PREDICT:
      feature_shape = mtf.Shape([
          mtf.Dimension("batch", batch_size),
          mtf.Dimension("length", sequence_length)
      ])
      mtf_features = {
          k: mtf.reshape(v, feature_shape)
          for k, v in six.iteritems(mtf_features)
      }
      inputs = mtf_features["inputs"]
      if predict_fn:
        mtf_samples = predict_fn(
            model=transformer_model,
            features=mtf_features,
            variable_dtype=get_variable_dtype())
      elif isinstance(transformer_model, transformer.Unitransformer):
        mtf_samples = transformer_model.sample_autoregressive(
            inputs, variable_dtype=get_variable_dtype())
      elif isinstance(transformer_model,
                      (transformer.Bitransformer, transformer.StudentTeacher)):
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
          _, _, length_dim = targets.shape
          inputs = mtf.shift(targets, offset=1, dim=length_dim, wrap=False)
        else:
          inputs = mtf_features["inputs"]

        if isinstance(transformer_model, transformer.Unitransformer):
          position_kwargs = dict(
              sequence_id=mtf_features.get("targets_segmentation", None),
              position=mtf_features.get("targets_position", None),
          )
        elif isinstance(
            transformer_model,
            transformer.Bitransformer) or model_type == "bi_student_teacher":
          position_kwargs = dict(
              encoder_sequence_id=mtf_features.get("inputs_segmentation", None),
              decoder_sequence_id=mtf_features.get("targets_segmentation",
                                                   None),
              decoder_subsequence_id=mtf_features.get("targets_subsegmentation",
                                                      None),
              encoder_position=mtf_features.get("inputs_position", None),
              decoder_position=mtf_features.get("targets_position", None),
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

      if hasattr(transformer_model, "initialize"):
        with mtf.utils.outside_all_rewrites():
          transformer_model.initialize()

      with mtf.utils.outside_all_rewrites():
        # Copy master variables to slices. Must be called first.
        restore_hook = mtf.MtfRestoreHook(lowering)
        saver = tf.train.Saver(
            tf.global_variables(),
            sharded=True,
            max_to_keep=keep_checkpoint_max,
            keep_checkpoint_every_n_hours=2,
            defer_build=False,
            save_relative_paths=True)
        tf.add_to_collection(tf.GraphKeys.SAVERS, saver)
        saver_listener = mtf.MtfCheckpointSaverListener(lowering)
        saver_hook = tf.train.CheckpointSaverHook(
            model_dir,
            save_steps=save_checkpoints_steps,
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
           vocabulary,
           checkpoint_path=None):
  """Decode from an input_fn.

  Args:
    estimator: a TPUEstimator
    input_fn: function that returns a tf.Dataset
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

  return decodes


def write_lines_to_file(lines, filename):
  """Write each line to a filename, replacing the file if it exists.

  Args:
    lines: list of str, lines to write out.
    filename: str, path to filename.
  """
  if tf.io.gfile.exists(filename):
    tf.io.gfile.remove(filename)
  with tf.io.gfile.GFile(filename, "w") as output_file:
    for line in lines:
      output_file.write("{}\n".format(line))


def get_step_from_checkpoint_path(checkpoint_path):
  """Returns the global step for the checkpoint at `checkpoint_path`.

  Assumes `checkpoint_path` corresponds to a file which contains the substring
  model.ckpt-{global_step}

  Args:
    checkpoint_path: str of path to a checkpoint file.

  Returns:
    int of the global step corresponding to the checkpoint file.

  Raises:
    ValueError if checkpoint_path does not correspond to a model checkpoint file
    which contains the global_step in its filename.
  """
  match = re.compile(r".*model\.ckpt\-(\d+).*").match(checkpoint_path)
  if match is None:
    raise ValueError("Invalid checkpoint path {}".format(checkpoint_path))
  return match.group(1)


@gin.configurable
def decode_from_file(estimator,
                     vocabulary,
                     model_type,
                     batch_size,
                     sequence_length,
                     checkpoint_path=None,
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

  checkpoint_step = get_step_from_checkpoint_path(checkpoint_path)
  decodes = decode(
      estimator, input_fn, vocabulary, checkpoint_path=checkpoint_path
  )
  # Remove any padded examples
  dataset_size = len(inputs)
  decodes = decodes[:dataset_size]
  output_filename = "{}-{}".format(output_filename, checkpoint_step)
  write_lines_to_file(decodes, output_filename)


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


def get_checkpoint_iterator(checkpoint_step, model_dir):
  """Get an iterable of checkpoint paths from a provided checkpoint step(s).

  Args:
    checkpoint_step: If checkpoint_step is an int, find the checkpoint with the
      closest global step and return a singleton list. If checkpoint_step is a
      list of ints, replace each int with the path to the checkpoint with the
      closest global step. If checkpoint_step is None, return
      `tf.contrib.training.checkpoints_iterator` for `model_dir`.
    model_dir: str, directory to look for checkpoints in.

  Returns:
    An iterable which yields checkpoint paths.
  """

  def _get_closest_checkpoint(target_checkpoint):
    """Returns checkpoint with closest global step to `target_checkpoint`."""
    checkpoints = set()
    for f in tf.io.gfile.listdir(model_dir):
      try:
        checkpoints.add(int(get_step_from_checkpoint_path(f)))
      except ValueError:
        continue
    if not checkpoints:
      raise ValueError("No checkpoint files found in {}".format(model_dir))
    closest = float("inf")
    for c in checkpoints:
      if abs(target_checkpoint - c) < abs(target_checkpoint - closest):
        closest = c
    if closest != target_checkpoint:
      tf.logging.info(
          "Using checkpoint at step %d which is closest to requested step %d",
          closest,
          target_checkpoint,
      )
    return closest

  def _get_checkpoint_path(step):
    return os.path.join(model_dir, "model.ckpt-{}".format(step))

  if checkpoint_step is None:
    return tf.contrib.training.checkpoints_iterator(model_dir)
  elif isinstance(checkpoint_step, int):
    return [_get_checkpoint_path(_get_closest_checkpoint(checkpoint_step))]
  else:
    closests = np.unique([_get_closest_checkpoint(c) for c in checkpoint_step])
    return [_get_checkpoint_path(closest) for closest in closests]


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
        eval_checkpoint_step=None,
        mode="train",
        iterations_per_loop=100,
        save_checkpoints_steps=1000,
        keep_checkpoint_max=10,
        eval_summary_dir=None,
        batch_size=("tokens_per_replica", 2048),
        train_steps=auto_train_steps,
        sequence_length=gin.REQUIRED,
        mesh_shape=gin.REQUIRED,
        layout_rules=gin.REQUIRED,
        learning_rate_schedule=None,
        optimizer=None,
        predict_fn=None):
  """Run training/eval/inference.

  Args:
    tpu_job_name: string, name of TPU worker binary
    tpu: string, the Cloud TPU to use for training
    gcp_project: string, project name for the Cloud TPU-enabled project
    tpu_zone: string, GCE zone where the Cloud TPU is located in
    model_dir: string, estimator model_dir
    model_type: a string - either "bitransformer", "bi_student_teacher", lm" or
      "aligned"
    vocabulary: a vocabulary.Vocabulary or (inputs_vocabulary,
      targets_vocabulary) tuple.
    train_dataset_fn: A function returning a tf.data.Dataset. Must be provided
      for mode="train". Should accept the following arguments:
        - batch_size: int, number of entries in each batch.
        - sequence_length: int, length of each packed or padded sequence.
        - vocabulary: Vocabulary instance to use for encoding.
        - dataset_split: str, which dataset split to load.
    eval_dataset_fn: A function returning a list of dataset.EvalDataset tuples.
      Must be provided for mode="eval". Should accept the following arguments:
        - batch_size: int, number of entries in each batch.
        - sequence_length: int, length of each packed or padded sequence.
        - vocabulary: Vocabulary instance to use for encoding.
        - dataset_split: str, which dataset split to load.
      dataset.EvalDataset tuples are namedtuples with the following fields:
        - name: string, the task name
        - dataset_fn: function which returns a tf.data.Dataset of tokenized and
          padded examples. Must not require any arguments and must include the
          feature keys 'inputs' and 'targets_plaintext'.
        - postprocess_fn: function which converts plaintext targets to values
          that can be processed by a `metric_fn`.
        - list_of_metric_fns: list of metric functions with the call signature
          `metric_fn(targets, predictions)` which returns a dict mapping
          submetric names to scalar values. TensorBoard summaries and other tags
          will be written out using the submetric names.
        - dataset_size: number of entries in the dataset.
    dataset_split: a string
    autostack: boolean, internally combine variables
    eval_checkpoint_step: int, list of ints, or None. Only used when mode="eval"
      or mode="infer". If an int or list of ints, evaluation or inference will
      be run on the checkpoint files in `model_dir` whose global steps are
      closest to the global steps provided. If None and mode="eval", run eval
      continuously waiting for new checkpoints via
      `tf.contrib.training.checkpoints_iterator`.
    mode: string, train/eval/infer
    iterations_per_loop: integer, steps per train loop
    save_checkpoints_steps: integer, steps per checkpoint
    keep_checkpoint_max: an integer, keep up to this many checkpoints
    eval_summary_dir: str, path to write TensorBoard events file summaries for
      eval. If None, use model_dir/eval_{split}.
    batch_size: An integer or a (method, value) pair to pass to
      compute_batch_size(). Note that this is the global batch size and not the
      per-shard batch size.
    train_steps: An integer or a function with the same signature as
      auto_train_steps().  Total number of training steps.
    sequence_length: an integer
    mesh_shape: an input to mtf.convert_to_shape()
    layout_rules: an input to mtf.convert_to_layout_rules()
    learning_rate_schedule: an optional function taking the scalar name argument
      `step` and the numeric argument `total_train_steps` and return the scalar
      learning rate
    optimizer: a class extending optimize.Optimizer, required for training
    predict_fn: an optional function that can be used to override the default
      transformer prediction behavior. Must return a tensor of shape [batch_dim,
      length_dim] that will be the prediction for each example. Must accept the
      following arguments:
        - model: a Unitransformer or Bitransformer
        - features: a dict representing an example. Every value will be an
          mtf.Tensor with shape [batch_dim, length_dim].
        - variable_dtype: an mtf.VariableDType
  """
  if not isinstance(batch_size, int):
    batch_size = compute_batch_size(
        sequence_length, mesh_shape, layout_rules, batch_size)

  if not isinstance(train_steps, int):
    train_steps = train_steps(batch_size, sequence_length)

  if callable(learning_rate_schedule):
    learning_rate_schedule = functools.partial(
        learning_rate_schedule, total_train_steps=train_steps)

  tf.logging.info("model_type=%s" % model_type,)
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
      tpu_config=my_tpu_config,
      # We use a saver hook, so disable checkpoints here to prevent double
      # saving.
      save_checkpoints_steps=None,
      save_checkpoints_secs=None)

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
      keep_checkpoint_max=keep_checkpoint_max,
      save_checkpoints_steps=save_checkpoints_steps,
      optimizer=optimizer,
      predict_fn=predict_fn)

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

  elif mode == "eval":
    if eval_dataset_fn is None:
      raise ValueError("Must provide eval_dataset_fn through gin for eval.")

    eval_datasets = eval_dataset_fn(
        batch_size=batch_size,
        sequence_length=sequence_length,
        vocabulary=vocabulary,
        dataset_split=dataset_split,
    )

    valid_eval_datasets = []
    for eval_dataset in eval_datasets:
      if not eval_dataset.metric_fns:
        tf.logging.info(
            "Skipping %s because metric_fns is empty", eval_dataset.name
        )
        continue
      # Convert to EvalDataset tuple in case eval_dataset_fn returns raw tuples
      valid_eval_datasets.append(transformer_dataset.EvalDataset(*eval_dataset))
    eval_datasets = valid_eval_datasets

    if not eval_datasets:
      raise ValueError(
          "All provided EvalDatasets have metric_fns=[]; eval is not possible."
      )

    eval_summary_dir = eval_summary_dir or os.path.join(
        model_dir, "{}_eval".format(dataset_split)
    )
    summary_writer = tf.summary.FileWriter(eval_summary_dir)

    # Pre-load in all of the targets once before entering continuous eval loop
    cached_targets = {}
    # Need to create a separate graph for loading in plaintext targets
    # or else TF will complain that we modified the graph
    with tf.Graph().as_default():
      for eval_dataset in eval_datasets:
        if eval_dataset.metric_fns:
          ds = eval_dataset.dataset_fn()
          # De-batch the dataset so that we iterate over examples, not batches
          ds = ds.flat_map(tf.data.Dataset.from_tensor_slices)
          # Strip off padded examples.
          ds = ds.take(eval_dataset.dataset_size)
          # Create list of postprocessed text targets
          ds = tfds.as_numpy(ds)
          targets = [
              eval_dataset.postprocess_fn(d["targets_plaintext"], example=d)
              for d in ds]
          targets_filename = os.path.join(
              eval_summary_dir, "{}_targets".format(eval_dataset.name),
          )
          write_lines_to_file(targets, targets_filename)
          cached_targets[eval_dataset.name] = targets

    def input_fn(params):
      """Eval input function for estimator."""
      del params
      # Concatenate all dataset inputs to only have to do one decode loop
      combined_ds = None
      for eval_dataset in eval_datasets:
        # Only cache targets for those tasks with eval functions provides
        if eval_dataset.metric_fns:
          ds = eval_dataset.dataset_fn()
          # Only pass those variables which will be used for decoding
          ds = ds.map(
              lambda x: {k: v for k, v in x.items() if k in _INPUT_FEATURES}
          )
          combined_ds = ds if not combined_ds else combined_ds.concatenate(ds)
      return combined_ds

    checkpoint_paths = get_checkpoint_iterator(eval_checkpoint_step, model_dir)
    for checkpoint_path in checkpoint_paths:
      decodes = decode(estimator, input_fn, vocabulary, checkpoint_path)
      # Keep track of where in the predictions list each EvalDataset starts
      dataset_start = 0
      for eval_dataset in eval_datasets:
        # Extract the portion of decodes corresponding to this dataset
        dataset_end = dataset_start + eval_dataset.dataset_size
        dataset_decodes = decodes[dataset_start:dataset_end]
        predictions = [
            eval_dataset.postprocess_fn(d, example=None)
            for d in dataset_decodes]
        # Set the start location for the next dataset to be the number of
        # batches in this dataset
        dataset_batches = int(np.ceil(eval_dataset.dataset_size/batch_size))
        entries_in_padded_dataset = dataset_batches*batch_size
        padded_dataset_end = dataset_start + entries_in_padded_dataset
        dataset_start = padded_dataset_end

        global_step = int(get_step_from_checkpoint_path(checkpoint_path))

        predictions_filename = os.path.join(
            eval_summary_dir,
            "{}_{}_predictions".format(eval_dataset.name, global_step),
        )
        write_lines_to_file(predictions, predictions_filename)

        for metric_fn in eval_dataset.metric_fns:
          summary = tf.Summary()
          targets = cached_targets[eval_dataset.name]
          metric_result = metric_fn(targets, predictions)
          for metric_name, metric_value in metric_result.items():
            tag = "eval/{}/{}".format(eval_dataset.name, metric_name)
            tf.logging.info(
                "%s at step %d: %.3f", tag, global_step, metric_value
            )
            summary.value.add(tag=tag, simple_value=metric_value)
            summary_writer.add_summary(summary, global_step)
        summary_writer.flush()

      assert dataset_start == len(decodes)

  elif mode == "infer":
    checkpoint_paths = get_checkpoint_iterator(eval_checkpoint_step, model_dir)
    for checkpoint_path in checkpoint_paths:
      decode_from_file(
          estimator,
          vocabulary=vocabulary,
          model_type=model_type,
          batch_size=batch_size,
          sequence_length=sequence_length,
          checkpoint_path=checkpoint_path)
  else:
    raise ValueError(
        "unknown mode %s - must be train/eval/infer" % mode)
