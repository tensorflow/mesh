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

# Lint as: python3
"""Run masked LM/next sentence masked_lm pre-training for BERT."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import time

import mesh_tensorflow as mtf
import mesh_tensorflow.bert.bert as bert_lib
import mesh_tensorflow.bert.optimization as optimization_lib
from six.moves import range
import tensorflow.compat.v1 as tf

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "input_train_files", None,
    "Input TF example files for training (can be a glob or comma separated).")

flags.DEFINE_string(
    "input_eval_files", None,
    "Input TF example files for evaluation (can be a glob or comma separated).")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded. Must match data generation.")

flags.DEFINE_integer(
    "max_predictions_per_seq", 20,
    "Maximum number of masked LM predictions per sequence. "
    "Must match data generation.")

flags.DEFINE_string(
    "mode", "train_and_eval",
    "One of {\"train_and_eval\", \"train\", \"eval\"}.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_bool("clip_gradients", True, "Apply gradient clipping.")

flags.DEFINE_string("optimizer", "adam", "adam/adafactor")

flags.DEFINE_integer("num_train_steps", 100000, "Number of training steps.")

flags.DEFINE_integer("num_warmup_steps", 10000, "Number of warmup steps.")

flags.DEFINE_integer("steps_per_eval", 5000,
                     "How often to evaluate the checkpoint.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer("max_eval_steps", 100, "Maximum number of eval steps.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string("mesh_shape", "batch:8", "mesh shape")
tf.flags.DEFINE_string(
    "layout",
    "batch:batch,vocab:model,intermediate:model,num_heads:model,experts:batch",
    "layout rules")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")


# pylint: disable=unused-argument
def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    # MTF setup.
    graph = mtf.Graph()
    mesh_shape = mtf.convert_to_shape(FLAGS.mesh_shape)
    layout_rules = mtf.convert_to_layout_rules(FLAGS.layout)

    if FLAGS.use_tpu:
      ctx = params["context"]
      num_hosts = ctx.num_hosts
      host_placement_fn = ctx.tpu_host_placement_function
      device_list = [host_placement_fn(host_id=t) for t in range(num_hosts)]
      tf.logging.info("device_list = %s" % device_list,)
      replica_cache_size = 300 * 1000000  # 300M per replica
      # Worker 0 caches all the TPU binaries.
      worker0_mem = replica_cache_size * ctx.num_replicas
      devices_memeory_usage = [worker0_mem] + [0] * (num_hosts - 1)
      var_placer = mtf.utils.BalancedVariablePlacer(device_list,
                                                    devices_memeory_usage)
      mesh_devices = [""] * mesh_shape.size
      physical_shape = list(ctx.device_assignment.topology.mesh_shape)
      logical_to_physical = mtf.simd_mesh_impl.auto_logical_to_physical_tpu(
          mesh_shape.to_integer_list, physical_shape)
      mesh_impl = mtf.simd_mesh_impl.SimdMeshImpl(
          mesh_shape,
          layout_rules,
          mesh_devices,
          ctx.device_assignment,
          logical_to_physical=logical_to_physical)
    else:
      mesh_impl = mtf.placement_mesh_impl.PlacementMeshImpl(
          mesh_shape, layout_rules, [""] * mesh_shape.size)
      var_placer = None

    mesh = mtf.Mesh(graph, "bert_mesh", var_placer)
    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    masked_lm_positions = features["masked_lm_positions"]
    masked_lm_ids = features["masked_lm_ids"]
    masked_lm_weights = features["masked_lm_weights"]
    next_sentence_labels = tf.squeeze(features["next_sentence_labels"], 1)

    batch_size = input_ids.get_shape()[0].value
    batch_dim = mtf.Dimension("batch", batch_size)

    seq_length = input_ids.get_shape()[1].value
    seq_dim = mtf.Dimension("seq", seq_length)
    max_predictions_per_seq = masked_lm_positions.get_shape()[1].value
    max_predictions_per_seq_dim = mtf.Dimension("max_pred_seq",
                                                max_predictions_per_seq)

    mtf_input_ids = mtf.import_tf_tensor(mesh, input_ids, [batch_dim, seq_dim])
    mtf_input_mask = mtf.import_tf_tensor(mesh, input_mask,
                                          [batch_dim, seq_dim])
    mtf_segment_ids = mtf.import_tf_tensor(mesh, segment_ids,
                                           [batch_dim, seq_dim])
    mtf_masked_lm_positions = mtf.import_tf_tensor(
        mesh, masked_lm_positions, [batch_dim, max_predictions_per_seq_dim])
    mtf_masked_lm_ids = mtf.import_tf_tensor(
        mesh, masked_lm_ids, [batch_dim, max_predictions_per_seq_dim])

    mtf_masked_lm_weights = mtf.import_tf_tensor(
        mesh, masked_lm_weights, [batch_dim, max_predictions_per_seq_dim])
    mtf_next_sentence_labels = mtf.import_tf_tensor(
        mesh, next_sentence_labels, [batch_dim])

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    model = bert_lib.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=mtf_input_ids,
        input_mask=mtf_input_mask,
        token_type_ids=mtf_segment_ids,
        layout=layout_rules,
        mesh_shape=mesh_shape)

    (masked_lm_loss, masked_lm_example_loss,
     masked_lm_logits) = model.get_masked_lm_output(
         mtf_masked_lm_positions, mtf_masked_lm_ids, mtf_masked_lm_weights)

    (next_sentence_loss, next_sentence_example_loss,
     next_sentence_logits) = model.get_next_sentence_output(
         mtf_next_sentence_labels)

    extra_loss = model.get_extra_loss()

    total_loss = masked_lm_loss + next_sentence_loss
    total_loss = mtf.anonymize(total_loss)
    masked_lm_example_loss = mtf.anonymize(masked_lm_example_loss)
    masked_lm_logits = mtf.anonymize(masked_lm_logits)
    next_sentence_example_loss = mtf.anonymize(next_sentence_example_loss)
    next_sentence_logits = mtf.anonymize(next_sentence_logits)

    # TRAIN mode
    if mode == tf.estimator.ModeKeys.TRAIN:
      _, update_ops = optimization_lib.create_optimizer(
          total_loss + extra_loss,
          learning_rate,
          num_train_steps,
          num_warmup_steps,
          optimizer=FLAGS.optimizer,
          clip_gradients=FLAGS.clip_gradients)

    lowering = mtf.Lowering(graph, {mesh: mesh_impl})

    tf_loss = tf.to_float(lowering.export_to_tf_tensor(total_loss))

    if mode == tf.estimator.ModeKeys.TRAIN:
      global_step = tf.train.get_global_step()
      tf_update_ops = [lowering.lowered_operation(op) for op in update_ops]
      tf_update_ops.append(tf.assign_add(global_step, 1))
      tf.logging.info("tf_update_ops: {}".format(tf_update_ops))
      train_op = tf.group(tf_update_ops)
    elif mode == tf.estimator.ModeKeys.EVAL:

      def metric_fn(masked_lm_example_loss, masked_lm_logits, masked_lm_ids,
                    masked_lm_weights, next_sentence_example_loss,
                    next_sentence_logits, next_sentence_labels):
        """Computes the loss and accuracy of the model."""
        masked_lm_logits = tf.reshape(masked_lm_logits,
                                      [-1, masked_lm_logits.shape[-1]])
        masked_lm_predictions = tf.argmax(
            masked_lm_logits, axis=-1, output_type=tf.int32)
        masked_lm_example_loss = tf.reshape(masked_lm_example_loss, [-1])
        masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
        masked_lm_weights = tf.reshape(masked_lm_weights, [-1])
        masked_lm_accuracy = tf.metrics.accuracy(
            labels=masked_lm_ids,
            predictions=masked_lm_predictions,
            weights=masked_lm_weights)
        masked_lm_mean_loss = tf.metrics.mean(
            values=masked_lm_example_loss, weights=masked_lm_weights)

        next_sentence_logits = tf.reshape(
            next_sentence_logits, [-1, next_sentence_logits.shape[-1]])
        next_sentence_predictions = tf.argmax(
            next_sentence_logits, axis=-1, output_type=tf.int32)
        next_sentence_labels = tf.reshape(next_sentence_labels, [-1])
        next_sentence_accuracy = tf.metrics.accuracy(
            labels=next_sentence_labels, predictions=next_sentence_predictions)
        next_sentence_mean_loss = tf.metrics.mean(
            values=next_sentence_example_loss)

        return {
            "masked_lm_accuracy": masked_lm_accuracy,
            "masked_lm_loss": masked_lm_mean_loss,
            "next_sentence_accuracy": next_sentence_accuracy,
            "next_sentence_loss": next_sentence_mean_loss,
        }

      eval_metrics = (metric_fn, [
          lowering.export_to_tf_tensor(masked_lm_example_loss),
          lowering.export_to_tf_tensor(masked_lm_logits), masked_lm_ids,
          masked_lm_weights,
          lowering.export_to_tf_tensor(next_sentence_example_loss),
          lowering.export_to_tf_tensor(next_sentence_logits),
          next_sentence_labels
      ])

    with mtf.utils.outside_all_rewrites():
      # Copy master variables to slices. Must be called first.
      restore_hook = mtf.MtfRestoreHook(lowering)
      if mode == tf.estimator.ModeKeys.TRAIN:
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
            FLAGS.output_dir,
            save_steps=1000,
            saver=saver,
            listeners=[saver_listener])

        return tf.estimator.tpu.TPUEstimatorSpec(
            tf.estimator.ModeKeys.TRAIN,
            loss=tf_loss,
            train_op=train_op,
            training_hooks=[restore_hook, saver_hook])
      elif mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.tpu.TPUEstimatorSpec(
            tf.estimator.ModeKeys.EVAL,
            evaluation_hooks=[restore_hook],
            loss=tf_loss,
            eval_metrics=eval_metrics)

  return model_fn


def input_fn_builder(input_files,
                     max_seq_length,
                     max_predictions_per_seq,
                     is_training,
                     num_cpu_threads=4):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    name_to_features = {
        "input_ids":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "input_mask":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "segment_ids":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "masked_lm_positions":
            tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_ids":
            tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_weights":
            tf.FixedLenFeature([max_predictions_per_seq], tf.float32),
        "next_sentence_labels":
            tf.FixedLenFeature([1], tf.int64),
    }

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    if is_training:
      d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
      d = d.repeat()
      d = d.shuffle(buffer_size=len(input_files))

      # `cycle_length` is the number of parallel files that get read.
      cycle_length = min(num_cpu_threads, len(input_files))

      # `sloppy` mode means that the interleaving is not exact. This adds
      # even more randomness to the training pipeline.
      d = d.apply(
          tf.data.experimental.parallel_interleave(
              tf.data.TFRecordDataset,
              sloppy=is_training,
              cycle_length=cycle_length))
      d = d.shuffle(buffer_size=100)
    else:
      d = tf.data.TFRecordDataset(input_files)
      # Since we evaluate for a fixed number of steps we don't want to encounter
      # out-of-range exceptions.
      d = d.repeat()

    # We must `drop_remainder` on training because the TPU requires fixed
    # size dimensions. For eval, we assume we are evaluating on the CPU or GPU
    # and we *don't* want to drop the remainder, otherwise we wont cover
    # every sample.
    d = d.apply(
        tf.data.experimental.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            num_parallel_batches=num_cpu_threads,
            drop_remainder=True))
    return d

  return input_fn


def _decode_record(record, name_to_features):
  """Decodes a record to a TensorFlow example."""
  example = tf.parse_single_example(record, name_to_features)

  # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
  # So cast all int64 to int32.
  for name in list(example.keys()):
    t = example[name]
    if t.dtype == tf.int64:
      t = tf.to_int32(t)
    example[name] = t

  return example


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  bert_config = bert_lib.BertConfig.from_json_file(FLAGS.bert_config_file)

  tf.gfile.MakeDirs(FLAGS.output_dir)

  input_train_files = []
  for input_pattern in FLAGS.input_train_files.split(","):
    input_train_files.extend(tf.gfile.Glob(input_pattern))

  tf.logging.info("*** Input Training Files ***")
  for input_train_file in input_train_files:
    tf.logging.info("  %s" % input_train_file)

  input_eval_files = []
  for input_pattern in FLAGS.input_eval_files.split(","):
    input_eval_files.extend(tf.gfile.Glob(input_pattern))

  tf.logging.info("*** Input Evaluation Files ***")
  for input_eval_file in input_eval_files:
    tf.logging.info("  %s" % input_eval_file)

  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  run_config = tf.estimator.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=tf.estimator.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_cores_per_replica=1,
          per_host_input_for_training=tf.estimator.tpu.InputPipelineConfig
          .BROADCAST))

  model_fn = model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=FLAGS.num_train_steps,
      num_warmup_steps=FLAGS.num_warmup_steps)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.estimator.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size)

  if FLAGS.mode in ("train_and_eval", "train"):
    tf.logging.info("Set train batch size = %d", FLAGS.train_batch_size)
    train_input_fn = input_fn_builder(
        input_files=input_train_files,
        max_seq_length=FLAGS.max_seq_length,
        max_predictions_per_seq=FLAGS.max_predictions_per_seq,
        is_training=True)

  if FLAGS.mode in ("train_and_eval", "eval"):
    tf.logging.info("Set eval batch size = %d", FLAGS.eval_batch_size)
    eval_input_fn = input_fn_builder(
        input_files=input_eval_files,
        max_seq_length=FLAGS.max_seq_length,
        max_predictions_per_seq=FLAGS.max_predictions_per_seq,
        is_training=False)

    try:
      current_step = tf.train.load_variable(FLAGS.output_dir,
                                            tf.GraphKeys.GLOBAL_STEP)
    except (TypeError, ValueError, tf.errors.NotFoundError):
      current_step = 0

    while current_step < FLAGS.num_train_steps:
      if FLAGS.mode == "train_and_eval":
        # Train for up to steps_per_eval number of steps.
        # At the end of training, a checkpoint will be written to --model_dir.
        next_checkpoint = min(current_step + FLAGS.steps_per_eval,
                              FLAGS.num_train_steps)
      elif FLAGS.mode == "train":
        next_checkpoint = FLAGS.num_train_steps

      if FLAGS.mode in ("train_and_eval", "train"):
        start_timestamp = time.time()  # This time will include compilation time
        tf.logging.info("Starting to train.")
        estimator.train(input_fn=train_input_fn, max_steps=next_checkpoint)
        current_step = next_checkpoint

        tf.logging.info("Finished training up to step %d. Elapsed seconds %d.",
                        current_step, int(time.time() - start_timestamp))

      if FLAGS.mode in ("train_and_eval", "eval"):
        tf.logging.info("Starting to evaluate.")
        result = estimator.evaluate(
            input_fn=eval_input_fn, steps=FLAGS.max_eval_steps)
        output_eval_file = os.path.join(
            FLAGS.output_dir, "eval_results_{}.txt".format(current_step))
        with tf.gfile.GFile(output_eval_file, "w") as writer:
          tf.logging.info("***** Eval results *****")
          for key in sorted(result.keys()):
            tf.logging.info("  %s = %s", key, str(result[key]))
            writer.write("%s = %s\n" % (key, str(result[key])))
        if FLAGS.mode == "eval":
          tf.logging.info("Exit eval mode")
          break


if __name__ == "__main__":
  flags.mark_flag_as_required("input_train_files")
  flags.mark_flag_as_required("input_eval_files")
  flags.mark_flag_as_required("bert_config_file")
  flags.mark_flag_as_required("output_dir")
  tf.disable_v2_behavior()
  tf.app.run()
