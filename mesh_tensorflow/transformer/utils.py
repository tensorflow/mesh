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

import mesh_tensorflow as mtf
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tpu.python.tpu import tpu_estimator


def tpu_estimator_model_fn(model,
                           model_dir,
                           use_tpu,
                           mesh_shape,
                           layout_rules,
                           text2self,
                           variable_dtype,
                           batch_size,
                           length,
                           temperature=0.0,
                           beam_size=1,
                           alpha=0.0,
                           autostack=True):
  """Create a TPUEstimator model function.

  Args:
    model: a transformer.Unitransformer or transformer.Bitransformer
    model_dir: a string
    use_tpu: a boolean
    mesh_shape: a mtf.Shape
    layout_rules: a mtf.LayoutRules
    text2self: a boolean
    variable_dtype: a mtf.VariableDType
    batch_size: an integer
    length: an integer
    temperature: a float between 0 and 1 (for inference)
    beam_size: a positive integer (for inference)
    alpha: a float (for inference)
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

    def _import_feature(key):
      """Import a feature from the features dictionary into a mtf.Tensor.

      Args:
        key: a string

      Returns:
        a mtf.Tensor with dtype int32 and shape [batch_dim, length_dim]
      """
      batch_dim = mtf.Dimension("batch", batch_size)
      length_dim = mtf.Dimension("length", length)
      mtf_shape = mtf.Shape([batch_dim, length_dim])
      if key not in features:
        return None
      x = tf.to_int32(features[key])
      if not use_tpu:
        x = tf.Print(x, [x], "import feature %s" % key, summarize=1000,
                     first_n=1)
      return mtf.import_fully_replicated(mesh, x, mtf_shape, name=key)

    # PREDICT mode
    if mode == tf.estimator.ModeKeys.PREDICT:
      inputs = _import_feature("inputs")
      if text2self:
        mtf_samples = model.sample_autoregressive(
            inputs,
            variable_dtype=variable_dtype,
            temperature=temperature)
      else:
        mtf_samples = model.decode(
            inputs,
            variable_dtype=variable_dtype,
            beam_size=beam_size,
            alpha=alpha,
            temperature=temperature)
      mtf_samples = mtf.anonymize(mtf_samples)
      lowering = mtf.Lowering(
          graph, {mesh: mesh_impl}, autostack=autostack)
      outputs = lowering.export_to_tf_tensor(mtf_samples)
      predictions = {
          "outputs": outputs
      }
      return tpu_estimator.TPUEstimatorSpec(
          mode=tf.estimator.ModeKeys.PREDICT,
          predictions=predictions,
          prediction_hooks=[mtf.MtfRestoreHook(lowering)])

    targets = _import_feature("targets")
    anon_targets = mtf.anonymize(targets)
    if text2self:
      _, length_dim = targets.shape
      inputs = mtf.shift(targets, offset=1, dim=length_dim, wrap=False)
    else:
      inputs = _import_feature("inputs")

    if text2self:
      position_kwargs = dict(
          sequence_id=_import_feature("targets_segmentation"),
          position=_import_feature("targets_position"),
      )
    else:
      position_kwargs = dict(
          encoder_sequence_id=_import_feature("inputs_segmentation"),
          decoder_sequence_id=_import_feature("targets_segmentation"),
          encoder_position=_import_feature("inputs_position"),
          decoder_position=_import_feature("targets_position"),
      )

    logits, loss = model.call_simple(
        inputs=inputs,
        targets=targets,
        compute_loss=True,
        mode=mode,
        variable_dtype=variable_dtype,
        **position_kwargs
    )

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
          model_dir,
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
  return my_model_fn


def decode_from_file(estimator,
                     batch_size,
                     length,
                     input_filename,
                     output_filename,
                     inputs_encoder,
                     targets_encoder,
                     text2self):
  """Decode from a text file.

  Args:
    estimator: a TPUEstimator
    batch_size: an integer
    length: an integer (maximum decode length)
    input_filename: a string
    output_filename: a string
    inputs_encoder: a mtf.transformer.dataset.Encoder
    targets_encoder: a mtf.transformer.dataset.Encoder
    text2self: a boolean
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
    ids = inputs_encoder.encode(line.strip())
    if not text2self:
      # for text2self problems, the inputs represent a partial sequence
      # to be continued, and should not be terminated by EOS.
      # for sequence-to-sequence problems, the input needs to be EOS-terminated
      ids += [1]
    if len(ids) > length:
      ids = ids[:length]
    else:
      ids.extend([0] * (length - len(ids)))
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
  result_iter = estimator.predict(input_fn)
  vocab_size = targets_encoder.vocab_size
  decodes = []
  for i, result in enumerate(result_iter):
    output_ids = clean_decodes(list(result["outputs"]), vocab_size)
    output_string = targets_encoder.decode(output_ids)
    decodes.append(output_string)
    if i < 3:
      # LOG THE FIRST FEW DECODES
      tf.logging.info(inputs[i])
      tf.logging.info(output_string)
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
      ret.append(i)
  return ret
