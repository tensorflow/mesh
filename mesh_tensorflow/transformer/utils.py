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
from mesh_tensorflow.transformer import dataset as transformer_dataset
from mesh_tensorflow.transformer import model_builder
from mesh_tensorflow.transformer import transformer
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tpu.python.tpu import tpu_config
from tensorflow.contrib.tpu.python.tpu import tpu_estimator


@gin.configurable
def tpu_estimator_model_fn(transformer_model,
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
    transformer_model: a transformer.Unitransformer or transformer.Bitransformer
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
        x = tf.Print(
            x, [x], "import feature %s" % key, summarize=1000, first_n=1)
      return mtf.import_fully_replicated(mesh, x, mtf_shape, name=key)

    # PREDICT mode
    if mode == tf.estimator.ModeKeys.PREDICT:
      inputs = _import_feature("inputs")
      if text2self:
        mtf_samples = transformer_model.sample_autoregressive(
            inputs, variable_dtype=variable_dtype, temperature=temperature)
      else:
        mtf_samples = transformer_model.decode(
            inputs,
            variable_dtype=variable_dtype,
            beam_size=beam_size,
            alpha=alpha,
            temperature=temperature)
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

    logits, loss = transformer_model.call_simple(
        inputs=inputs,
        targets=targets,
        compute_loss=True,
        mode=mode,
        variable_dtype=variable_dtype,
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
                     batch_size,
                     length,
                     inputs_encoder,
                     targets_encoder,
                     text2self,
                     input_filename=gin.REQUIRED,
                     output_filename=gin.REQUIRED):
  """Decode from a text file.

  Args:
    estimator: a TPUEstimator
    batch_size: an integer
    length: an integer (maximum decode length)
    inputs_encoder: a mtf.transformer.dataset.Encoder
    targets_encoder: a mtf.transformer.dataset.Encoder
    text2self: a boolean
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


@gin.configurable()
def model(input_vocab_size,
          output_vocab_size,
          text2self,
          num_layers=gin.REQUIRED,
          d_ff=gin.REQUIRED,
          d_kv=gin.REQUIRED,
          d_model=gin.REQUIRED,
          num_heads=gin.REQUIRED,
          dropout=gin.REQUIRED,
          max_length=gin.REQUIRED,
          length=gin.REQUIRED,
          label_smoothing=gin.REQUIRED,
          layout=gin.REQUIRED,
          mesh_shape=gin.REQUIRED):
  """Build a simple Transformer model.

  Args:
    input_vocab_size: an integer
    output_vocab_size: an integer
    text2self: a boolean meaning a language model (True) or encoder/decoder
      (False)
    num_layers: integer, number of transformer layers
    d_ff: integer, size of feed-forward hidden layers
    d_kv: integer, size of attention keys/values
    d_model: integer, size of hidden state
    num_heads: integer, heads per attention layer
    dropout: float, dropout rate
    max_length: maximum sequence length (checkpoints depend on this)
    length: actual sequence length - defaults to max_length
    label_smoothing: label smoothing
    layout: a string
    mesh_shape: a string

  Returns:
    a mtf.Unitransformer or mtf.Bitransformer
  """
  # Needed for Gin injection.
  del length

  def layer_stack(include_encdec_attention):
    """Create a LayerStack.

    TODO(noam): implement a way to configure custom layer stacks using
    hyperparameters. (as in mtf_transformer2 in the tensor2tensor library).
    That functionality should go in transformer/model_builder.py

    Args:
      include_encdec_attention: a boolean

    Returns:
      a transformer.LayerStack
    """
    return model_builder.simple_layer_stack(
        include_encdec_attention=include_encdec_attention,
        num_layers=num_layers,
        d_ff=d_ff,
        d_kv=d_kv,
        num_heads=num_heads,
        dropout_rate=dropout)

  if text2self:
    return transformer.Unitransformer(
        layer_stack=layer_stack(include_encdec_attention=False),
        d_model=d_model,
        input_vocab_size=input_vocab_size,
        output_vocab_size=output_vocab_size,
        autoregressive=True,
        max_length=max_length,
        shared_embedding_and_softmax_weights=True,
        label_smoothing=label_smoothing,
        layout=layout,
        mesh_shape=mesh_shape)
  else:
    return transformer.Bitransformer(
        encoder_layer_stack=layer_stack(include_encdec_attention=False),
        decoder_layer_stack=layer_stack(include_encdec_attention=True),
        encoder_d_model=d_model,
        decoder_d_model=d_model,
        input_vocab_size=input_vocab_size,
        output_vocab_size=output_vocab_size,
        max_length=max_length,
        shared_embedding=False,
        shared_embedding_and_softmax_weights=True,
        label_smoothing=label_smoothing,
        layout=layout,
        mesh_shape=mesh_shape)


@gin.configurable
def get_tfds_dataset(dataset_name, data_dir, text2self=gin.REQUIRED):
  """Loads the TFDS dataset specified by datatset_name.

  Args:
    dataset_name: TensorFlow Datasets dataset name.
    data_dir: string, data_dir for TensorFlow Datasets
    text2self: Whether to train a language model (True) or encoder-decoder
      text-to-text model (False).

  Returns:
    A transformer_dataset.Dataset.
  """
  text2self = gin.query_parameter("run.text2self")
  return transformer_dataset.TokenizedTFDSDataset(
      dataset_name, text2self=text2self, data_dir=data_dir or None)


@gin.configurable(blacklist=["dataset"])
def run(tpu_job_name,
        master_dtype,
        slice_dtype,
        activation_dtype,
        tpu,
        gcp_project,
        tpu_zone,
        autostack,
        model_dir,
        dataset,
        mode=gin.REQUIRED,
        iterations_per_loop=gin.REQUIRED,
        save_checkpoints_steps=gin.REQUIRED,
        eval_steps=gin.REQUIRED,
        train_steps=gin.REQUIRED,
        batch_size=gin.REQUIRED,
        text2self=gin.REQUIRED):
  """Run training/eval/inference.

  Args:
    tpu_job_name: string, name of TPU worker binary
    master_dtype: string, datatype for checkpoints
    slice_dtype: string, datatype for variables in memory
    activation_dtype: string, datatype for activations
    tpu: string, the Cloud TPU to use for training
    gcp_project: string, project name for the Cloud TPU-enabled project
    tpu_zone: string, GCE zone where the Cloud TPU is located in
    autostack: boolean, internally combine variables
    model_dir: string, estimator model_dir
    dataset: A transformer_dataset.Dataset to read data from.
    mode: string, train/evaluate/infer
    iterations_per_loop: integer, steps per train loop
    save_checkpoints_steps: integer, steps per checkpoint
    eval_steps: integer, number of evaluation steps
    train_steps: Total number of training steps.
    batch_size: Mini-batch size for the training. Note that this is the global
      batch size and not the per-shard batch.
    text2self: Whether to train a language model (True) or encoder-decoder
      text-to-text model (False).
  """
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

  output_encoder = dataset.encoders["targets"]
  if text2self:
    input_encoder = output_encoder
  else:
    input_encoder = dataset.encoders["inputs"]

  transformer_model = model(
      input_vocab_size=transformer_dataset.padded_vocab_size(
          input_encoder.vocab_size, 128),
      output_vocab_size=transformer_dataset.padded_vocab_size(
          output_encoder.vocab_size, 128),
      text2self=text2self)
  mesh_shape = mtf.convert_to_shape(gin.query_parameter("model.mesh_shape"))
  layout_rules = mtf.convert_to_layout_rules(
      gin.query_parameter("model.layout"))
  # Data-types used for variables and activations
  # See comments in the FLAGS
  master_dtype = tf.as_dtype(master_dtype)
  if slice_dtype:
    slice_dtype = tf.as_dtype(slice_dtype)
  elif not tpu or mode == "train":
    slice_dtype = tf.float32
  else:
    slice_dtype = tf.bfloat16
  if activation_dtype:
    activation_dtype = tf.as_dtype(activation_dtype)
  else:
    activation_dtype = tf.bfloat16 if tpu else tf.float32
  variable_dtype = mtf.VariableDType(
      master_dtype=master_dtype,
      slice_dtype=slice_dtype,
      activation_dtype=activation_dtype)

  length_from_config = gin.query_parameter(
      "model.length") or gin.query_parameter("model.max_length")

  model_fn = tpu_estimator_model_fn(
      transformer_model=transformer_model,
      model_dir=model_dir,
      use_tpu=tpu,
      mesh_shape=mesh_shape,
      layout_rules=layout_rules,
      text2self=text2self,
      variable_dtype=variable_dtype,
      batch_size=batch_size,
      length=length_from_config,
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
    return dataset.load(
        batch_size=batch_size,
        length=length_from_config,
        train=(mode == "train"),
        pack=True)

  if mode == "train":
    estimator.train(input_fn=input_fn, max_steps=train_steps)
  elif mode == "evaluate":
    estimator.evaluate(
        input_fn=input_fn,
        steps=eval_steps,
    )
  elif mode == "infer":
    decode_from_file(
        estimator,
        batch_size=batch_size,
        length=length_from_config,
        inputs_encoder=dataset.encoders["targets" if text2self else "inputs"],
        targets_encoder=dataset.encoders["targets"],
        text2self=text2self)
  else:
    raise ValueError("unknown mode %s - must be train/evaluate/infer" % mode)
