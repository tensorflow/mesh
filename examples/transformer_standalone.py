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
from mesh_tensorflow.transformer import dataset as transformer_dataset
from mesh_tensorflow.transformer import model_builder
from mesh_tensorflow.transformer import transformer
from mesh_tensorflow.transformer import utils
import tensorflow as tf
from tensorflow.contrib.tpu.python.tpu import tpu_config
from tensorflow.contrib.tpu.python.tpu import tpu_estimator

tf.flags.DEFINE_string("model_dir", "/tmp/mnist_model", "Estimator model_dir")
tf.flags.DEFINE_string("dataset", "wmt_translate_ende/ende_subwords8k_t2t",
                       "TensorFlow Datasets dataset name")
tf.flags.DEFINE_string(
    "data_dir",
    ""
    ,
    "data_dir for TensorFlow Datasets"
)
tf.flags.DEFINE_boolean(
    "text2self",
    False,
    "Whether to train a language model (True) or encoder-decoder text-to-text "
    "model (False)."
)

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
tf.flags.DEFINE_float("dropout", 0.1, "dropout rate")
tf.flags.DEFINE_float("label_smoothing", 0.1, "label smoothing")
tf.flags.DEFINE_integer(
    "train_steps", 10000000, "Total number of training steps.")

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
tf.flags.DEFINE_float(
    "alpha",
    0.6,
    "length adjustment for beam search (ignored when text2self=True)"
)
tf.flags.DEFINE_integer(
    "beam_size", 1,
    "use a value >1 for beam search (ignored when text2self=True)"
)
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
      num_layers=FLAGS.num_layers,
      d_ff=FLAGS.d_ff,
      num_heads=FLAGS.num_heads,
      d_kv=FLAGS.d_kv,
      dropout_rate=FLAGS.dropout)


def build_model(input_vocab_size, output_vocab_size):
  """Build a simple Transformer model.

  Args:
    input_vocab_size: an integer
    output_vocab_size: an integer

  Returns:
    a mtf.Unitransformer or mtf.Bitransformer
  """
  if FLAGS.text2self:
    return transformer.Unitransformer(
        layer_stack=layer_stack(include_encdec_attention=False),
        d_model=FLAGS.d_model,
        input_vocab_size=input_vocab_size,
        output_vocab_size=output_vocab_size,
        autoregressive=True,
        max_length=FLAGS.max_length,
        shared_embedding_and_softmax_weights=True,
        label_smoothing=FLAGS.label_smoothing,
        layout=FLAGS.layout,
        mesh_shape=FLAGS.mesh_shape)
  else:
    return transformer.Bitransformer(
        encoder_layer_stack=layer_stack(include_encdec_attention=False),
        decoder_layer_stack=layer_stack(include_encdec_attention=True),
        encoder_d_model=FLAGS.d_model,
        decoder_d_model=FLAGS.d_model,
        input_vocab_size=input_vocab_size,
        output_vocab_size=output_vocab_size,
        max_length=FLAGS.max_length,
        shared_embedding=False,
        shared_embedding_and_softmax_weights=True,
        label_smoothing=FLAGS.label_smoothing,
        layout=FLAGS.layout,
        mesh_shape=FLAGS.mesh_shape)


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

  dataset = transformer_dataset.TokenizedTFDSDataset(
      FLAGS.dataset, text2self=FLAGS.text2self, data_dir=FLAGS.data_dir or None)

  output_encoder = dataset.encoders["targets"]
  if FLAGS.text2self:
    input_encoder = output_encoder
  else:
    input_encoder = dataset.encoders["inputs"]

  model = build_model(
      input_vocab_size=transformer_dataset.padded_vocab_size(
          input_encoder.vocab_size, 128),
      output_vocab_size=transformer_dataset.padded_vocab_size(
          output_encoder.vocab_size, 128))
  mesh_shape = mtf.convert_to_shape(FLAGS.mesh_shape)
  layout_rules = mtf.convert_to_layout_rules(FLAGS.layout)
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

  model_fn = utils.tpu_estimator_model_fn(
      model=model,
      model_dir=FLAGS.model_dir,
      use_tpu=FLAGS.tpu,
      mesh_shape=mesh_shape,
      layout_rules=layout_rules,
      text2self=FLAGS.text2self,
      variable_dtype=variable_dtype,
      batch_size=FLAGS.batch_size,
      length=length_from_flags(),
      temperature=FLAGS.temperature,
      beam_size=FLAGS.beam_size,
      alpha=FLAGS.alpha,
      autostack=FLAGS.autostack)

  estimator = tpu_estimator.TPUEstimator(
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.batch_size,
      eval_batch_size=FLAGS.batch_size,
      predict_batch_size=FLAGS.batch_size,
      use_tpu=FLAGS.tpu,
      export_to_tpu=False,
      params={})

  def input_fn(params):
    del params
    return dataset.load(batch_size=FLAGS.batch_size,
                        length=length_from_flags(),
                        train=(FLAGS.mode == "train"),
                        pack=True)

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
    utils.decode_from_file(
        estimator,
        batch_size=FLAGS.batch_size,
        length=length_from_flags(),
        input_filename=FLAGS.input_file,
        output_filename=FLAGS.output_file,
        inputs_encoder=dataset.encoders[
            "targets" if FLAGS.text2self else "inputs"],
        targets_encoder=dataset.encoders["targets"],
        text2self=FLAGS.text2self)
  else:
    raise ValueError(
        "unknown mode %s - must be train/evaluate/infer" % FLAGS.mode)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
