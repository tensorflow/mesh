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

import os
import gin

from mesh_tensorflow.transformer import utils
import tensorflow as tf

tf.flags.DEFINE_string(
    "tpu_job_name", None,
    "Name of TPU worker binary. Only necessary if job name is changed from"
    " default tpu_worker.")
tf.flags.DEFINE_string(
    "model_dir", "/tmp/transformer_standalone", "Estimator model_dir")
tf.flags.DEFINE_string(
    "data_dir",
    ""
    ,
    "data_dir for TensorFlow Datasets")

# DATA TYPES (each should be float32 or bfloat16)
# master_dtype must be the same between training and eval/inference
# slice_dtype should be float32 for training (otherwise bad quality)
tf.flags.DEFINE_string("master_dtype", "bfloat16", "datatype for checkpoints")
tf.flags.DEFINE_string(
    "slice_dtype", "", "datatype for variables in memory. "
    "Defaults to float32 during training and on non-TPU. "
    "Defaults to bfloat16 during non-training on TPU. ")
tf.flags.DEFINE_string(
    "activation_dtype", "",
    "datatype for activations.  Defaults to bfloat16 on TPU else float32")

# MISC

tf.flags.DEFINE_string(
    "tpu",
    default=None,
    help="The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.")

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

# GIN PARAMETERS
tf.flags.DEFINE_multi_string("gin_file", None,
                             "List of paths to the config files.")
tf.flags.DEFINE_multi_string(
    "gin_param", None, "Newline separated list of Gin parameter bindings.")

FLAGS = tf.flags.FLAGS

_DEFAULT_CONFIG_FILE = "./transformer_standalone_defaults.gin"


def main(_):
  # Set up the default values for the configurable parameters. These values will
  # be overridden by any user provided gin files/parameters.
  gin.parse_config_file(
      os.path.join(os.path.dirname(__file__), _DEFAULT_CONFIG_FILE))
  gin.parse_config_files_and_bindings(FLAGS.gin_file, FLAGS.gin_param)
  dataset = utils.get_tfds_dataset(data_dir=FLAGS.data_dir)
  utils.run(
      dataset=dataset,
      tpu_job_name=FLAGS.tpu_job_name,
      master_dtype=FLAGS.master_dtype,
      slice_dtype=FLAGS.slice_dtype,
      activation_dtype=FLAGS.activation_dtype,
      tpu=FLAGS.tpu,
      gcp_project=FLAGS.gcp_project,
      tpu_zone=FLAGS.tpu_zone,
      autostack=FLAGS.autostack,
      model_dir=FLAGS.model_dir)


if __name__ == "__main__":
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run()
