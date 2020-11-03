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

"""Tests for mesh_tensorflow.transformer.transformer_layers."""

import collections
from mesh_tensorflow import test_utils
from mesh_tensorflow.transformer import transformer_layers
import numpy as np
import tensorflow as tf


class TransformerLayersTest(tf.test.TestCase):

  def test_conv1d_call_same_input_output_dims(self):
    converter = test_utils.NumpyConverter()
    batch = 2
    d_model = 6
    length = 3
    inputs = np.random.randint(0, 10, size=[batch, length])
    inputs_mtf = converter.convert_np_array_to_mtf_tensor(
        inputs, dim_names=["batch", "length"])
    # Dummy context with necessary information for Conv1DLayer.call
    Context = collections.namedtuple("Context",
                                     ["inputs", "activation_dtype", "mode"])
    context = Context(
        inputs=inputs_mtf, activation_dtype=tf.float32, mode="train")
    x = np.random.randn(batch, length, d_model)
    x_mtf = converter.convert_np_array_to_mtf_tensor(
        x, dtype=tf.float32, dim_names=["batch", "length", "d_model"])
    conv_layer = transformer_layers.Conv1DLayer(
        filter_size=3, output_size=d_model)
    output_mtf = conv_layer.call(context, x_mtf)
    self.assertAllEqual([batch, length, d_model],
                        output_mtf.shape.to_integer_list)


if __name__ == "__main__":
  tf.test.main()
