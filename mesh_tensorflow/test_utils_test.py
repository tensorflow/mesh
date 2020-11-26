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

"""Tests for mesh_tensorflow.test_utils."""

import mesh_tensorflow as mtf
from mesh_tensorflow import test_utils
import numpy as np
import tensorflow.compat.v1 as tf
# pylint:disable=g-direct-tensorflow-import
from tensorflow.python.framework import test_util as tf_test_util


class TestUtilsTest(tf.test.TestCase):

  @tf_test_util.run_in_graph_and_eager_modes
  def test_convert_mtf_tensor_to_np_array(self):
    x_np = np.array([[1, 2, 3], [4, 5, 6]])
    converter = test_utils.NumpyConverter()
    shape = mtf.Shape([mtf.Dimension("dim0", 2), mtf.Dimension("dim1", 3)])
    x_mtf = mtf.constant(converter.mesh, x_np, shape=shape, dtype=tf.int32)
    actual = converter.convert_mtf_tensor_to_np_array(x_mtf)
    self.assertAllEqual(x_np, actual)

  @tf_test_util.run_in_graph_and_eager_modes
  def test_convert_mtf_tensor_to_np_array_with_trainable_variable(self):
    converter = test_utils.NumpyConverter()
    shape = mtf.Shape([mtf.Dimension("dim0", 2), mtf.Dimension("dim1", 3)])
    x_mtf = mtf.get_variable(
        converter.mesh,
        name="x",
        shape=shape,
        dtype=tf.float32,
        initializer=tf.zeros_initializer())
    actual = converter.convert_mtf_tensor_to_np_array(x_mtf)
    self.assertAllClose(np.zeros_like(actual), actual)

  def test_convert_mtf_tensor_to_tf_tensor(self):
    x_np = np.array([[1, 2, 3], [4, 5, 6]])
    converter = test_utils.NumpyConverter()
    shape = mtf.Shape([mtf.Dimension("dim0", 2), mtf.Dimension("dim1", 3)])
    x_mtf = mtf.constant(converter.mesh, x_np, shape=shape, dtype=tf.int32)
    _, x_tf = converter.convert_mtf_tensor_to_tf_tensor(x_mtf)
    actual = self.evaluate(x_tf)
    self.assertAllEqual(x_np, actual)


if __name__ == "__main__":
  tf.test.main()
