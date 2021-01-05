# coding=utf-8
# Copyright 2021 The Mesh TensorFlow Authors.
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
"""Tests for mesh_tensorflow.transformer.dataset."""
from absl.testing import absltest
from absl.testing import parameterized
from mesh_tensorflow.transformer import dataset
import numpy as np
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds

tf.disable_v2_behavior()
tf.enable_eager_execution()


class DatasetTest(parameterized.TestCase):

  _PACK_PARAMETERS = ({"use_custom_ops": False},)

  def assert_dataset(self, ds, expected_ds, expected_dtypes):
    actual_ds = list(tfds.as_numpy(ds))
    self.assertLen(actual_ds, len(expected_ds))
    for actual, expected in zip(actual_ds, expected_ds):
      self.assertCountEqual(list(actual.keys()), list(expected.keys()))
      for k, v in actual.items():
        np.testing.assert_array_equal(v, expected[k])
        if k in expected_dtypes:
          self.assertEqual(v.dtype.type, expected_dtypes[k])

  @parameterized.parameters(*_PACK_PARAMETERS)
  def test_pack_dataset(self, use_custom_ops):
    x = [{"inputs": [7, 8, 5, 1], "targets": [3, 9, 1], "idx": [0]},
         {"inputs": [8, 4, 9, 3, 1], "targets": [4, 1], "idx": [1]}]
    ds = create_default_dataset(x, feature_names=("inputs", "targets", "idx"))
    packed_ds = dataset.pack_dataset(
        ds,
        length={"inputs": 10, "targets": 7},
        keys=("inputs", "targets"),
        use_custom_ops=use_custom_ops)

    expected = [{
        "inputs": [7, 8, 5, 1, 8, 4, 9, 3, 1, 0],
        "inputs_segmentation": [1, 1, 1, 1, 2, 2, 2, 2, 2, 0],
        "inputs_position": [0, 1, 2, 3, 0, 1, 2, 3, 4, 0],
        "targets": [3, 9, 1, 4, 1, 0, 0],
        "targets_position": [0, 1, 2, 0, 1, 0, 0],
        "targets_segmentation": [1, 1, 1, 2, 2, 0, 0],
    }]
    self.assert_dataset(
        packed_ds, expected, {"inputs": tf.int32, "targets": tf.int32})

  @parameterized.parameters(*_PACK_PARAMETERS)
  def test_pack_dataset_no_eos(self, use_custom_ops):
    x = [{"inputs": [7, 8, 5], "targets": [3, 9]},
         {"inputs": [8, 4, 9, 3], "targets": [4]}]
    ds = create_default_dataset(x)
    packed_ds = dataset.pack_dataset(
        ds,
        length={"inputs": 8, "targets": 5},
        use_custom_ops=use_custom_ops)

    # Packing still works without the eos.
    expected = [{
        "inputs": [7, 8, 5, 8, 4, 9, 3, 0],
        "inputs_segmentation": [1, 1, 1, 2, 2, 2, 2, 0],
        "inputs_position": [0, 1, 2, 0, 1, 2, 3, 0],
        "targets": [3, 9, 4, 0, 0],
        "targets_position": [0, 1, 0, 0, 0],
        "targets_segmentation": [1, 1, 2, 0, 0],
    }]
    self.assert_dataset(
        packed_ds, expected, {"inputs": tf.int32, "targets": tf.int32})

  @parameterized.parameters(*_PACK_PARAMETERS)
  def test_pack_dataset_long_seq(self, use_custom_ops):
    x = [{"inputs": [7, 8, 5, 6, 9, 4, 1], "targets": [3, 9, 1]},
         {"inputs": [8, 4, 9, 3, 5, 7, 9, 1], "targets": [4, 1]}]
    ds = create_default_dataset(x)
    packed_ds = dataset.pack_dataset(
        ds,
        length={"inputs": 7, "targets": 3},
        use_custom_ops=use_custom_ops)
    expected = [{
        "inputs": [7, 8, 5, 6, 9, 4, 1],
        "inputs_segmentation": [1, 1, 1, 1, 1, 1, 1],
        "inputs_position": [0, 1, 2, 3, 4, 5, 6],
        "targets": [3, 9, 1],
        "targets_position": [0, 1, 2],
        "targets_segmentation": [1, 1, 1],
    }, {
        # EOS is trimmed
        "inputs": [8, 4, 9, 3, 5, 7, 9],
        "inputs_segmentation": [1, 1, 1, 1, 1, 1, 1],
        "inputs_position": [0, 1, 2, 3, 4, 5, 6],
        "targets": [4, 1, 0],
        "targets_position": [0, 1, 0],
        "targets_segmentation": [1, 1, 0],
    }]
    self.assert_dataset(
        packed_ds, expected, {"inputs": tf.int32, "targets": tf.int32})


def create_default_dataset(x, feature_names=("inputs", "targets")):
  output_types = {feature_name: tf.int32 for feature_name in feature_names}
  output_shapes = {feature_name: [None] for feature_name in feature_names}

  ds = tf.data.Dataset.from_generator(
      lambda: x, output_types=output_types, output_shapes=output_shapes)
  return ds

if __name__ == "__main__":
  absltest.main()
