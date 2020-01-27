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

"""Tests for third_party.py.mesh_tensorflow.experimental.data_aug_lib."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mesh_tensorflow.experimental.data_aug_lib as data_aug_lib
import tensorflow.compat.v1 as tf


class MtfUnetDataAugTest(tf.test.TestCase):

  def constant_3d_image(self):
    return tf.constant(
        [[[-100, 2], [2, 3]], [[4, 35], [-1024, 7]]], dtype=tf.float32)

  def constant_3d_label(self):
    return tf.constant(
        [[[0, 0], [0, 1]], [[1, 1], [2, 2]]], dtype=tf.float32)

  def test_flip(self):
    with tf.Session() as sess:
      image_3d = self.constant_3d_image()
      image_3d_np = sess.run(image_3d)

      for flip_axis in [0, 1, 2]:
        image_3d_flip, _ = data_aug_lib.maybe_flip(
            image_3d, tf.zeros_like(image_3d), flip_axis, 0.0)
        image_3d_flip_np = sess.run(image_3d_flip)
        self.assertAllClose(image_3d_flip_np, image_3d_np)

      image_3d_flip = image_3d
      for flip_axis in [0, 1, 2]:
        if flip_axis == 0:
          image_3d_np = image_3d_np[::-1, ...]
        elif flip_axis == 1:
          image_3d_np = image_3d_np[:, ::-1, :]
        else:
          image_3d_np = image_3d_np[..., ::-1]
        image_3d_flip, _ = data_aug_lib.maybe_flip(
            image_3d_flip, tf.zeros_like(image_3d_flip), flip_axis, 1.0)
        image_3d_flip_np = sess.run(image_3d_flip)
        self.assertAllClose(image_3d_flip_np, image_3d_np)

  def test_rot180(self):
    with tf.Session() as sess:
      image_3d = self.constant_3d_image()
      image_3d_np = sess.run(image_3d)

      for constant_axis in [0, 1, 2]:
        image_3d_rot360, _ = data_aug_lib.maybe_rot180(
            image_3d, tf.zeros_like(image_3d), constant_axis, 2)
        image_3d_rot360, _ = data_aug_lib.maybe_rot180(
            image_3d_rot360, tf.zeros_like(image_3d_rot360), constant_axis, 2)
        image_3d_rot360_np = sess.run(image_3d_rot360)
        self.assertAllClose(image_3d_rot360_np, image_3d_np)

  def test_gen_fake_data(self):
    with tf.Session() as sess:
      image_3d = self.constant_3d_image()
      label_3d = self.constant_3d_label()
      image_3d_np = sess.run(image_3d)
      label_3d_np = sess.run(label_3d)

      image_3d_aug, label_3d_aug = \
          data_aug_lib.maybe_gen_fake_data_based_on_real_data(
              image_3d, label_3d, reso=2,
              min_fake_lesion_ratio=0.0, gen_fake_probability=0.0)

      image_3d_aug_np = sess.run(image_3d_aug)
      label_3d_aug_np = sess.run(label_3d_aug)
      self.assertAllClose(image_3d_aug_np, image_3d_np)
      self.assertAllClose(label_3d_aug_np, label_3d_np)


if __name__ == "__main__":
  tf.test.main()
