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

"""Tests for third_party.py.mesh_tensorflow.experimental.unet."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mesh_tensorflow.experimental.unet as unet
import tensorflow as tf


class MtfUnetDataAugTest(tf.test.TestCase):

  def constant_3d_image(self):
    return tf.constant(
        [[[-100, 2], [2, 3]], [[4, 35], [-1024, 7]]], dtype=tf.float32)

  def constant_3d_label(self):
    return tf.constant(
        [[[0, 0], [0, 1]], [[1, 1], [2, 2]]], dtype=tf.float32)

  def test_transform(self):
    with tf.Session() as sess:
      image_3d = self.constant_3d_image()
      image_3d_np = sess.run(image_3d)

      for constant_axis in [0, 1, 2]:
        for interpolation in ("BILINEAR", "NEAREST"):
          image_3d_proj = unet._transform_slices(
              image_3d, [1, 0, 0, 0, 1, 0, 0, 0], constant_axis, interpolation)
          image_3d_proj_np = sess.run(image_3d_proj)
          self.assertAllClose(image_3d_proj_np, image_3d_np)

  def test_flip(self):
    with tf.Session() as sess:
      image_3d = self.constant_3d_image()
      image_3d_np = sess.run(image_3d)

      for flip_axis in [0, 1, 2]:
        image_3d_flip = unet._flip_slices(
            image_3d, tf.constant(0.0), flip_axis)
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
        image_3d_flip = unet._flip_slices(
            image_3d_flip, tf.constant(1.0), flip_axis)
        image_3d_flip_np = sess.run(image_3d_flip)
        self.assertAllClose(image_3d_flip_np, image_3d_np)

  def test_add_noise(self):
    with tf.Session() as sess:
      image_3d = self.constant_3d_image()
      image_3d_np = sess.run(image_3d)

      image_3d_noisy = unet._maybe_add_noise(
          image_3d, 1, 2, 1.0, 1e-12)
      image_3d_noisy_np = sess.run(image_3d_noisy)
      self.assertAllClose(image_3d_noisy_np, image_3d_np)

  def test_rot90(self):
    with tf.Session() as sess:
      image_3d = self.constant_3d_image()
      image_3d_np = sess.run(image_3d)

      for constant_axis in [0, 1, 2]:
        image_3d_rot360 = unet._rot90_slices(
            image_3d, 4, constant_axis)
        image_3d_rot360_np = sess.run(image_3d_rot360)
        self.assertAllClose(image_3d_rot360_np, image_3d_np)

  def test_gen_fake_data(self):
    with tf.Session() as sess:
      image_3d = self.constant_3d_image()
      label_3d = self.constant_3d_label()
      image_3d_np = sess.run(image_3d)
      label_3d_np = sess.run(label_3d)

      image_3d_aug, label_3d_aug = unet._maybe_gen_fake_data_based_on_real_data(
          image_3d, label_3d, reso=image_3d.shape[0],
          min_fake_lesion_ratio=0.0,
          gen_prob_indicator=0.0, gen_probability=1.0)

      image_3d_aug_np = sess.run(image_3d_aug)
      label_3d_aug_np = sess.run(label_3d_aug)
      self.assertAllClose(image_3d_aug_np, image_3d_np)
      self.assertAllClose(label_3d_aug_np, label_3d_np)


if __name__ == "__main__":
  tf.test.main()
