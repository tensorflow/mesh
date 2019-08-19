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

"""Tests for third_party.py.mesh_tensorflow.experimental.mtf_unet3d."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mesh_tensorflow.experimental.mtf_unet3d as mtf_unet3d
import tensorflow as tf


class MtfUnetDataAugTest(tf.test.TestCase):

  def static_3d_image(self):
    return tf.constant(
        [[[-100, 2], [2, 3]], [[4, 35], [-1024, 7]]], dtype=tf.float32)

  def test_transform(self):
    with tf.Session() as sess:
      image_3d = self.static_3d_image()
      image_3d_np = sess.run(image_3d)

      for static_axis in [0, 1, 2]:
        for interpolation in ("BILINEAR", "NEAREST"):
          image_3d_proj = mtf_unet3d._transform_3d(
              image_3d, [1, 0, 0, 0, 1, 0, 0, 0], static_axis, interpolation)
          image_3d_proj_np = sess.run(image_3d_proj)
          self.assertAllClose(image_3d_proj_np, image_3d_np)

  def test_flip(self):
    with tf.Session() as sess:
      image_3d = self.static_3d_image()
      image_3d_np = sess.run(image_3d)

      for flip_axis in [0, 1, 2]:
        image_3d_flip = mtf_unet3d._flip_3d(
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
        image_3d_flip = mtf_unet3d._flip_3d(
            image_3d_flip, tf.constant(1.0), flip_axis)
        image_3d_flip_np = sess.run(image_3d_flip)
        self.assertAllClose(image_3d_flip_np, image_3d_np)

  def test_add_noise(self):
    with tf.Session() as sess:
      image_3d = self.static_3d_image()
      image_3d_np = sess.run(image_3d)

      image_3d_noisy = mtf_unet3d._maybe_add_noise(
          image_3d, [1, 4], 1.0, 1e-12)
      image_3d_noisy_np = sess.run(image_3d_noisy)
      self.assertAllClose(image_3d_noisy_np, image_3d_np)


if __name__ == "__main__":
  tf.test.main()
