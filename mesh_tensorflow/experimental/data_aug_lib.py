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

"""Data augmentation lib for the Liver Tumor Segmentation (LiTS) dataset.

A set of data augmentation functions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import google_type_annotations
from __future__ import print_function


import tensorflow as tf


def _rand_noise(noise_mean, noise_dev, scale, shape):
  """Generate random noise given a particular scale and shape."""
  noise_shape = [x // scale for x in shape]
  noise_shape = [1 if x == 0 else x for x in noise_shape]
  noise = tf.random.normal(
      shape=noise_shape, mean=noise_mean, stddev=noise_dev)
  noise = tf.clip_by_value(
      noise, noise_mean - 2.0 * noise_dev, noise_mean + 2.0 * noise_dev)

  if scale != 1:
    noise = tf.image.resize_images(
        noise, [shape[0], shape[1]])
    noise = tf.transpose(noise, [0, 2, 1])
    noise = tf.image.resize_images(
        noise, [shape[0], shape[2]])
    noise = tf.transpose(noise, [0, 2, 1])
  return noise


def _gen_rand_mask(ratio_mean, ratio_stddev, scale, shape, smoothness=0):
  """Generate a binary mask."""
  scale = max(scale, 1)

  ratio = tf.random.normal(
      shape=[], mean=ratio_mean, stddev=ratio_stddev)
  low_bound = tf.maximum(0.0, ratio_mean - 2 * ratio_stddev)
  up_bound = tf.minimum(1.0, ratio_mean + 2 * ratio_stddev)
  percentil_q = tf.cast(
      100.0 * tf.clip_by_value(ratio, low_bound, up_bound),
      tf.int32)

  pattern = _rand_noise(0.0, 1.0, scale, shape)
  if smoothness > 0:
    smoothness = int(smoothness) // 2 * 2 + 1
    pattern = tf.expand_dims(tf.expand_dims(pattern, 0), -1)
    pattern = tf.nn.conv3d(
        pattern, filter=tf.ones([smoothness, smoothness, smoothness, 1, 1]),
        strides=[1, 1, 1, 1, 1], padding='SAME', dilations=[1, 1, 1, 1, 1])
    pattern = tf.reduce_sum(pattern, 0)
    pattern = tf.reduce_sum(pattern, -1)

  thres = tf.contrib.distributions.percentile(pattern, q=percentil_q)
  rand_mask = tf.less(pattern, thres)

  return rand_mask


def maybe_gen_fake_data_based_on_real_data(
    image, label, reso, min_fake_lesion_ratio, gen_fake_probability):
  """Remove real lesion and synthesize lesion."""
  # TODO(lehou): Replace magic numbers with flag variables.
  gen_prob_indicator = tf.random_uniform(
      shape=[], minval=0.0, maxval=1.0, dtype=tf.float32)

  background_mask = tf.less(label, 0.5)
  lesion_mask = tf.greater(label, 1.5)
  liver_mask = tf.logical_not(tf.logical_or(background_mask, lesion_mask))

  liver_intensity = tf.boolean_mask(image, liver_mask)
  lesion_intensity = tf.boolean_mask(image, lesion_mask)

  intensity_diff = tf.reduce_mean(liver_intensity) - (
      tf.reduce_mean(lesion_intensity))
  intensity_diff = tf.cond(tf.is_nan(intensity_diff),
                           lambda: 0.0, lambda: intensity_diff)

  lesion_area = tf.reduce_sum(tf.cast(lesion_mask, tf.float32))
  liver_area = tf.reduce_sum(tf.cast(liver_mask, tf.float32))
  lesion_liver_ratio = tf.cond(tf.greater(liver_area, 0.0),
                               lambda: lesion_area / liver_area, lambda: 0.0)
  lesion_liver_ratio += min_fake_lesion_ratio

  fake_lesion_mask = tf.logical_and(
      _gen_rand_mask(ratio_mean=lesion_liver_ratio, ratio_stddev=0.0,
                     scale=reso // 8, shape=label.shape,
                     smoothness=reso // 16),
      tf.logical_not(background_mask))
  liver_mask = tf.logical_not(tf.logical_or(background_mask, fake_lesion_mask))

  # Remove real lesion and add fake lesion.
  # If the intensitify is too small (maybe no liver or lesion region labeled),
  # do not generate fake data.
  gen_prob_indicator = tf.cond(
      tf.greater(intensity_diff, 0.0001),
      lambda: gen_prob_indicator, lambda: 0.0)
  # pylint: disable=g-long-lambda
  image = tf.cond(
      tf.greater(gen_prob_indicator, 1 - gen_fake_probability),
      lambda: image + intensity_diff * tf.cast(lesion_mask, tf.float32) \
                    - intensity_diff * tf.cast(fake_lesion_mask, tf.float32),
      lambda: image)
  label = tf.cond(
      tf.greater(gen_prob_indicator, 1 - gen_fake_probability),
      lambda: tf.cast(background_mask, tf.float32) * 0 + \
          tf.cast(liver_mask, tf.float32) * 1 + \
          tf.cast(fake_lesion_mask, tf.float32) * 2,
      lambda: label)
  # pylint: enable=g-long-lambda

  return image, label


def maybe_flip(image, label, flip_axis, flip_indicator=None):
  """Randomly flip the image."""
  if flip_indicator is None:
    flip_indicator = tf.random_uniform(shape=[])
  flip_or_not = tf.greater(flip_indicator, 0.5)

  def _maybe_flip(data):
    """Flip or not according to flip_or_not."""
    data = tf.cond(tf.logical_and(flip_or_not, tf.equal(flip_axis, 1)),
                   lambda: tf.transpose(data, [1, 0, 2]),
                   lambda: data)
    data = tf.cond(tf.logical_and(flip_or_not, tf.equal(flip_axis, 2)),
                   lambda: tf.transpose(data, [2, 1, 0]),
                   lambda: data)

    data = tf.cond(flip_or_not,
                   lambda: tf.image.flip_up_down(data),
                   lambda: data)

    data = tf.cond(tf.logical_and(flip_or_not, tf.equal(flip_axis, 1)),
                   lambda: tf.transpose(data, [1, 0, 2]),
                   lambda: data)
    data = tf.cond(tf.logical_and(flip_or_not, tf.equal(flip_axis, 2)),
                   lambda: tf.transpose(data, [2, 1, 0]),
                   lambda: data)
    return data

  return _maybe_flip(image), _maybe_flip(label)


def maybe_rot180(image, label, static_axis, rot180_k=None):
  """Randomly rotate the image 180 degrees."""
  if rot180_k is None:
    rot180_k = 2 * tf.random_uniform(
        shape=[], minval=0, maxval=2, dtype=tf.int32)
  rot_or_not = tf.not_equal(rot180_k, 0)

  def _maybe_rot180(data):
    """Rotate or not according to rot_or_not."""
    data = tf.cond(tf.logical_and(rot_or_not, tf.equal(static_axis, 0)),
                   lambda: tf.transpose(data, [2, 1, 0]),
                   lambda: data)
    data = tf.cond(tf.logical_and(rot_or_not, tf.equal(static_axis, 1)),
                   lambda: tf.transpose(data, [0, 2, 1]),
                   lambda: data)

    data = tf.cond(rot_or_not,
                   lambda: tf.image.rot90(data, k=rot180_k),
                   lambda: data)

    data = tf.cond(tf.logical_and(rot_or_not, tf.equal(static_axis, 0)),
                   lambda: tf.transpose(data, [2, 1, 0]),
                   lambda: data)
    data = tf.cond(tf.logical_and(rot_or_not, tf.equal(static_axis, 1)),
                   lambda: tf.transpose(data, [0, 2, 1]),
                   lambda: data)
    return data

  return _maybe_rot180(image), _maybe_rot180(label)
