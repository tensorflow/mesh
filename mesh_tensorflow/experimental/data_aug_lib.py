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

"""Data augmentation lib for the Liver Tumor Segmentation (LiTS) dataset.

A set of data augmentation functions.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import tensorflow.compat.v1 as tf
import tensorflow_probability as tfp
from tensorflow.contrib import image as contrib_image


def _truncated_normal(mean, stddev):
  v = tf.random.normal(shape=[], mean=mean, stddev=stddev)
  v = tf.clip_by_value(v, -2 * stddev + mean, 2 * stddev + mean)
  return v


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


def projective_transform(
    image, label, reso, image_translate_ratio, image_transform_ratio,
    sampled_2d_slices=False):
  """Apply projective transformation on image and label."""

  if image_translate_ratio < 0.000001 and (
      image_transform_ratio < 0.000001):
    return image, label

  def _projective_transform(data, proj_matrix, static_axis, interpolation):
    """Apply projective transformation."""
    if static_axis == 2:
      data = contrib_image.transform(data, proj_matrix, interpolation)
    elif static_axis == 1:
      data = tf.transpose(data, [0, 2, 1])
      data = contrib_image.transform(data, proj_matrix, interpolation)
      data = tf.transpose(data, [0, 2, 1])
    else:
      data = tf.transpose(data, [2, 1, 0])
      data = contrib_image.transform(data, proj_matrix, interpolation)
      data = tf.transpose(data, [2, 1, 0])
    return data

  for static_axis in [0, 1, 2]:
    if sampled_2d_slices and static_axis != 2:
      continue
    a0 = _truncated_normal(1.0, image_transform_ratio)
    a1 = _truncated_normal(0.0, image_transform_ratio)
    a2 = _truncated_normal(
        0.0, image_translate_ratio * reso)
    b0 = _truncated_normal(0.0, image_transform_ratio)
    b1 = _truncated_normal(1.0, image_transform_ratio)
    b2 = _truncated_normal(
        0.0, image_translate_ratio * reso)
    c0 = _truncated_normal(0.0, image_transform_ratio)
    c1 = _truncated_normal(0.0, image_transform_ratio)
    proj_matrix = [a0, a1, a2, b0, b1, b2, c0, c1]

    image = _projective_transform(image, proj_matrix, static_axis, 'BILINEAR')
    label = _projective_transform(label, proj_matrix, static_axis, 'NEAREST')

  return image, label


def maybe_add_noise(image, noise_shape, scale0, scale1,
                    image_noise_probability, image_noise_ratio):
  """Add noise at two scales."""

  if image_noise_probability < 0.000001 or (
      image_noise_ratio < 0.000001):
    return image

  noise_list = []
  for scale in [scale0, scale1]:
    rand_image_noise_ratio = tf.random.uniform(
        shape=[], minval=0.0, maxval=image_noise_ratio)
    noise_list.append(
        _rand_noise(0.0, rand_image_noise_ratio, scale, noise_shape))

  skip_noise = tf.greater(tf.random.uniform([]), image_noise_probability)
  image = tf.cond(skip_noise,
                  lambda: image, lambda: image + noise_list[0])
  image = tf.cond(skip_noise,
                  lambda: image, lambda: image + noise_list[1])

  return image


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

  thres = tfp.stats.percentile(pattern, q=percentil_q)
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
  intensity_diff *= 1.15
  intensity_diff = tf.cond(tf.is_nan(intensity_diff),
                           lambda: 0.0, lambda: intensity_diff)

  lesion_liver_ratio = 0.0
  lesion_liver_ratio += tf.random.normal(shape=[], mean=0.01, stddev=0.01)
  lesion_liver_ratio += tf.random.normal(shape=[], mean=0.0, stddev=0.05)
  lesion_liver_ratio = tf.clip_by_value(
      lesion_liver_ratio, min_fake_lesion_ratio, min_fake_lesion_ratio + 0.20)

  fake_lesion_mask = tf.logical_and(
      _gen_rand_mask(ratio_mean=lesion_liver_ratio, ratio_stddev=0.0,
                     scale=reso // 32, shape=label.shape,
                     smoothness=reso // 32),
      tf.logical_not(background_mask))
  liver_mask = tf.logical_not(tf.logical_or(background_mask, fake_lesion_mask))

  # Blur the masks
  lesion_mask_blur = tf.squeeze(tf.nn.conv3d(
      tf.expand_dims(tf.expand_dims(tf.cast(lesion_mask, tf.float32), -1), 0),
      filter=tf.ones([reso // 32] * 3 + [1, 1], tf.float32) / (reso // 32) ** 3,
      strides=[1, 1, 1, 1, 1],
      padding='SAME'))
  fake_lesion_mask_blur = tf.squeeze(tf.nn.conv3d(
      tf.expand_dims(tf.expand_dims(
          tf.cast(fake_lesion_mask, tf.float32), -1), 0),
      filter=tf.ones([reso // 32] * 3 + [1, 1], tf.float32) / (reso // 32) ** 3,
      strides=[1, 1, 1, 1, 1],
      padding='SAME'))

  # Remove real lesion and add fake lesion.
  # If the intensitify is too small (maybe no liver or lesion region labeled),
  # do not generate fake data.
  gen_prob_indicator = tf.cond(
      tf.greater(intensity_diff, 0.0001),
      lambda: gen_prob_indicator, lambda: 0.0)
  # pylint: disable=g-long-lambda
  image = tf.cond(
      tf.greater(gen_prob_indicator, 1 - gen_fake_probability),
      lambda: image + intensity_diff * lesion_mask_blur \
                    - intensity_diff * fake_lesion_mask_blur,
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


def intensity_shift(
    image, label, per_class_intensity_scale, per_class_intensity_shift):
  """Perturb intensity in lesion and non-lesion regions."""

  if per_class_intensity_scale < 0.000001 and (
      per_class_intensity_shift < 0.000001):
    return image

  # Randomly change (mostly increase) intensity of non-lesion region.
  per_class_noise = _truncated_normal(
      per_class_intensity_shift, per_class_intensity_scale)
  image = image + per_class_noise * (
      image * tf.cast(tf.greater(label, 1.5), tf.float32))

  # Randomly change (mostly decrease) intensity of lesion region.
  per_class_noise = _truncated_normal(
      -per_class_intensity_shift, per_class_intensity_scale)
  image = image + per_class_noise * (
      image * tf.cast(tf.less(label, 1.5), tf.float32))

  return image


def image_corruption(
    image, label, reso, image_corrupt_ratio_mean, image_corrupt_ratio_stddev):
  """Randomly drop non-lesion pixels."""

  if image_corrupt_ratio_mean < 0.000001 and (
      image_corrupt_ratio_stddev < 0.000001):
    return image

  # Corrupt non-lesion region according to keep_mask.
  keep_mask = _gen_rand_mask(
      1 - image_corrupt_ratio_mean,
      image_corrupt_ratio_stddev,
      reso // 3, image.shape)

  keep_mask = tf.logical_or(tf.greater(label, 1.5), keep_mask)
  image *= tf.cast(keep_mask, tf.float32)

  return image
