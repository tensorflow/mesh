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

"""MeshTensorflow network of Unet with spatial partition.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os

import mesh_tensorflow as mtf
import numpy as np
from six.moves import range
import tensorflow.compat.v1 as tf  # tf

# pylint: disable=g-direct-tensorflow-import,g-direct-third-party-import
from mesh_tensorflow.experimental import data_aug_lib
from tensorflow.python.platform import flags


FLAGS = flags.FLAGS

flags.DEFINE_boolean('sampled_2d_slices', False,
                     'Whether to build model on 2D CT slices instead of 3D.')

flags.DEFINE_integer('ct_resolution', 128,
                     'Resolution of CT images along depth, height and '
                     'width dimensions.')

flags.DEFINE_integer('n_dataset_read_interleave', 16,
                     'The number of interleave processes.')
flags.DEFINE_integer('n_dataset_processes', 16,
                     'The number of data augmentation processes.')
flags.DEFINE_integer('batch_size_train', 32, 'Training batch size.')
flags.DEFINE_integer('batch_size_eval', 32, 'Evaluation batch size.')
flags.DEFINE_integer('image_nx_block', 8, 'The number of x blocks.')
flags.DEFINE_integer('image_ny_block', 8, 'The number of y blocks.')
flags.DEFINE_integer('image_c', 1,
                     'The number of input image channels. '
                     'If sampled_2d_slices is False, image_c should be 1.')
flags.DEFINE_integer('label_c', 3, 'The number of output classes.')
flags.DEFINE_integer('pred_downsample', 2,
                     'Down-sampling the results by the factor of '
                     '`pred_downsample`, before outputing the results.')
flags.DEFINE_boolean('output_ground_truth', True,
                     'Whether to return the ground truth tensor in Unet, '
                     'in addition to returning the prediction tensor.')

flags.DEFINE_integer('n_base_filters', 32, 'The number of filters.')
flags.DEFINE_integer('network_depth', 4, 'The number of pooling layers.')
flags.DEFINE_integer('n_conv_per_block', 2,
                     'The number of conv layers between poolings.')
flags.DEFINE_boolean('with_batch_norm', True, 'Whether to use batch norm.')
flags.DEFINE_float('dropout_keep_p', 0.5, 'Probability to keep activations.')

flags.DEFINE_float('xen_liver_weight', 8,
                   'The weight of liver region pixels, '
                   'when computing the cross-entropy loss')
flags.DEFINE_float('xen_lesion_weight', 16,
                   'The weight of lesion region pixels, '
                   'when computing the cross-entropy loss')
flags.DEFINE_float('dice_loss_weight', 0.2,
                   'The weight of dice loss, ranges from 0 to 1')
flags.DEFINE_float('dice_epsilon', 0.1,
                   'A small value that prevents 0 dividing.')

flags.DEFINE_float('image_translate_ratio', 0.0,
                   'How much you want to translate the image and label, '
                   'for data augmentation.')
flags.DEFINE_float('image_transform_ratio', 0.0,
                   'How much you want to sheer the image and label, '
                   'for data augmentation.')
flags.DEFINE_float('image_noise_probability', 0.0,
                   'Probability of adding noise during data augmentation.')
flags.DEFINE_float('image_noise_ratio', 0.0,
                   'How much random noise you want to add to CT images.')
flags.DEFINE_float('image_corrupt_ratio_mean', 0.0,
                   'How much non-liver area you want to block-out in average.')
flags.DEFINE_float('image_corrupt_ratio_stddev', 0.0,
                   'Std-dev of how much non-liver area you want to block-out.')
flags.DEFINE_float('per_class_intensity_scale', 0.0,
                   'How much to scale intensities of lesion/non-lesion areas.')
flags.DEFINE_float('per_class_intensity_shift', 0.0,
                   'How much to shift intensities of lesion/non-lesion areas.')

flags.DEFINE_string('mtf_dtype', 'bfloat16', 'dtype for MeshTensorflow.')

flags.DEFINE_string('layout',
                    'batch:cores, image_nx_block:rows, image_ny_block:columns',
                    'layout rules')

flags.DEFINE_string('train_file_pattern', '',
                    'Path to CT scan training data.')

flags.DEFINE_string('eval_file_pattern', '',
                    'Path to CT scan evalutation data.')


def get_layout():
  return mtf.convert_to_layout_rules(FLAGS.layout)


def get_dataset_creator(dataset_str):
  """Returns a function that creates an unbatched dataset."""
  if dataset_str == 'train':
    data_file_pattern = FLAGS.train_file_pattern.format(FLAGS.ct_resolution)
    shuffle = True
    interleave = True
  else:
    assert dataset_str == 'eval'
    data_file_pattern = FLAGS.eval_file_pattern.format(FLAGS.ct_resolution)
    shuffle = False
    interleave = False

  def _dataset_creator():
    """Returns an unbatch dataset."""

    def _get_stacked_2d_slices(image_3d, label_3d):
      """Return 2d slices of the 3d scan."""
      image_stack = []
      label_stack = []

      for begin_idx in range(0, FLAGS.ct_resolution - FLAGS.image_c + 1):
        slice_begin = [0, 0, begin_idx]
        slice_size = [FLAGS.ct_resolution, FLAGS.ct_resolution, FLAGS.image_c]
        image = tf.slice(image_3d, slice_begin, slice_size)

        slice_begin = [0, 0, begin_idx + FLAGS.image_c // 2]
        slice_size = [FLAGS.ct_resolution, FLAGS.ct_resolution, 1]
        label = tf.slice(label_3d, slice_begin, slice_size)

        spatial_dims_w_blocks = [FLAGS.image_nx_block,
                                 FLAGS.ct_resolution // FLAGS.image_nx_block,
                                 FLAGS.image_ny_block,
                                 FLAGS.ct_resolution // FLAGS.image_ny_block]

        image = tf.reshape(image, spatial_dims_w_blocks + [FLAGS.image_c])
        label = tf.reshape(label, spatial_dims_w_blocks)

        label = tf.cast(label, tf.int32)
        label = tf.one_hot(label, FLAGS.label_c)

        data_dtype = tf.as_dtype(FLAGS.mtf_dtype)
        image = tf.cast(image, data_dtype)
        label = tf.cast(label, data_dtype)

        image_stack.append(image)
        label_stack.append(label)

      return tf.stack(image_stack), tf.stack(label_stack)

    def _parser_fn(serialized_example):
      """Parses a single tf.Example into image and label tensors."""
      features = {}
      features['image/ct_image'] = tf.FixedLenFeature([], tf.string)
      features['image/label'] = tf.FixedLenFeature([], tf.string)
      parsed = tf.parse_single_example(serialized_example, features=features)

      spatial_dims = [FLAGS.ct_resolution] * 3
      if FLAGS.sampled_2d_slices:
        noise_shape = [FLAGS.ct_resolution] * 2 + [FLAGS.image_c]
      else:
        noise_shape = [FLAGS.ct_resolution] * 3

      image = tf.decode_raw(parsed['image/ct_image'], tf.float32)
      label = tf.decode_raw(parsed['image/label'], tf.float32)

      if dataset_str != 'train':
        # Preprocess intensity, clip to 0 ~ 1.
        # The training set is already preprocessed.
        image = tf.clip_by_value(image / 1024.0 + 0.5, 0, 1)

      image = tf.reshape(image, spatial_dims)
      label = tf.reshape(label, spatial_dims)

      if dataset_str == 'eval' and FLAGS.sampled_2d_slices:
        return _get_stacked_2d_slices(image, label)

      if FLAGS.sampled_2d_slices:
        # Take random slices of images and label
        begin_idx = tf.random_uniform(
            shape=[], minval=0,
            maxval=FLAGS.ct_resolution - FLAGS.image_c + 1, dtype=tf.int32)
        slice_begin = [0, 0, begin_idx]
        slice_size = [FLAGS.ct_resolution, FLAGS.ct_resolution, FLAGS.image_c]

        image = tf.slice(image, slice_begin, slice_size)
        label = tf.slice(label, slice_begin, slice_size)

      if dataset_str == 'train':
        for flip_axis in [0, 1, 2]:
          image, label = data_aug_lib.maybe_flip(image, label, flip_axis)
        image, label = data_aug_lib.maybe_rot180(image, label, static_axis=2)
        image = data_aug_lib.intensity_shift(
            image, label,
            FLAGS.per_class_intensity_scale, FLAGS.per_class_intensity_shift)
        image = data_aug_lib.image_corruption(
            image, label, FLAGS.ct_resolution,
            FLAGS.image_corrupt_ratio_mean, FLAGS.image_corrupt_ratio_stddev)
        image = data_aug_lib.maybe_add_noise(
            image, noise_shape, 1, 4,
            FLAGS.image_noise_probability, FLAGS.image_noise_ratio)
        image, label = data_aug_lib.projective_transform(
            image, label, FLAGS.ct_resolution,
            FLAGS.image_translate_ratio, FLAGS.image_transform_ratio,
            FLAGS.sampled_2d_slices)

      if FLAGS.sampled_2d_slices:
        # Only get the center slice of label.
        label = tf.slice(label, [0, 0, FLAGS.image_c // 2],
                         [FLAGS.ct_resolution, FLAGS.ct_resolution, 1])

      spatial_dims_w_blocks = [FLAGS.image_nx_block,
                               FLAGS.ct_resolution // FLAGS.image_nx_block,
                               FLAGS.image_ny_block,
                               FLAGS.ct_resolution // FLAGS.image_ny_block]
      if not FLAGS.sampled_2d_slices:
        spatial_dims_w_blocks += [FLAGS.ct_resolution]

      image = tf.reshape(image, spatial_dims_w_blocks + [FLAGS.image_c])
      label = tf.reshape(label, spatial_dims_w_blocks)

      label = tf.cast(label, tf.int32)
      label = tf.one_hot(label, FLAGS.label_c)

      data_dtype = tf.as_dtype(FLAGS.mtf_dtype)
      image = tf.cast(image, data_dtype)
      label = tf.cast(label, data_dtype)
      return image, label

    dataset_fn = functools.partial(tf.data.TFRecordDataset,
                                   compression_type='GZIP')
    dataset = tf.data.Dataset.list_files(data_file_pattern,
                                         shuffle=shuffle).repeat()

    if interleave:
      dataset = dataset.apply(
          tf.data.experimental.parallel_interleave(
              lambda file_name: dataset_fn(file_name).prefetch(1),
              cycle_length=FLAGS.n_dataset_read_interleave,
              sloppy=True))
    else:
      dataset = dataset.apply(
          tf.data.experimental.parallel_interleave(
              lambda file_name: dataset_fn(file_name).prefetch(1),
              cycle_length=1,
              sloppy=False))

    if shuffle:
      dataset = dataset.shuffle(FLAGS.n_dataset_processes).map(
          _parser_fn, num_parallel_calls=FLAGS.n_dataset_processes)
    else:
      dataset = dataset.map(_parser_fn)

    if dataset_str == 'eval' and FLAGS.sampled_2d_slices:
      # When evaluating on slices, unbatch slices that belong to one CT scan.
      dataset = dataset.unbatch()

    return dataset

  return _dataset_creator


def get_input_mtf_shapes(dataset_str):
  """Returns a list of mtf.Shapes of input tensors."""
  if dataset_str == 'train':
    batch_dim = mtf.Dimension('batch', FLAGS.batch_size_train)
  else:
    assert dataset_str == 'eval'
    batch_dim = mtf.Dimension('batch', FLAGS.batch_size_eval)
  image_nx_dim = mtf.Dimension('image_nx_block', FLAGS.image_nx_block)
  image_ny_dim = mtf.Dimension('image_ny_block', FLAGS.image_ny_block)
  image_sx_dim = mtf.Dimension('image_sx_block',
                               FLAGS.ct_resolution // FLAGS.image_nx_block)
  image_sy_dim = mtf.Dimension('image_sy_block',
                               FLAGS.ct_resolution // FLAGS.image_ny_block)

  batch_spatial_dims = [batch_dim,
                        image_nx_dim, image_sx_dim,
                        image_ny_dim, image_sy_dim]
  if not FLAGS.sampled_2d_slices:
    image_sz_dim = mtf.Dimension('image_sz_block', FLAGS.ct_resolution)
    batch_spatial_dims += [image_sz_dim]

  image_c_dim = mtf.Dimension('image_c', FLAGS.image_c)
  mtf_image_shape = mtf.Shape(batch_spatial_dims + [image_c_dim])

  label_c_dim = mtf.Dimension('label_c', FLAGS.label_c)
  mtf_label_shape = mtf.Shape(batch_spatial_dims + [label_c_dim])

  return [mtf_image_shape, mtf_label_shape]


class PostProcessor(object):
  """Merge and save evaluation results."""

  def __init__(self):
    self._area_int = []
    self._area_sum = []
    self._instance_i = 0

  def record(self, results, pred_output_dir):
    """Do whatever to the results returned by unet_with_spatial_partition."""
    if FLAGS.output_ground_truth:
      pred_liver, pred_lesion, label, area_int, area_sum, _, global_step = (
          results)
    else:
      pred_liver, pred_lesion, area_int, area_sum, _, global_step = results

    if not tf.gfile.IsDirectory(pred_output_dir):
      tf.gfile.MakeDirs(pred_output_dir)

    if FLAGS.sampled_2d_slices:
      with tf.gfile.Open(os.path.join(
          pred_output_dir, 'pred_liver_{}_{}.npy'.format(
              global_step, self._instance_i)), 'wb') as f:
        np.save(f, pred_liver)

      with tf.gfile.Open(os.path.join(
          pred_output_dir, 'pred_lesion_{}_{}.npy'.format(
              global_step, self._instance_i)), 'wb') as f:
        np.save(f, pred_lesion)

      if FLAGS.output_ground_truth:
        with tf.gfile.Open(os.path.join(
            pred_output_dir, 'label_{}_{}.npy'.format(
                global_step, self._instance_i)), 'wb') as f:
          np.save(f, label)

      self._instance_i += 1
    else:
      pred_liver = self._reshape_to_cubes(pred_liver)
      for ins_i, pred_liver_instance in enumerate(pred_liver):
        with tf.gfile.Open(os.path.join(
            pred_output_dir, 'pred_liver_{}_{}.npy'.format(
                global_step, self._instance_i + ins_i)), 'wb') as f:
          np.save(f, pred_liver_instance)

      pred_lesion = self._reshape_to_cubes(pred_lesion)
      for ins_i, pred_lesion_instance in enumerate(pred_lesion):
        with tf.gfile.Open(os.path.join(
            pred_output_dir, 'pred_lesion_{}_{}.npy'.format(
                global_step, self._instance_i + ins_i)), 'wb') as f:
          np.save(f, pred_lesion_instance)

      if FLAGS.output_ground_truth:
        label = self._reshape_to_cubes(label)
        for ins_i, label_instance in enumerate(label):
          with tf.gfile.Open(os.path.join(
              pred_output_dir, 'label_{}_{}.npy'.format(
                  global_step, self._instance_i + ins_i)), 'wb') as f:
            np.save(f, label_instance)

      self._instance_i += len(pred_liver)

    self._area_int.append(area_int)
    self._area_sum.append(area_sum)

  def finish(self):
    """Merge the results and compute final dice scores."""
    area_int = np.concatenate(self._area_int)
    area_sum = np.concatenate(self._area_sum)

    if FLAGS.sampled_2d_slices:
      # Merge the results on 2d slices.
      assert area_int.size % (FLAGS.ct_resolution - FLAGS.image_c + 1) == 0, (
          'Wrong number of results: {}'.format(area_int.shape))
      area_int = area_int.reshape([-1, FLAGS.ct_resolution - FLAGS.image_c + 1])
      area_int = area_int.sum(axis=1)
      area_sum = area_sum.reshape([-1, FLAGS.ct_resolution - FLAGS.image_c + 1])
      area_sum = area_sum.sum(axis=1)

    dice_per_case = (2 * area_int / (area_sum + 0.001)).mean()
    dice_global = 2 * area_int.sum() / (area_sum.sum() + 0.001)
    # pylint: disable=logging-format-interpolation
    tf.logging.info('dice_per_case: {}, dice_global: {}'.format(
        dice_per_case, dice_global))
    # pylint: enable=logging-format-interpolation

  def _reshape_to_cubes(self, data):
    reso = FLAGS.ct_resolution // FLAGS.pred_downsample
    data = np.transpose(data, (0, 1, 3, 2, 4, 5))
    data = np.reshape(data, (data.shape[0], reso, reso, reso))
    return data


def conv_with_spatial_partition(
    x, sampled_2d_slices, image_nx_dim, image_ny_dim, n_filters,
    keep_p, with_batch_norm, is_training, odim_name, variable_dtype, name):
  """Conv with spatial partition, batch_noram and activation."""
  if sampled_2d_slices:
    x = mtf.layers.conv2d_with_blocks(
        x, mtf.Dimension(odim_name, n_filters),
        filter_size=(3, 3), strides=(1, 1), padding='SAME',
        h_blocks_dim=image_nx_dim, w_blocks_dim=image_ny_dim,
        variable_dtype=variable_dtype,
        name=name,
    )
  else:
    x = mtf.layers.conv3d_with_blocks(
        x, mtf.Dimension(odim_name, n_filters),
        filter_size=(3, 3, 3), strides=(1, 1, 1), padding='SAME',
        d_blocks_dim=image_nx_dim, h_blocks_dim=image_ny_dim,
        variable_dtype=variable_dtype,
        name=name,
    )

  if with_batch_norm:
    x, bn_update_ops = mtf.layers.batch_norm(
        x, is_training, momentum=0.90, epsilon=0.000001,
        dims_idx_start=0, dims_idx_end=-1, name=name)
  else:
    bn_update_ops = []

  x = mtf.leaky_relu(x, 0.1)

  if is_training:
    x = mtf.dropout(x, keep_p)

  return x, bn_update_ops


def deconv_with_spatial_partition(
    x, sampled_2d_slices, image_nx_dim, image_ny_dim, n_filters, keep_p,
    odim_name, variable_dtype, name):
  """Deconvolution with spatial partition."""
  if sampled_2d_slices:
    x = mtf.layers.conv2d_transpose_with_blocks(
        x, mtf.Dimension(odim_name, n_filters),
        filter_size=(2, 2), strides=(2, 2), padding='SAME',
        h_blocks_dim=image_nx_dim, w_blocks_dim=image_ny_dim,
        variable_dtype=variable_dtype,
        name=name,
    )
  else:
    x = mtf.layers.conv3d_transpose_with_blocks(
        x, mtf.Dimension(odim_name, n_filters),
        filter_size=(2, 2, 2), strides=(2, 2, 2), padding='SAME',
        d_blocks_dim=image_nx_dim, h_blocks_dim=image_ny_dim,
        variable_dtype=variable_dtype,
        name=name,
    )

  x = mtf.dropout(x, keep_p)

  return x


def unet_with_spatial_partition(mesh, mesh_impl, dataset_str, images, labels):
  """Builds the UNet model graph, train op and eval metrics.

  Args:
    mesh: a MeshTensorflow.mesh object.
    mesh_impl: a mesh implementation, such as SimdMeshImpl and
      PlacementMeshImpl.
    dataset_str: a string of either train or eval. This is used for batch_norm.
    images: a laid out Tensor with shape [batch, x, y, num_channels]
      or [batch, x, y, z, num_channels].
    labels: a laid out Tensor with shape [batch, x, y, num_classes]
      or [batch, x, y, z, num_classes].

  Returns:
    Prediction and loss.
  """

  is_training = (dataset_str == 'train')
  if dataset_str == 'train':
    batch_dim = mtf.Dimension('batch', FLAGS.batch_size_train)
  else:
    assert dataset_str == 'eval'
    batch_dim = mtf.Dimension('batch', FLAGS.batch_size_eval)
  image_nx_dim = mtf.Dimension('image_nx_block', FLAGS.image_nx_block)
  image_ny_dim = mtf.Dimension('image_ny_block', FLAGS.image_ny_block)
  image_sx_dim = mtf.Dimension('image_sx_block',
                               FLAGS.ct_resolution // FLAGS.image_nx_block)
  image_sy_dim = mtf.Dimension('image_sy_block',
                               FLAGS.ct_resolution // FLAGS.image_ny_block)
  image_sz_dim = mtf.Dimension('image_sz_block', FLAGS.ct_resolution)
  image_c_dim = mtf.Dimension('image_c', FLAGS.image_c)
  label_c_dim = mtf.Dimension('label_c', FLAGS.label_c)
  mtf_images_shape, mtf_labels_shape = get_input_mtf_shapes(dataset_str)

  mtf_dtype = tf.as_dtype(FLAGS.mtf_dtype)
  variable_dtype = mtf.VariableDType(mtf_dtype, mtf_dtype, mtf_dtype)

  # Import input features.
  x = mtf.import_laid_out_tensor(
      mesh,
      mesh_impl.LaidOutTensor(images),
      mtf_images_shape)
  x = mtf.cast(x, mtf_dtype)

  # Import ground truth labels.
  t = mtf.import_laid_out_tensor(
      mesh,
      mesh_impl.LaidOutTensor(labels),
      mtf_labels_shape)
  t = mtf.cast(t, mtf_dtype)

  # Transpose the blocks.
  if FLAGS.sampled_2d_slices:
    x = mtf.transpose(x, [batch_dim,
                          image_nx_dim, image_ny_dim,
                          image_sx_dim, image_sy_dim,
                          image_c_dim])

    t = mtf.transpose(t, [batch_dim,
                          image_nx_dim, image_ny_dim,
                          image_sx_dim, image_sy_dim,
                          label_c_dim])
  else:
    x = mtf.transpose(x, [batch_dim,
                          image_nx_dim, image_ny_dim,
                          image_sx_dim, image_sy_dim,
                          image_sz_dim, image_c_dim])

    t = mtf.transpose(t, [batch_dim,
                          image_nx_dim, image_ny_dim,
                          image_sx_dim, image_sy_dim,
                          image_sz_dim, label_c_dim])

  # Network.
  levels = []
  all_bn_update_ops = []
  # add levels with convolution or down-sampling
  for depth in range(FLAGS.network_depth):
    for n_conv in range(FLAGS.n_conv_per_block):
      if depth == 0 and n_conv == 0:
        # no dropout in 1st layer.
        dropout_keep_p = 1.0
      else:
        dropout_keep_p = FLAGS.dropout_keep_p
      x, bn_update_ops = conv_with_spatial_partition(
          x, FLAGS.sampled_2d_slices,
          image_nx_dim, image_ny_dim,
          FLAGS.n_base_filters * (2**depth),
          dropout_keep_p,
          FLAGS.with_batch_norm,
          is_training,
          'conv_{}_{}'.format(depth, n_conv),
          variable_dtype,
          'conv_down_{}_{}'.format(depth, n_conv))
      all_bn_update_ops.extend(bn_update_ops)
    levels.append(x)

    if depth < FLAGS.network_depth - 1:
      if FLAGS.sampled_2d_slices:
        x = mtf.layers.max_pool2d(x, ksize=(2, 2))
      else:
        x = mtf.layers.max_pool3d(x, ksize=(2, 2, 2))

  # add levels with up-convolution or up-sampling
  for depth in range(FLAGS.network_depth - 1)[::-1]:
    x = deconv_with_spatial_partition(
        x, FLAGS.sampled_2d_slices, image_nx_dim, image_ny_dim,
        FLAGS.n_base_filters * (2**depth),
        FLAGS.dropout_keep_p,
        'conv_{}_{}'.format(depth, FLAGS.n_conv_per_block - 1),
        variable_dtype, 'deconv_{}_0'.format(depth))
    x = mtf.concat(
        [x, levels[depth]],
        concat_dim_name='conv_{}_{}'.format(depth, FLAGS.n_conv_per_block - 1))

    for n_conv in range(FLAGS.n_conv_per_block):
      x, bn_update_ops = conv_with_spatial_partition(
          x, FLAGS.sampled_2d_slices,
          image_nx_dim, image_ny_dim,
          FLAGS.n_base_filters * (2**depth),
          FLAGS.dropout_keep_p,
          FLAGS.with_batch_norm,
          is_training,
          'conv_{}_{}'.format(depth, n_conv),
          variable_dtype,
          'conv_up_{}_{}'.format(depth, n_conv))
      all_bn_update_ops.extend(bn_update_ops)

  # no dropout in the final layer.
  if FLAGS.sampled_2d_slices:
    y = mtf.layers.conv2d_with_blocks(
        x, mtf.Dimension('label_c', FLAGS.label_c),
        filter_size=(1, 1), strides=(1, 1), padding='SAME',
        h_blocks_dim=image_nx_dim, w_blocks_dim=image_ny_dim,
        variable_dtype=variable_dtype,
        name='final_conv_{}'.format(FLAGS.label_c),
    )
  else:
    y = mtf.layers.conv3d_with_blocks(
        x, mtf.Dimension('label_c', FLAGS.label_c),
        filter_size=(1, 1, 1), strides=(1, 1, 1), padding='SAME',
        d_blocks_dim=image_nx_dim, h_blocks_dim=image_ny_dim,
        variable_dtype=variable_dtype,
        name='final_conv_{}'.format(FLAGS.label_c),
    )

  # use mtf.constant to make sure there is no CPU-side constants.
  def scalar(v, dtype):
    return mtf.constant(mesh, v, shape=[], dtype=dtype)

  argmax_t = mtf.argmax(t, label_c_dim)
  liver_t = mtf.cast(mtf.equal(argmax_t, scalar(1, tf.int32)), mtf_dtype)
  lesion_t = mtf.cast(mtf.equal(argmax_t, scalar(2, tf.int32)), mtf_dtype)

  argmax_y = mtf.argmax(y, label_c_dim)
  lesion_y = mtf.cast(mtf.equal(argmax_y, scalar(2, tf.int32)), mtf_dtype)

  # summary of class ratios.
  lesion_pred_ratio = mtf.reduce_mean(lesion_y)
  lesion_label_ratio = mtf.reduce_mean(lesion_t)

  # summary of accuracy.
  accuracy = mtf.reduce_mean(mtf.cast(mtf.equal(argmax_y, argmax_t), mtf_dtype))

  # Cross-entropy loss. Up-weight the liver region.
  pixel_loss = mtf.layers.softmax_cross_entropy_with_logits(y, t, label_c_dim)
  pixel_weight = scalar(1, mtf_dtype) + \
      liver_t * scalar(FLAGS.xen_liver_weight - 1, mtf_dtype) + \
      lesion_t * scalar(FLAGS.xen_lesion_weight - FLAGS.xen_liver_weight,
                        mtf_dtype)
  loss_xen = mtf.reduce_mean(pixel_loss * pixel_weight)

  # Dice loss
  y_prob = mtf.softmax(y, reduced_dim=label_c_dim)
  lesion_prob = mtf.reduce_sum(mtf.slice(y_prob, 2, 1, 'label_c'),
                               reduced_dim=mtf.Dimension('label_c', 1))
  prob_intersect = mtf.reduce_sum(lesion_prob * lesion_t,
                                  output_shape=mtf.Shape([batch_dim]))
  prob_area_sum = mtf.reduce_sum(lesion_prob + lesion_t,
                                 output_shape=mtf.Shape([batch_dim]))
  loss_dice_per_case = mtf.reduce_mean(
      scalar(-2, mtf_dtype) * prob_intersect / (
          prob_area_sum + scalar(FLAGS.dice_epsilon, mtf_dtype)))
  loss_dice_global = scalar(-2, mtf_dtype) * mtf.reduce_sum(prob_intersect) / (
      mtf.reduce_sum(prob_area_sum) + scalar(FLAGS.dice_epsilon, mtf_dtype))

  loss_dice = (loss_dice_per_case + loss_dice_global) * scalar(0.5, mtf_dtype)

  loss = scalar(FLAGS.dice_loss_weight, mtf_dtype) * loss_dice + scalar(
      1 - FLAGS.dice_loss_weight, mtf_dtype) * loss_xen

  intersect = mtf.reduce_sum(lesion_y * lesion_t,
                             output_shape=mtf.Shape([batch_dim]))
  area_sum = mtf.reduce_sum(lesion_y + lesion_t,
                            output_shape=mtf.Shape([batch_dim]))
  # summary of dice.
  dice_per_case = mtf.reduce_mean(scalar(2, mtf_dtype) * intersect / (
      area_sum + scalar(0.000001, mtf_dtype)))
  dice_global = scalar(2, mtf_dtype) * mtf.reduce_sum(intersect) / (
      mtf.reduce_sum(area_sum) + scalar(0.000001, mtf_dtype))

  eval_metrics = {
      'lesion_pred_ratio': lesion_pred_ratio,
      'lesion_label_ratio': lesion_label_ratio,
      'accuracy_of_all_classes': accuracy,
      'lesion_dice_per_case': dice_per_case,
      'lesion_dice_global': dice_global,
      'loss_xen': loss_xen,
      'loss_dice': loss_dice,
      'loss_dice_per_case': loss_dice_per_case,
      'loss_dice_global': loss_dice_global,
  }

  if FLAGS.sampled_2d_slices:
    y_prob_downsampled = mtf.layers.avg_pool2d(
        y_prob, ksize=(FLAGS.pred_downsample,) * 2)
    if FLAGS.output_ground_truth:
      lesion_gt_downsampled = mtf.layers.avg_pool2d(
          mtf.slice(t, 2, 1, 'label_c'), ksize=(FLAGS.pred_downsample,) * 2)
  else:
    y_prob_downsampled = mtf.layers.avg_pool3d(
        y_prob, ksize=(FLAGS.pred_downsample,) * 3)
    if FLAGS.output_ground_truth:
      lesion_gt_downsampled = mtf.layers.avg_pool3d(
          mtf.slice(t, 2, 1, 'label_c'), ksize=(FLAGS.pred_downsample,) * 3)

  liver_prob_downsampled = mtf.slice(y_prob_downsampled, 1, 1, 'label_c')
  lesion_prob_downsampled = mtf.slice(y_prob_downsampled, 2, 1, 'label_c')
  preds = [
      mtf.reduce_sum(liver_prob_downsampled,
                     reduced_dim=mtf.Dimension('label_c', 1)),
      mtf.reduce_sum(lesion_prob_downsampled,
                     reduced_dim=mtf.Dimension('label_c', 1))]

  if FLAGS.output_ground_truth:
    preds.append(mtf.reduce_sum(
        lesion_gt_downsampled, reduced_dim=mtf.Dimension('label_c', 1)))

  preds.extend([intersect, area_sum])

  return preds, loss, eval_metrics, all_bn_update_ops
