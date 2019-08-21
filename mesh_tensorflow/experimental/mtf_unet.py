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

"""MeshTensorflow network of Unet with spatial partition.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import google_type_annotations
from __future__ import print_function

import functools
import mesh_tensorflow as mtf
import tensorflow as tf

# pylint: disable=g-direct-tensorflow-import
from tensorflow.python.platform import flags


FLAGS = flags.FLAGS

tf.flags.DEFINE_boolean('sample_slices', False,
                        'Whether to build model on 2D CT slices instead of 3D.')

tf.flags.DEFINE_integer('ct_resolution', 128,
                        'Resolution of CT images along depth, height and '
                        'width dimensions.')

tf.flags.DEFINE_integer('batch_size_train', 32, 'Training batch size.')
tf.flags.DEFINE_integer('batch_size_eval', 32, 'Evaluation batch size.')
tf.flags.DEFINE_integer('image_nx_block', 8, 'The number of x blocks.')
tf.flags.DEFINE_integer('image_ny_block', 8, 'The number of y blocks.')
tf.flags.DEFINE_integer('image_c', 1,
                        'The number of input image channels. '
                        'If sample_slices is False, image_c should be 1.')
tf.flags.DEFINE_integer('label_c', 3, 'The number of output classes.')

tf.flags.DEFINE_integer('n_base_filters', 32, 'The number of filters.')
tf.flags.DEFINE_integer('network_depth', 4, 'The number of pooling layers.')
tf.flags.DEFINE_boolean('with_batch_norm', True, 'Whether to use batch norm.')

tf.flags.DEFINE_float('xen_liver_weight', 128,
                      'The weight of liver region pixels, '
                      'when computing the cross-entropy loss')
tf.flags.DEFINE_float('xen_lesion_weight', 256,
                      'The weight of lesion region pixels, '
                      'when computing the cross-entropy loss')
tf.flags.DEFINE_float('dice_loss_weight', 0.2,
                      'The weight of dice loss, ranges from 0 to 1')
tf.flags.DEFINE_float('dice_epsilon', 1e-1,
                      'A small value that prevents 0 dividing.')

tf.flags.DEFINE_float('image_translate_ratio', 0.1,
                      'How much you want to translate the image and label, '
                      'for data augmentation.')
tf.flags.DEFINE_float('image_transform_ratio', 0.1,
                      'How much you want to sheer the image and label, '
                      'for data augmentation.')
tf.flags.DEFINE_float('image_noise_probability', 0.80,
                      'Probability of adding noise during data augmentation.')
tf.flags.DEFINE_float('image_noise_ratio', 0.02,
                      'How much random noise you want to add to CT images.')

tf.flags.DEFINE_string('mtf_dtype', 'bfloat16', 'dtype for MeshTensorflow.')

tf.flags.DEFINE_string(
    'layout',
    'batch:rows, image_nx_block:columns, image_ny_block:cores',
    'layout rules')

tf.flags.DEFINE_string('train_file_pattern', '',
                       'Path to CT scan training data.')

tf.flags.DEFINE_string('eval_file_pattern', '',
                       'Path to CT scan evalutation data.')


def get_layout():
  return mtf.convert_to_layout_rules(FLAGS.layout)


def _transform_slices(slices, proj_matrix, static_axis, interpolation):
  """Apply rotation."""
  assert static_axis in [0, 1, 2]
  if static_axis == 2:
    slices = tf.contrib.image.transform(
        slices, proj_matrix, interpolation)
  elif static_axis == 1:
    slices = tf.transpose(slices, [0, 2, 1])
    slices = tf.contrib.image.transform(
        slices, proj_matrix, interpolation)
    slices = tf.transpose(slices, [0, 2, 1])
  else:
    slices = tf.transpose(slices, [2, 1, 0])
    slices = tf.contrib.image.transform(
        slices, proj_matrix, interpolation)
    slices = tf.transpose(slices, [2, 1, 0])
  return slices


def _maybe_add_noise(slices, scales,
                     image_noise_probability, image_noise_ratio):
  """Add noise at multiple scales."""
  noise_list = []
  for scale in scales:
    rand_image_noise_ratio = tf.random.uniform(
        shape=[], minval=0.0, maxval=image_noise_ratio)
    # Eliminate the background intensity with 60 percentile.
    noise_dev = rand_image_noise_ratio * (
        tf.contrib.distributions.percentile(slices, q=95) - \
        tf.contrib.distributions.percentile(slices, q=60))
    noise_shape = [x // scale for x in slices.shape]
    noise = tf.random.normal(
        shape=noise_shape, mean=0.0, stddev=noise_dev)
    noise = tf.clip_by_value(
        noise, -2.0 * noise_dev, 2.0 * noise_dev)
    if scale != 1:
      noise = tf.image.resize_images(
          noise, [slices.shape[0], slices.shape[1]])
      noise = tf.transpose(noise, [0, 2, 1])
      noise = tf.image.resize_images(
          noise, [slices.shape[0], slices.shape[2]])
      noise = tf.transpose(noise, [0, 2, 1])
    noise_list.append(noise)

  skip_noise = tf.greater(tf.random.uniform([]), image_noise_probability)
  slices = tf.cond(skip_noise,
                   lambda: slices, lambda: slices + noise_list[0])
  slices = tf.cond(skip_noise,
                   lambda: slices, lambda: slices + noise_list[1])
  return slices


def _flip_slices(slices, flip_indicator, flip_axis):
  """Randomly flip the image."""
  if flip_axis == 1:
    slices = tf.transpose(slices, [1, 0, 2])
  elif flip_axis == 2:
    slices = tf.transpose(slices, [2, 1, 0])
  slices = tf.cond(tf.less(flip_indicator, 0.5),
                   lambda: slices,
                   lambda: tf.image.flip_up_down(slices))
  if flip_axis == 1:
    slices = tf.transpose(slices, [1, 0, 2])
  elif flip_axis == 2:
    slices = tf.transpose(slices, [2, 1, 0])
  return slices


def _rot90_slices(slices, rot90_k, static_axis):
  """Randomly rotate the image 90/180/270 degrees."""
  if static_axis == 0:
    slices = tf.transpose(slices, [2, 1, 0])
  elif static_axis == 1:
    slices = tf.transpose(slices, [0, 2, 1])
  slices = tf.image.rot90(slices, k=rot90_k)
  if static_axis == 0:
    slices = tf.transpose(slices, [2, 1, 0])
  elif static_axis == 1:
    slices = tf.transpose(slices, [0, 2, 1])
  return slices


def _data_augmentation(image, label, sample_slices):
  """image and label augmentation."""
  def _truncated_normal(stddev):
    v = tf.random.normal(shape=[], mean=0.0, stddev=stddev)
    v = tf.clip_by_value(v, -2 * stddev, 2 * stddev)
    return v

  for static_axis in [0, 1, 2]:
    if sample_slices and static_axis != 2:
      continue
    a0 = _truncated_normal(FLAGS.image_transform_ratio) + 1.0
    a1 = _truncated_normal(FLAGS.image_transform_ratio)
    a2 = _truncated_normal(FLAGS.image_translate_ratio) * FLAGS.ct_resolution
    b0 = _truncated_normal(FLAGS.image_transform_ratio)
    b1 = _truncated_normal(FLAGS.image_transform_ratio) + 1.0
    b2 = _truncated_normal(FLAGS.image_translate_ratio) * FLAGS.ct_resolution
    c0 = _truncated_normal(FLAGS.image_transform_ratio)
    c1 = _truncated_normal(FLAGS.image_transform_ratio)
    proj_matrix = [a0, a1, a2, b0, b1, b2, c0, c1]

    image = _transform_slices(image, proj_matrix, static_axis, 'BILINEAR')
    label = _transform_slices(label, proj_matrix, static_axis, 'NEAREST')

  if FLAGS.image_noise_ratio > 1e-6:
    image = _maybe_add_noise(
        image, [1, 4], FLAGS.image_noise_probability, FLAGS.image_noise_ratio)

  for flip_axis in [0, 1, 2]:
    flip_indicator = tf.random.uniform(shape=[])
    image = _flip_slices(image, flip_indicator, flip_axis)
    label = _flip_slices(label, flip_indicator, flip_axis)

  for static_axis in [0, 1, 2]:
    if sample_slices and static_axis != 2:
      continue
    rot90_k = tf.random_uniform(shape=[], minval=0, maxval=4, dtype=tf.int32)
    image = _rot90_slices(image, rot90_k, static_axis)
    label = _rot90_slices(label, rot90_k, static_axis)

  return image, label


def get_dataset_creator(dataset_str):
  """Returns a function that creates an unbatched dataset."""
  if dataset_str == 'train':
    data_file_pattern = FLAGS.train_file_pattern.format(FLAGS.ct_resolution)
    shuffle = True
    repeat = True
    interleave = True
  else:
    assert dataset_str == 'eval'
    data_file_pattern = FLAGS.eval_file_pattern.format(FLAGS.ct_resolution)
    shuffle = False
    repeat = False
    interleave = False

  def _dataset_creator():
    """Returns an unbatch dataset."""

    def _parser_fn(serialized_example):
      """Parses a single tf.Example into image and label tensors."""
      features = {}
      features['image/ct_image'] = tf.FixedLenFeature([], tf.string)
      features['image/label'] = tf.FixedLenFeature([], tf.string)
      parsed = tf.parse_single_example(serialized_example, features=features)

      spatial_dims = [FLAGS.ct_resolution] * 3

      image = tf.decode_raw(parsed['image/ct_image'], tf.float32)
      label = tf.decode_raw(parsed['image/label'], tf.float32)

      # Preprocess color, as suggested by https://arxiv.org/pdf/1707.07734.pdf
      image = tf.clip_by_value(image / 255.0, -2, 2)

      image = tf.reshape(image, spatial_dims)
      label = tf.reshape(label, spatial_dims)

      if FLAGS.sample_slices:
        # Take random slices of images and label
        begin_idx = tf.random_uniform(
            shape=[], minval=0,
            maxval=FLAGS.ct_resolution - FLAGS.image_c + 1, dtype=tf.int32)
        slice_begin = tf.stack([0, 0, begin_idx])
        slice_size = [FLAGS.ct_resolution, FLAGS.ct_resolution, FLAGS.image_c]

        image = tf.slice(image, slice_begin, slice_size)
        label = tf.slice(label, slice_begin, slice_size)

        if dataset_str == 'train':
          image, label = _data_augmentation(image, label, sample_slices=True)

        # Only get the center slice of label.
        label = tf.slice(label, [0, 0, FLAGS.image_c // 2], slice_size)
      elif dataset_str == 'train':
        image, label = _data_augmentation(image, label, sample_slices=False)

      spatial_dims_w_blocks = [FLAGS.image_nx_block,
                               FLAGS.ct_resolution // FLAGS.image_nx_block,
                               FLAGS.image_ny_block,
                               FLAGS.ct_resolution // FLAGS.image_ny_block]
      if not FLAGS.sample_slices:
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
    if repeat:
      dataset = tf.data.Dataset.list_files(data_file_pattern,
                                           shuffle=shuffle).repeat()
    else:
      dataset = tf.data.Dataset.list_files(data_file_pattern, shuffle=shuffle)

    if interleave:
      dataset = dataset.apply(
          tf.data.experimental.parallel_interleave(
              lambda file_name: dataset_fn(file_name).prefetch(1),
              cycle_length=32,
              sloppy=True))
    else:
      dataset = dataset.apply(
          tf.data.experimental.parallel_interleave(
              lambda file_name: dataset_fn(file_name).prefetch(1),
              cycle_length=1,
              sloppy=False))

    if shuffle:
      dataset = dataset.shuffle(64).map(_parser_fn, num_parallel_calls=64)
    else:
      dataset = dataset.map(_parser_fn)

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
  if not FLAGS.sample_slices:
    image_sz_dim = mtf.Dimension('image_sz_block', FLAGS.ct_resolution)
    batch_spatial_dims += [image_sz_dim]

  image_c_dim = mtf.Dimension('image_c', FLAGS.image_c)
  mtf_image_shape = mtf.Shape(batch_spatial_dims + [image_c_dim])

  label_c_dim = mtf.Dimension('label_c', FLAGS.label_c)
  mtf_label_shape = mtf.Shape(batch_spatial_dims + [label_c_dim])

  return [mtf_image_shape, mtf_label_shape]


def conv_with_spatial_partition(x, sample_slices, image_nx_dim, image_ny_dim,
                                n_filters, with_batch_norm, is_training,
                                odim_name, variable_dtype, name):
  """Conv with spatial partition, batch_noram and activation."""
  if sample_slices:
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
        x, is_training, momentum=0.90, epsilon=1e-6,
        dims_idx_start=0, dims_idx_end=-1, name=name)
  else:
    bn_update_ops = []

  return mtf.leaky_relu(x, 0.1), bn_update_ops


def deconv_with_spatial_partition(x, sample_slices, image_nx_dim, image_ny_dim,
                                  n_filters, odim_name, variable_dtype, name):
  """Deconvolution with spatial partition."""
  if sample_slices:
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
  return x


def unet_with_spatial_partition(mesh, dataset_str, images, labels):
  """Builds the UNet model graph, train op and eval metrics.

  Args:
    mesh: a MeshTensorflow.mesh object.
    dataset_str: a string of either train or eval. This is used for batch_norm.
    images: input image Tensor. Shape [batch, x, y, num_channels]
      or [batch, x, y, z, num_channels].
    labels: input label Tensor. Shape [batch, x, y, num_classes]
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
      mtf.simd_mesh_impl.SimdMeshImpl.LaidOutTensor([images]),
      mtf_images_shape)
  x = mtf.cast(x, mtf_dtype)

  # Import ground truth labels.
  t = mtf.import_laid_out_tensor(
      mesh,
      mtf.simd_mesh_impl.SimdMeshImpl.LaidOutTensor([labels]),
      mtf_labels_shape)
  t = mtf.cast(t, mtf_dtype)

  # Transpose the blocks.
  if FLAGS.sample_slices:
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
    x, bn_update_ops = conv_with_spatial_partition(
        x, FLAGS.sample_slices, image_nx_dim, image_ny_dim,
        FLAGS.n_base_filters * (2**depth),
        FLAGS.with_batch_norm, is_training,
        'conv_{}_0'.format(depth),
        variable_dtype, 'conv_down_{}_0'.format(depth))
    all_bn_update_ops.extend(bn_update_ops)

    x, bn_update_ops = conv_with_spatial_partition(
        x, FLAGS.sample_slices, image_nx_dim, image_ny_dim,
        FLAGS.n_base_filters * (2**depth),
        FLAGS.with_batch_norm, is_training,
        'conv_{}_1'.format(depth),
        variable_dtype, 'conv_down_{}_1'.format(depth))
    levels.append(x)
    all_bn_update_ops.extend(bn_update_ops)

    if depth < FLAGS.network_depth - 1:
      if FLAGS.sample_slices:
        x = mtf.layers.max_pool2d(x, ksize=(2, 2))
      else:
        x = mtf.layers.max_pool3d(x, ksize=(2, 2, 2))

  # add levels with up-convolution or up-sampling
  for depth in range(FLAGS.network_depth - 1)[::-1]:
    x = deconv_with_spatial_partition(
        x, FLAGS.sample_slices, image_nx_dim, image_ny_dim,
        FLAGS.n_base_filters * (2**depth),
        'conv_{}_1'.format(depth),
        variable_dtype, 'deconv_{}_0'.format(depth))
    x = mtf.concat([x, levels[depth]],
                   concat_dim_name='conv_{}_1'.format(depth))

    x, bn_update_ops = conv_with_spatial_partition(
        x, FLAGS.sample_slices, image_nx_dim, image_ny_dim,
        FLAGS.n_base_filters * (2**depth),
        FLAGS.with_batch_norm, is_training,
        'conv_{}_0'.format(depth),
        variable_dtype, 'conv_up_{}_0'.format(depth))
    all_bn_update_ops.extend(bn_update_ops)

    x, bn_update_ops = conv_with_spatial_partition(
        x, FLAGS.sample_slices, image_nx_dim, image_ny_dim,
        FLAGS.n_base_filters * (2**depth),
        FLAGS.with_batch_norm, is_training,
        'conv_{}_1'.format(depth),
        variable_dtype, 'conv_up_{}_1'.format(depth))
    all_bn_update_ops.extend(bn_update_ops)

  if FLAGS.sample_slices:
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

  argmax_t = mtf.argmax(t, label_c_dim)
  # Record liver region.
  liver_t = mtf.cast(mtf.equal(argmax_t, 1), mtf_dtype)
  # Keep the lesion (tumor) and background classes. Remove the liver class.
  lesion_idx = mtf.cast(mtf.equal(argmax_t, 2), tf.int32)
  t = mtf.one_hot(lesion_idx * 2, label_c_dim, dtype=mtf_dtype)

  argmax_y = mtf.argmax(y, label_c_dim)
  argmax_t = mtf.argmax(t, label_c_dim)
  lesion_y = mtf.cast(mtf.equal(argmax_y, 2), mtf_dtype)
  lesion_t = mtf.cast(mtf.equal(argmax_t, 2), mtf_dtype)

  # summary of class ratios.
  lesion_pred_ratio = mtf.reduce_mean(lesion_y)
  lesion_label_ratio = mtf.reduce_mean(lesion_t)

  # summary of accuracy.
  accuracy = mtf.reduce_mean(
      mtf.cast(
          mtf.equal(mtf.argmax(y, label_c_dim), mtf.argmax(t, label_c_dim)),
          mtf_dtype
      )
  )

  # Cross-entropy loss. Up-weight the liver region.
  pixel_loss = mtf.layers.softmax_cross_entropy_with_logits(y, t, label_c_dim)
  pixel_weight = 1 + liver_t * (FLAGS.xen_liver_weight - 1) + lesion_t * (
      FLAGS.xen_lesion_weight - FLAGS.xen_liver_weight - 1)
  loss_xen = mtf.reduce_mean(pixel_loss * pixel_weight)

  # Dice loss
  y_prob = mtf.softmax(y, reduced_dim=label_c_dim)
  lesion_prob = mtf.reduce_sum(mtf.slice(y_prob, 2, 1, 'label_c'),
                               reduced_dim=mtf.Dimension('label_c', 1))
  prob_intersect = mtf.reduce_sum(lesion_prob * lesion_t,
                                  output_shape=mtf.Shape([batch_dim]))
  prob_area_sum = mtf.reduce_sum(lesion_prob + lesion_t,
                                 output_shape=mtf.Shape([batch_dim]))
  loss_dice = mtf.reduce_mean(
      -2 * prob_intersect / (prob_area_sum + FLAGS.dice_epsilon))

  loss = FLAGS.dice_loss_weight * loss_dice + (
      1 - FLAGS.dice_loss_weight) * loss_xen

  intersect = mtf.reduce_sum(lesion_y * lesion_t,
                             output_shape=mtf.Shape([batch_dim]))
  area_sum = mtf.reduce_sum(lesion_y + lesion_t,
                            output_shape=mtf.Shape([batch_dim]))
  # summary of dice.
  dice = mtf.reduce_mean(2 * intersect / (area_sum + 1e-6))

  # Transpose it back to the its input shape.
  if FLAGS.sample_slices:
    y = mtf.transpose(y, [batch_dim,
                          image_nx_dim, image_sx_dim,
                          image_ny_dim, image_sy_dim,
                          label_c_dim])
  else:
    y = mtf.transpose(y, [batch_dim,
                          image_nx_dim, image_sx_dim,
                          image_ny_dim, image_sy_dim,
                          image_sz_dim, label_c_dim])

  eval_metrics = {
      'lesion_pred_ratio': lesion_pred_ratio,
      'lesion_label_ratio': lesion_label_ratio,
      'accuracy_of_all_classes': accuracy,
      'dice_of_lesion_region': dice,
      'loss_xen': loss_xen,
      'loss_dice': loss_dice,
  }

  return y, loss, eval_metrics, all_bn_update_ops
