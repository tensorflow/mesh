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

"""Apply data augmentation on the Liver Tumor Segmentation (LiTS) dataset."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os

from absl import app
from absl import flags

from six.moves import range
import tensorflow.compat.v1 as tf  # tf

# pylint: disable=g-direct-tensorflow-import,g-direct-third-party-import
from mesh_tensorflow.experimental import data_aug_lib

FLAGS = flags.FLAGS

flags.DEFINE_string('input_file_pattern', '', 'Path to input CT scans.')
flags.DEFINE_string('output_folder', '', 'Path to output folder.')
flags.DEFINE_string('output_file_prefix',
                    'augmented', 'Filename prefix.')

flags.DEFINE_integer('ct_resolution', 128,
                     'Resolution of CT images along depth, height and '
                     'width dimensions.')

flags.DEFINE_integer('num_data_aug', 1000,
                     'The number of data augmentation output.')
flags.DEFINE_integer('process_no', None, 'Which process number I am.')

flags.DEFINE_float('gen_fake_probability', 0.50,
                   'How much to scale intensities of lesion/non-lesion areas.')
flags.DEFINE_float('min_fake_lesion_ratio', 0.05,
                   'Minimum amount of synthetic lession in liver.')


def _dataset_creator():
  """Returns an unbatched dataset."""
  def _parser_fn(serialized_example):
    """Parses a single tf.Example into image and label tensors."""
    features = {}
    features['image/ct_image'] = tf.FixedLenFeature([], tf.string)
    features['image/label'] = tf.FixedLenFeature([], tf.string)
    parsed = tf.parse_single_example(serialized_example, features=features)

    image = tf.decode_raw(parsed['image/ct_image'], tf.float32)
    label = tf.decode_raw(parsed['image/label'], tf.float32)

    # Preprocess color, clip to 0 ~ 1.
    image = tf.clip_by_value(image / 1024.0 + 0.5, 0, 1)

    spatial_dims = [FLAGS.ct_resolution] * 3
    image = tf.reshape(image, spatial_dims)
    label = tf.reshape(label, spatial_dims)

    image, label = data_aug_lib.maybe_gen_fake_data_based_on_real_data(
        image, label, FLAGS.ct_resolution,
        FLAGS.min_fake_lesion_ratio, FLAGS.gen_fake_probability)

    return image, label

  dataset = tf.data.Dataset.list_files(
      FLAGS.input_file_pattern, shuffle=True).repeat()
  dataset = dataset.apply(functools.partial(
      tf.data.TFRecordDataset, compression_type='GZIP'))
  dataset = dataset.shuffle(2).map(_parser_fn, num_parallel_calls=2)

  return dataset


def save_to_tfrecord(image, label, process_no, idx,
                     output_path, output_file_prefix):
  """Save to TFRecord."""
  d_feature = {}
  d_feature['image/ct_image'] = tf.train.Feature(
      bytes_list=tf.train.BytesList(value=[image.reshape([-1]).tobytes()]))
  d_feature['image/label'] = tf.train.Feature(
      bytes_list=tf.train.BytesList(value=[label.reshape([-1]).tobytes()]))

  example = tf.train.Example(features=tf.train.Features(feature=d_feature))
  serialized = example.SerializeToString()

  result_file = os.path.join(
      output_path,
      '{}-{}-{}.tfrecords'.format(output_file_prefix, process_no, idx))
  options = tf.python_io.TFRecordOptions(
      tf.python_io.TFRecordCompressionType.GZIP)
  with tf.python_io.TFRecordWriter(result_file, options=options) as w:
    w.write(serialized)


def apply_data_aug():
  """Apply data augmentation and save augmented results."""
  if not tf.gfile.IsDirectory(FLAGS.output_folder):
    tf.gfile.MakeDirs(FLAGS.output_folder)

  dataset = _dataset_creator()
  ds_iterator = dataset.make_initializable_iterator()
  image, label = ds_iterator.get_next()

  with tf.Session() as sess:
    sess.run(ds_iterator.initializer)
    for idx in range(FLAGS.num_data_aug):
      image_np, label_np = sess.run([image, label])
      save_to_tfrecord(
          image_np, label_np, FLAGS.process_no, idx,
          FLAGS.output_folder, FLAGS.output_file_prefix)
  return


def main(argv):
  del argv
  apply_data_aug()


if __name__ == '__main__':
  app.run(main)
