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

r"""Dataset utilities for Transformer example.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import tensorflow as tf
import tensorflow_datasets as tfds


def pack_dataset(dataset, length, keys=None):
  """Creates a 'packed' version of a dataset on-the-fly.

  Borrowed from the tensor2tensor library.
  TODO(noam): make this faster
  TODO(noam): move to another file.

  This is meant to replace the irritation of having to create a separate
  "packed" version of a dataset to train efficiently on TPU.

  Each example in the output dataset represents several examples in the
  input dataset.

  For each key in the input dataset, two additional keys are created:
  <key>_segmentation: an int32 tensor identifying the parts
     representing the original example.
  <key>_position: an int32 tensor identifying the position within the original
     example.

  Example:
  Two input examples get combined to form an output example.
  The input examples are:
  {"inputs": [8, 7, 1, 0], "targets":[4, 1, 0]}
  {"inputs": [2, 3, 4, 1], "targets":[5, 6, 1]}
  The output example is:
  {
                 "inputs": [8, 7, 1, 2, 3, 4, 1, 0, 0, 0]
    "inputs_segmentation": [1, 1, 1, 2, 2, 2, 2, 0, 0, 0]
        "inputs_position": [0, 1, 2, 0, 1, 2, 3, 0, 0, 0]
                "targets": [4, 1, 5, 6, 1, 0, 0, 0, 0, 0]
   "targets_segmentation": [1, 1, 2, 2, 2, 0, 0, 0, 0, 0]
       "targets_position": [0, 1, 0, 1, 2, 0, 0, 0, 0, 0]
  }

  0 represents padding in both the inputs and the outputs.

  Sequences in the incoming examples are truncated to length "length", and the
  sequences in the output examples all have fixed (padded) length "length".

  Args:
    dataset: a tf.data.Dataset
    length: an integer
    keys: a list of strings (e.g. ["inputs", "targets"])

  Returns:
    a tf.data.Dataset
  """
  shapes = dataset.output_shapes
  if keys is None:
    keys = shapes.keys()
  for k in keys:
    if k not in shapes:
      raise ValueError("Key %s not found in dataset.  Available keys are %s"
                       % (k, shapes.keys()))
    if not shapes[k].is_compatible_with(tf.TensorShape([None])):
      raise ValueError("Tensors to be packed must be one-dimensional.")

  # trim to length
  dataset = dataset.map(lambda x: {k: x[k][:length] for k in keys})
  # Setting batch_size=length ensures that the concatenated sequences (if they
  # have length >=1) are sufficient to fill at least one packed example.
  batch_size = length
  dataset = dataset.padded_batch(
      batch_size, padded_shapes={k: [-1] for k in keys})
  return _pack_with_tf_ops(dataset, keys, length)


def _pack_with_tf_ops(dataset, keys, length):
  """Helper-function for packing a dataset which has already been batched.

  See pack_dataset()

  Uses tf.while_loop.  Slow.

  Args:
    dataset: a dataset containing padded batches of examples.
    keys: a list of strings
    length: an integer

  Returns:
    a dataset.
  """
  empty_example = {}
  for k in keys:
    empty_example[k] = tf.zeros([0], dtype=tf.int64)
    empty_example[k + "_position"] = tf.zeros([0], dtype=tf.int32)
  keys_etc = empty_example.keys()

  def write_packed_example(partial, outputs):
    new_partial = empty_example.copy()
    new_outputs = {}
    for k in keys_etc:
      new_outputs[k] = outputs[k].write(
          outputs[k].size(),
          tf.pad(partial[k], [[0, length - tf.size(partial[k])]]))
    return new_partial, new_outputs

  def map_fn(x):
    """Internal function to flat_map over.

    Consumes a batch of input examples and produces a variable number of output
    examples.

    Args:
      x: a single example
    Returns:
      a tf.data.Dataset
    """
    partial = empty_example.copy()
    i = tf.zeros([], dtype=tf.int32)
    dynamic_batch_size = tf.shape(x[keys[0]])[0]
    outputs = {}
    for k in keys:
      outputs[k] = tf.TensorArray(
          tf.int64, size=0, dynamic_size=True, element_shape=[length])
      outputs[k + "_position"] = tf.TensorArray(
          tf.int32, size=0, dynamic_size=True, element_shape=[length])
    def cond_fn(i, partial, outputs):
      del partial, outputs
      return i < dynamic_batch_size
    def body_fn(i, partial, outputs):
      """Body function for while_loop.

      Args:
        i: integer scalar
        partial: dictionary of Tensor (partially-constructed example)
        outputs: dictionary of TensorArray
      Returns:
        A triple containing the new values of the inputs.
      """
      can_append = True
      one_example = {}
      for k in keys:
        val = x[k][i]
        val = val[:tf.reduce_sum(tf.to_int32(tf.not_equal(val, 0)))]
        one_example[k] = val
      for k in keys:
        can_append = tf.logical_and(
            can_append,
            tf.less_equal(
                tf.size(partial[k]) + tf.size(one_example[k]), length))
      def false_fn():
        return write_packed_example(partial, outputs)
      def true_fn():
        return partial, outputs
      partial, outputs = tf.cond(can_append, true_fn, false_fn)
      new_partial = {}
      for k in keys:
        new_seq = one_example[k][:length]
        new_seq_len = tf.size(new_seq)
        new_partial[k] = tf.concat([partial[k], new_seq], 0)
        new_partial[k + "_position"] = tf.concat(
            [partial[k + "_position"],
             tf.range(new_seq_len, dtype=tf.int32)], 0)
      partial = new_partial
      return i+1, partial, outputs

    i, partial, outputs = tf.while_loop(
        cond_fn, body_fn, (i, partial, outputs),
        back_prop=False,
        shape_invariants=(
            tf.TensorShape([]),
            {k: tf.TensorShape([None]) for k in keys_etc},
            {k: tf.TensorShape(None) for k in keys_etc},
            ))
    partial, outputs = write_packed_example(partial, outputs)
    packed = {k: outputs[k].stack() for k in keys_etc}
    for k in keys:
      packed[k + "_segmentation"] = (
          tf.cumsum(tf.to_int32(tf.equal(packed[k + "_position"], 0)), axis=1) *
          tf.to_int32(tf.not_equal(packed[k], 0)))

    return tf.data.Dataset.from_tensor_slices(packed)
  dataset = dataset.flat_map(map_fn)
  return dataset


def _trim_and_pad(t, batch_size, length):
  """Trim/pad to get a tf.Tensor with shape [batch_size, length].

  Args:
    t: a 2d tf.Tensor
    batch_size: an integer
    length: an integer
  Returns:
    a 2d Tensor
  """
  t = t[:batch_size, :length]
  paddings = [
      [0, batch_size - tf.shape(t)[0]], [0, length - tf.shape(t)[1]]]
  t = tf.pad(t, paddings)
  return tf.reshape(t, [batch_size, length])


def trim_and_pad_all_features(features, batch_size, length):
  """Trim and pad all features."""
  return {k: _trim_and_pad(v, batch_size, length) for k, v in features.items()}


def add_eos(x):
  """Increase all ids by 1 and append EOS=1.

  Args:
    x: an unpadded 1d tensor of token ids, or a python list
  Returns:
    the same type as x
  """
  if isinstance(x, tf.Tensor):
    return tf.concat([x + 1, [1]], 0)
  elif isinstance(x, list):
    return [i + 1 for i in x] + [1]
  else:
    raise ValueError("unsupported type for x=%s" % (x,))


def clean_output(ids, vocab_size):
  """Decrease all ids by 1, stop at EOS or padding or OOV.

  Args:
    ids: a list of integers
    vocab_size: an integer
  Returns:
    a list of integers
  """
  ret = []
  for i in ids:
    i -= 1
    if i <= 0 or i >= vocab_size:
      break
    else:
      ret.append(i)
  return ret


def get_dataset(tfds_name, data_dir, train, batch_size, length):
  """Get a tf.data.Dataset. for training/eval.

  Args:
    tfds_name: a string
    data_dir: a string
    train: a boolean
    batch_size: an integer
    length: an integer
  Returns:
    a tf.data.Dataset
  """
  dataset = tfds.load(
      tfds_name,
      split=tfds.Split.TRAIN if train else tfds.Split.VALIDATION,
      as_supervised=True,
      data_dir=data_dir)
  if train:
    dataset = dataset.repeat()
  def my_fn(inputs, targets):
    return {"inputs": add_eos(inputs), "targets": add_eos(targets)}
  dataset = dataset.map(my_fn)
  dataset = pack_dataset(dataset, length=length)
  dataset = dataset.batch(batch_size, drop_remainder=False)
  dataset = dataset.map(
      functools.partial(trim_and_pad_all_features,
                        batch_size=batch_size,
                        length=length))
  return dataset


def padded_vocab_size(vocab_size):
  # shift to make room for EOS=1
  vocab_size += 1
  # pad to multiple of 128
  return vocab_size + (-vocab_size % 128)


def inputs_encoder(tfds_name):
  info = tfds.builder(tfds_name).info
  return info.features[info.supervised_keys[0]].encoder


def targets_encoder(tfds_name):
  info = tfds.builder(tfds_name).info
  return info.features[info.supervised_keys[1]].encoder


def inputs_vocab_size(tfds_name):
  info = tfds.builder(tfds_name).info
  return info.features[info.supervised_keys[0]].encoder.vocab_size


def targets_vocab_size(tfds_name):
  info = tfds.builder(tfds_name).info
  return info.features[info.supervised_keys[1]].encoder.vocab_size
