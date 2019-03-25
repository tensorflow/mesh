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


class Encoder(object):
  """Abstract class for encoding strings as lists of integers.

  We will subclass this and wrap multiple implementations of text encoders.
  We follow the convention that ids 0=PAD and 1=EOS are reserved.
  """

  @property
  def vocab_size(self):
    """Number of ids (including 0=PAD and 1=EOS).

    Returns:
      an integer
    """
    raise NotImplementedError("Not implemented.")

  def encode(self, s):
    """Encode a python string as a list of integers.

    Args:
      s: a string
    Returns:
      a list of integers (not terminated by EOS)
    """
    raise NotImplementedError("Not implemented.")

  def decode(self, ids):
    """Decode a list of integers to a python string.

    Args:
      ids: a list of integers (not terminated by EOS)
    Returns:
      a string
    """
    raise NotImplementedError("Not implemented.")

  def encode_tf(self, s):
    """Encode a tf.Scalar string to a tf.Tensor.

    This will be necessary for on-the-fly tokenization.

    Args:
      s: a tf.Scalar with dtype tf.string
    Returns:
      a 1d tf.Tensor with dtype tf.int32
    """
    raise NotImplementedError("Not implemented.")

  def decode_tf(self, ids):
    """Decode in TensorFlow.

    I don't know when we will use this, but it seems logical to
    have if we can.

    Args:
      ids: a 1d tf.Tensor with dtype tf.int32
    Returns:
      a tf Scalar with dtype tf.string
    """
    raise NotImplementedError("Not implemented.")


class TFDSEncoder(Encoder):
  """Wrapper for tensorflow_datasets encoders.

  In the TFDS encoders, ID=0 is reserved for padding.
  We want to also reserve ID=1 for EOS, so we shift all IDs up by 1.
  """

  def __init__(self, tfds_encoder):
    self._tfds_encoder = tfds_encoder

  @property
  def vocab_size(self):
    """Number of ids (including 0=PAD and 1=EOS).

    Returns:
      an integer
    """
    return self._tfds_encoder.vocab_size + 1

  def encode(self, s):
    """Encode a python string as a list of integers.

    Args:
      s: a string
    Returns:
      a list of integers (not terminated by EOS)
    """
    # shift IDs up by 1 to make room for EOS=1 (see class docstring)
    return [i + 1 for i in self._tfds_encoder.encode(s)]

  def decode(self, ids):
    """Decode a list of integers to a python string.

    Args:
      ids: a list of integers (not terminated by EOS)
    Returns:
      a string
    """
    return self._tfds_encoder.decode([i - 1 for i in ids])


class Dataset(object):
  """Abstract dataset class."""

  @property
  def feature_keys(self):
    return self.encoders.keys()

  @property
  def encoders(self):
    """A dictionary from feature key to Encoder.

    Returns:
       a dictionary from string to Encoder.
    """
    raise NotImplementedError("Not implemented")

  def load(self, batch_size, length, train, pack):
    """Get a tf.data.Dataset. for training/eval.

    The tensors in the returned tf.data.Dataset have shape
    [batch_size, length].  Zeros indicate padding.

    length indicates the length of the emitted examples.  Examples with
    inputs/targets longer than length get truncated.

    If pack=False, then each emitted example will contain one
    example emitted by load_internal().

    If pack=True, then multiple examples emitted by load_internal() are
    concatenated to form one combined example with the given length.
    See comments in the function pack_dataset().

    batch_size indicates the number of (combined) examples per batch,
    across all cores.

    Args:
      batch_size: an integer
      length: an integer
      train: a boolean
      pack: a boolean
    Returns:
      a tf.data.Dataset
    """
    dataset = self.load_internal(train)
    if train:
      dataset = dataset.repeat()
    def encode_and_append_eos(features):
      ret = {}
      for k in self.feature_keys:
        v = features[k]
        if v.dtype == tf.string:
          v = self.encoders[k].encode_tf(v)
        v = tf.concat([v, [1]], 0)
        ret[k] = v
      return ret
    dataset = dataset.map(encode_and_append_eos)
    if pack:
      dataset = pack_dataset(dataset, length=length)
    dataset = dataset.batch(batch_size, drop_remainder=False)
    dataset = dataset.map(
        functools.partial(trim_and_pad_all_features,
                          batch_size=batch_size,
                          length=length))
    return dataset

  def load_internal(self, train):
    """Get a tf.data.Dataset containing single examples with no EOS.

    The values in the returned examples can either be raw (tf.string) or
    tokenized (integer datatype).

    Args:
      train: a boolean
    Returns:
      a tf.data.Dataset
    """
    raise NotImplementedError("Not implemented")


class TokenizedTFDSDataset(Dataset):
  """Wrapper around pre-tokenized TFDS dataset."""

  def __init__(self, tfds_name, text2self=False, data_dir=None):
    self._tfds_name = tfds_name
    self._text2self = text2self
    self._data_dir = data_dir
    info = tfds.builder(tfds_name).info
    self._encoders = {
        "targets": TFDSEncoder(
            info.features[info.supervised_keys[1]].encoder)
    }
    if not text2self:
      self._encoders["inputs"] = TFDSEncoder(
          info.features[info.supervised_keys[0]].encoder)

  def load_internal(self, train):
    """Get a tf.data.Dataset containing single examples with no EOS.

    Args:
      train: a boolean
    Returns:
      a tf.data.Dataset
    """
    dataset = tfds.load(
        self._tfds_name,
        split=tfds.Split.TRAIN if train else tfds.Split.VALIDATION,
        as_supervised=True,
        data_dir=self._data_dir)
    def feature_map(inputs, targets):
      if self._text2self:
        return {"targets": targets + 1}
      else:
        return {"inputs": inputs + 1, "targets": targets + 1}
    return dataset.map(feature_map)

  @property
  def encoders(self):
    """A dictionary from feature key to Encoder.

    Returns:
       a dictionary from string to Encoder.
    """
    return self._encoders


def pack_dataset(dataset, length, keys=None):
  """Creates a 'packed' version of a dataset on-the-fly.

  Borrowed from the tensor2tensor library.
  TODO(noam): make this faster

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


def padded_vocab_size(vocab_size, divisor=128):
  """Round up vocabulary size so that it is a multiple of divisor.

  We can only split a dimension if it is a multiple of the mesh-dimension size.

  Args:
    vocab_size: an integer
    divisor: an integer
  Returns:
    an integer
  """
  return vocab_size + (-vocab_size % divisor)
