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

"""Tests for utilities."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mesh_tensorflow as mtf

from six.moves import xrange  # pylint: disable=redefined-builtin

import tensorflow.compat.v1 as tf


class UtilsTest(tf.test.TestCase):

  def test_variable_placer(self):
    sizes = [100, 0, 0, 0]
    device_list = ['cpu:0', 'cpu:1', 'cpu:2', 'cpu:3']

    with tf.Graph().as_default() as g:
      var_placer = mtf.utils.BalancedVariablePlacer(device_list, sizes)
      graph = mtf.Graph()
      mesh = mtf.Mesh(graph, 'my_mesh', var_placer)

      hidden_dim = mtf.Dimension('hidden', 10)
      output_dim = mtf.Dimension('output_feature', 10)

      for i in xrange(5):
        # Each variable takes 400 Bytes, and will be placed from cpu:1.
        mtf.get_variable(mesh, 'w{}'.format(i), [hidden_dim, output_dim])

      for i in xrange(5):
        var = g.get_tensor_by_name('w{}:0'.format(i))
        device = (i + 1) % len(device_list)
        self.assertEqual('cpu:{}'.format(device), var.device)


if __name__ == '__main__':
  tf.disable_v2_behavior()
  tf.test.main()
