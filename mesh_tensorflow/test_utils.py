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

"""Mesh TensorFlow test utilities."""

import mesh_tensorflow as mtf
import tensorflow.compat.v1 as tf


class NumpyConverter(object):
  """Converter class to convert between mtf.Tensor, tf.Tensor and np.array."""

  def __init__(self):
    self._graph = mtf.Graph()
    self._mesh = mtf.Mesh(self._graph, "mtf_mesh")
    self._session = tf.Session()

  def convert_np_array_to_mtf_tensor(self, x, dim_names=None, dtype=tf.int32):
    """Convert a numpy array to an equivalent mtf.Tensor."""
    dim_sizes = x.shape
    if not dim_names:
      dim_names = [f"dim{i}" for i in range(len(dim_sizes))]

    dims = []
    for dim_size, dim_name in zip(dim_sizes, dim_names):
      dims.append(mtf.Dimension(dim_name, dim_size))
    shape = mtf.Shape(dims)
    x_mtf = mtf.constant(self.mesh, x, shape=shape, dtype=dtype)
    return x_mtf

  def convert_mtf_tensor_to_np_array(self, x_mtf):
    """Convert an mtf.Tensor to a numpy array."""
    _, x_tf = self.convert_mtf_tensor_to_tf_tensor(x_mtf)
    if tf.executing_eagerly():
      return x_tf.numpy()
    else:
      self.session.run(tf.global_variables_initializer())
      return x_tf.eval(session=self.session)

  def convert_mtf_tensor_to_tf_tensor(self, mtf_tensor):
    """Convert an mtf.Tensor to a tf.Tensor."""
    mesh_impl = mtf.placement_mesh_impl.PlacementMeshImpl(
        shape=[], layout={}, devices=[""])
    lowering = mtf.Lowering(self.graph, {self.mesh: mesh_impl})
    return lowering, lowering.export_to_tf_tensor(mtf_tensor)

  @property
  def graph(self):
    return self._graph

  @property
  def mesh(self):
    return self._mesh

  @property
  def session(self):
    return self._session
