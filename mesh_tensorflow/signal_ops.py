"""Spectral ops for Mesh TensorFlow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import itertools
import operator
import os
import re

from mesh_tensorflow import utils
import numpy as np
import six
from six.moves import xrange  # pylint: disable=redefined-builtin

import tensorflow.compat.v1 as tf

from mesh_tensorflow import ops_with_redefined_builtins as mtf

class FFT3DOperation(mtf.Operation):
  """
  Computes the 3-dimensional discrete Fourier transform over the inner-most 3
  dimensions of input tensor. Note that the output FFT is transposed.

  Args:
    input: A Tensor. Must be one of the following types: complex64, complex128
    freq_dims: List of 3 Dimensions representing the frequency dimensions.
    name: A name for the operation (optional).

  Returns:
    A Tensor of shape `input.shape[:-3] + freq_dims`.
  """
  def __init__(self, input,  freq_dims, name=None):
    super(FFT3DOperation, self).__init__([input], name=name or "FFT3D")
    self._freq_dims = freq_dims
    self._output_shape = mtf.Shape(input.shape[:-3]+[freq_dims[1], freq_dims[2], freq_dims[0]])
    self._outputs = [mtf.Tensor(self, mtf.Shape(self._output_shape), input.dtype)]

  def gradient(self, grad_ys):
    dy = grad_ys[0]
    x = self.inputs[0]
    return [ifft3d(dy, x.shape[-3:])]

  def lower(self, lowering):
    mesh_impl = lowering.mesh_impl(self)
    x = self.inputs[0]
    naxes = len(x.shape)
    slices = lowering.tensors[self.inputs[0]]
    # Before performing any operations, we check the splitting
    split_axes = []
    for i in range(3):
        split_axes.append(mesh_impl.tensor_dimension_to_mesh_axis(x.shape.dims[-3:][i]))

    # Perform FFT followed by tranposes
    for i in range(2):
        # Apply FFT along last axis
        slices = mesh_impl.slicewise(tf.spectral.fft, slices)

        # Before transposing the array, making sure the new last dimension will
        # be contiguous
        if split_axes[-2] is not None:
            slices = mesh_impl.alltoall(slices, split_axes[-2],  naxes-1,  naxes-2)
            split_axes[-1] = split_axes[-2]
            split_axes[-2] = None
        perm = np.arange(len(x.shape))
        perm[-3:] = np.roll(perm[-3:], shift=1)
        slices = mesh_impl.slicewise(lambda x: tf.transpose(x, perm), slices)
        split_axes = [split_axes[2], split_axes[0], split_axes[1]]

    # Apply FFT along last axis
    slices = mesh_impl.slicewise(tf.spectral.fft, slices)
    lowering.set_tensor_lowering(self.outputs[0], slices)

def fft3d(x, freq_dims, name=None):
  """
  Computes the 3-dimensional discrete Fourier transform over the inner-most 3
  dimensions of input tensor. Note that the output FFT is transposed.

  Args:
    input: A Tensor. Must be one of the following types: complex64, complex128
    freq_dims: List of 3 Dimensions representing the frequency dimensions.
    name: A name for the operation (optional).

  Returns:
    A Tensor of shape `input.shape[:-3] + freq_dims`.
  """
  return FFT3DOperation(x, freq_dims, name).outputs[0]

class iFFT3DOperation(mtf.Operation):
  """
  Computes the inverse 3-dimensional discrete Fourier transform over the inner-most 3
  dimensions of input tensor. Note that the input FFT is assumed transposed.

  Args:
    input: A Tensor. Must be one of the following types: complex64, complex128
    dims: List of 3 Dimensions representing the direct space dimensions.
    name: A name for the operation (optional).

  Returns:
    A Tensor of shape `input.shape[:-3] + dims`.
  """
  def __init__(self, input,  dims, name=None):
    super(iFFT3DOperation, self).__init__([input], name=name or "iFFT3D")
    self._dims = dims
    self._output_shape = mtf.Shape(input.shape[:-3]+dims)
    self._outputs = [mtf.Tensor(self, mtf.Shape(self._output_shape), input.dtype)]

  def gradient(self, grad_ys):
    dy = grad_ys[0]
    ky, kz, kx = self.inputs[0].shape[-3:]
    return [fft3d(dy, [kx, ky, kz])]

  def lower(self, lowering):
    mesh_impl = lowering.mesh_impl(self)
    x = self.inputs[0]
    naxes = len(x.shape)
    slices = lowering.tensors[self.inputs[0]]
    # Before performing any operations, we check the splitting
    split_axes = []
    for i in range(3):
        split_axes.append(mesh_impl.tensor_dimension_to_mesh_axis(x.shape.dims[-3:][i]))

    # Perform FFT followed by tranposes
    for i in range(2):
        # Apply FFT along last axis
        slices = mesh_impl.slicewise(tf.spectral.ifft, slices)

        # Before transposing the array, making sure the new last dimension will
        # be contiguous
        if split_axes[0] is not None:
            slices = mesh_impl.alltoall(slices, split_axes[0],  naxes-1,  naxes-3)
            split_axes[-1] = split_axes[0]
            split_axes[0] = None
        perm = np.arange(len(x.shape))
        perm[-3:] = np.roll(perm[-3:], shift=-1)
        slices = mesh_impl.slicewise(lambda x: tf.transpose(x, perm), slices)
        split_axes = [split_axes[1], split_axes[2], split_axes[0]]

    # Apply FFT along last axis
    slices = mesh_impl.slicewise(tf.spectral.ifft, slices)
    lowering.set_tensor_lowering(self.outputs[0], slices)

def ifft3d(x, dims, name=None):
  """
  Computes the inverse 3-dimensional discrete Fourier transform over the inner-most 3
  dimensions of input tensor. Note that the input FFT is assumed transposed.

  Args:
    input: A Tensor. Must be one of the following types: complex64, complex128
    dims: List of 3 Dimensions representing the direct space dimensions.
    name: A name for the operation (optional).

  Returns:
    A Tensor of shape `input.shape[:-3] + dims`.
  """
  return iFFT3DOperation(x, dims, name).outputs[0]
