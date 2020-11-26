"""Spectral ops for Mesh TensorFlow."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import tensorflow.compat.v1 as tf

from mesh_tensorflow import ops_with_redefined_builtins as mtf


class FFT3DBaseOperation(mtf.Operation):
  def __init__(self, inputs, dims, inverse=False, name=None):
    self.inverse = inverse
    if self.inverse:
      self.default_name = 'IFFT3D'
      self.tf_op = tf.spectral.ifft
    else:
      self.default_name = 'FFT3D'
      self.tf_op = tf.spectral.fft
    super(FFT3DBaseOperation, self).__init__([inputs], name=name or self.default_name)
    self._dims = dims
    if self.inverse:
      dims_reordered = dims
    else:
      dims_reordered = [dims[1], dims[2], dims[0]]
    self._output_shape = mtf.Shape(inputs.shape[:-3]+dims_reordered)
    self._outputs = [mtf.Tensor(self, mtf.Shape(self._output_shape), inputs.dtype)]

  def gradient(self, grad_ys):
    dy = grad_ys[0]
    if self.inverse:
      ky, kz, kx = self.inputs[0].shape[-3:]
      return [fft3d(dy, [kx, ky, kz])]
    else:
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

      # Perform transform followed by tranposes
      for i in range(2):
        # Apply FFT along last axis
        slices = mesh_impl.slicewise(self.tf_op, slices)

        split_axes, slices = self._transpose(
          mesh_impl,
          split_axes,
          slices,
          naxes,
        )

      # Apply transform along last axis
      slices = mesh_impl.slicewise(self.tf_op, slices)
      lowering.set_tensor_lowering(self.outputs[0], slices)

  def _transpose(self, *args):
      raise NotImplementedError('This function needs to be implemented')


class FFT3DOperation(FFT3DBaseOperation):
  """
  Computes the 3-dimensional discrete Fourier transform over the inner-most 3
  dimensions of input tensor. Note that the output FFT is transposed.

  Args:
    input: A Tensor. Must be one of the following types: complex64, complex128
    freq_dims: List of 3 Dimensions representing the frequency dimensions.
    name: A name for the operation (optional).

  Returns:
    A Tensor of shape `input.shape[:-3] + freq_dims[1] + freq_dims[2] + freq_dims[0]`.
  """
  def __init__(self, inputs,  dims, name=None):
    super(FFT3DOperation, self).__init__(inputs, dims, inverse=False, name=name)

  def _transpose(self, mesh_impl, split_axes, slices, naxes):
      # Before transposing the array, making sure the new last dimension will
      # be contiguous
      if split_axes[-2] is not None:
          slices = mesh_impl.alltoall(slices, split_axes[-2],  naxes-1,  naxes-2)
          split_axes[-1] = split_axes[-2]
          split_axes[-2] = None
      perm = np.arange(naxes)
      perm[-3:] = np.roll(perm[-3:], shift=1)
      slices = mesh_impl.slicewise(lambda x: tf.transpose(x, perm), slices)
      split_axes = [split_axes[2], split_axes[0], split_axes[1]]
      return split_axes, slices


def fft3d(x, freq_dims, name=None):
  """
  Computes the 3-dimensional discrete Fourier transform over the inner-most 3
  dimensions of input tensor. Note that the output FFT is transposed.

  Args:
    input: A Tensor. Must be one of the following types: complex64, complex128
    freq_dims: List of 3 Dimensions representing the frequency dimensions.
    name: A name for the operation (optional).

  Returns:
    A Tensor of shape `input.shape[:-3] + freq_dims[1] + freq_dims[2] + freq_dims[0]`.
  """
  return FFT3DOperation(x, freq_dims, name).outputs[0]

class iFFT3DOperation(FFT3DBaseOperation):
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
  def __init__(self, inputs,  dims, name=None):
    super(iFFT3DOperation, self).__init__(inputs, dims, inverse=True, name=name)

  def _transpose(self, mesh_impl, split_axes, slices, naxes):
    # Before transposing the array, making sure the new last dimension will
    # be contiguous
    if split_axes[0] is not None:
      slices = mesh_impl.alltoall(slices, split_axes[0],  naxes-1,  naxes-3)
      split_axes[-1] = split_axes[0]
      split_axes[0] = None
    perm = np.arange(naxes)
    perm[-3:] = np.roll(perm[-3:], shift=-1)
    slices = mesh_impl.slicewise(lambda x: tf.transpose(x, perm), slices)
    split_axes = [split_axes[1], split_axes[2], split_axes[0]]
    return split_axes, slices


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
