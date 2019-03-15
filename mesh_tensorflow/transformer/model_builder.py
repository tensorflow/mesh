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

r"""Utilities for configuring transformer models.

In the future, maybe we can have a t2t-style registry of layers for building
models from hyperparameters.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from mesh_tensorflow.transformer import transformer
from mesh_tensorflow.transformer import transformer_layers


def simple_layer_stack(include_encdec_attention,
                       num_layers=6,
                       d_ff=2048,
                       num_heads=8,
                       d_kv=128,
                       dropout_rate=0.1):
  """Create a layer stack.

  Args:
    include_encdec_attention: a boolean
    num_layers: an integer
    d_ff: an integer
    num_heads: an integer
    d_kv: an integer
    dropout_rate: a float

  Returns:
    a LayerStack
  """
  ret = []
  for _ in xrange(num_layers):
    ret.append(
        transformer_layers.SelfAttention(
            num_heads=num_heads,
            key_value_size=d_kv,
            attention_kwargs={"dropout_rate": dropout_rate}))
    if include_encdec_attention:
      ret.append(
          transformer_layers.EncDecAttention(
              num_heads=num_heads,
              key_value_size=d_kv,
              attention_kwargs={"dropout_rate": dropout_rate}))
    ret.append(
        transformer_layers.DenseReluDense(
            hidden_size=d_ff,
            dropout_rate=dropout_rate))
  return transformer.LayerStack(ret)
