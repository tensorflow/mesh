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

# Lint as: python3
"""Implementation of Flat and ProductKey Memory Layers.

See the papers https://arxiv.org/abs/1907.05242 and
https://github.com/facebookresearch/XLM/blob/master/PKM-layer.ipynb
"""

from typing import Tuple
import gin
import mesh_tensorflow as mtf
from mesh_tensorflow.transformer import transformer


@gin.configurable
class ProductKeyValueMemory(transformer.TransformerLayer):
  """Memory network with product key-value structure."""

  def __init__(self,
               key_size: int = gin.REQUIRED,
               n_keys: int = gin.REQUIRED,
               n_heads: int = gin.REQUIRED,
               knn: int = gin.REQUIRED):
    """Creates a ProductKeyValueMemory layer."""
    self.key_size = key_size
    self.n_keys = n_keys
    self.n_values = n_keys**2
    self.n_heads = n_heads
    self.knn = knn

  def call(self, context, x: mtf.Tensor) -> mtf.Tensor:
    """Call the layer."""
    # Initialize Memory Keys and Values
    n_key_dim = mtf.Dimension("n_keys", self.n_keys)
    n_value_dim = mtf.Dimension("n_values", self.n_values)
    key_dim = mtf.Dimension("key", self.key_size // 2)
    value_dim = x.shape.dims[-1]
    head_dim = mtf.Dimension("n_heads", self.n_heads)
    product_dim = mtf.Dimension("product_key", 2)
    keys = mtf.get_variable(
        context.mesh,
        name="keys",
        shape=mtf.Shape([head_dim, product_dim, n_key_dim, key_dim]),
        dtype=context.variable_dtype)
    values = mtf.layers.embedding_weights(
        context.mesh,
        vocab_dim=n_value_dim,
        output_dim=value_dim,
        variable_dtype=context.variable_dtype,
        name="values")

    # Compute query
    new_dims = [head_dim, product_dim, key_dim]
    reduce_dims = x.shape.dims[-1:]
    query = mtf.layers.dense(
        x,
        new_dims,
        reduced_dims=reduce_dims,
        activation=None,
        use_bias=True,
        variable_dtype=context.variable_dtype,
        name="query")  # [b, l, h, 2, k]

    # Note: We use layer norm instead of batch norm to normalize queries.
    # The main advantage is that layer norm works well with the codebase
    # whereas the implementation of batch norm requires handling of tf ops.
    query = mtf.layers.layer_norm(query, query.shape.dims[-1])

    # Retrieve indices and scores
    scores, indices = self.get_indices(keys, query)  # [b, l, h, k]
    scores = mtf.softmax(scores, reduced_dim=scores.shape.dims[-1])
    top_values = mtf.gather(values, indices, n_value_dim)  # [b, l, h, k, v]
    out_values = mtf.einsum([top_values, scores],
                            reduced_dims=scores.shape.dims[-2:])  # [b, l, v]
    return out_values

  def get_indices(self, keys: mtf.Tensor,
                  query: mtf.Tensor) -> Tuple[mtf.Tensor, mtf.Tensor]:
    """Generate score and indices for the query."""
    score_shape = mtf.Shape(query.shape.dims[:-1] + keys.shape.dims[2:3])
    scores = mtf.einsum([query, keys],
                        output_shape=score_shape)  # [b, l, h, 2, n_keys]
    knn_dim = mtf.Dimension("knn", self.knn)
    scores, indices = mtf.top_k(scores, score_shape.dims[-1],
                                knn_dim)  # [b, l, h, 2, knn]

    # Computes the top cartesian products and their indices
    knn_square_dim = mtf.Dimension("knn_square_dim", self.knn**2)
    scores1, scores2 = mtf.unstack(scores, scores.shape.dims[-2])
    scores2 = mtf.rename_dimension(scores2, "knn", "knn2")
    out_shape = mtf.Shape(scores1.shape.dims + scores2.shape.dims[-1:])
    all_scores = mtf.add(scores1, scores2, output_shape=out_shape)
    all_scores = mtf.replace_dimensions(all_scores, out_shape[-2:],
                                        knn_square_dim)

    indices1, indices2 = mtf.unstack(indices, indices.shape.dims[-2])
    indices1 = mtf.multiply(indices1, self.n_keys)
    indices2 = mtf.rename_dimension(indices2, "knn", "knn2")
    all_indices = mtf.add(indices1, indices2, output_shape=out_shape)
    all_indices = mtf.replace_dimensions(all_indices, out_shape[-2:],
                                         knn_square_dim)

    scores, best_indices = mtf.top_k(all_scores, all_scores.shape.dims[-1],
                                     knn_dim)
    return scores, mtf.gather(all_indices, best_indices, knn_square_dim)
