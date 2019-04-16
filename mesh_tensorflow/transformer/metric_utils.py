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

"""Utility functions for computing metrics.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin
from mesh_tensorflow.transformer import metrics

METRICS = {
    "padded_neg_log_perplexity": metrics.padded_neg_log_perplexity,
    "bleu": metrics.bleu,
}


@gin.configurable
def get_metric_fns(metric_names, labels, logits):
  """Generate a dictionary of metric name to metric function.

  Args:
    metric_names: list of strings enumerating the different metrics.
    labels: a tensor where batch is the first dimension.
    logits: a tensor with one more dimension than labels and where the batch is
      the first dimension.

  Returns:
    metric_fns: dict of metric functions keyed by their name.
  """
  metric_fns = {}
  for metric_name in metric_names:
    metric_fn = METRICS.get(metric_name)
    if metric_fn:
      metric_fns[metric_name] = metric_fn(labels, logits)
    else:
      raise ValueError("Metric {} is not implemented".format(metric_name))

  return metric_fns

