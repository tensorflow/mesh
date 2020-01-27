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

"""Compute schedules to minimize peak memory usage.

Implementation of alternative methods to compute schedules for the layout
optimizer.

Sample Usage:
  # Construct Mesh TensorFlow graph, mtf_graph.
  graph = mtf.auto_mtf.graph_interface.GraphInterface(mtf_graph)
  schedule = scheduler.MinimizePeakMemory(graph, 'LIST')
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import heapq
import six


def minimize_peak_memory(graph, scheduler_alg):
  """Computes a schedule to minimize peak memory.

  Args:
    graph: an mtf.auto_mtf.graph_interface.GraphInterface.
    scheduler_alg: a string, one of 'NAIVE' or 'LIST'

  Returns:
    an iterable of integers representing the schedule.
  """
  if scheduler_alg == 'NAIVE':
    return _minimize_peak_memory_naive(graph)
  elif scheduler_alg == 'LIST':
    return _minimize_peak_memory_list(graph)
  else:
    raise NotImplementedError('{} is not a scheduler algorithm. It should be '
                              'one of NAIVE or LIST.'
                              .format(scheduler_alg))


def _minimize_peak_memory_naive(graph):
  """Computes the naive schedule [0, 1, 2, ...].

  Args:
    graph: an mtf.auto_mtf.graph_interface.GraphInterface.

  Returns:
    an iterable of integers representing the schedule.
  """
  return six.moves.range(graph.get_num_operations())


def _minimize_peak_memory_list(graph):
  """Computes schedule according to the greedy list heuristic.

  Greedy list heuristic: schedule the operation which results in the most bytes
  of memory being (immediately) freed.
  TODO(joshuawang): Experiment with tiebreaking by preferring more successors.

  Args:
    graph: an mtf.auto_mtf.graph_interface.GraphInterface.

  Returns:
    an iterable of integers representing the schedule.
  """
  schedule = []
  bytes_freed = {}  # {operation_name: bytes freed}
  users_of = collections.defaultdict(set)  # {tensor_name: set(operation_name)}
  in_degree = collections.defaultdict(int)  # {operation_name: in degree}
  operation_id = {}  # {operation_name: id}
  # We want an updatable priority queue, so we use the following workaround:
  # docs.python.org/2/library/heapq.html#priority-queue-implementation-notes
  priority_queue = []  # (negative bytes freed, operation name)

  # Set up the (greedy) topological sort.
  for i, operation_name in enumerate(graph.get_all_operation_names()):
    operation_id[operation_name] = i

    for input_name in graph.get_operation_input_names(operation_name):
      # Note that in _HybridGraphInterface, an operation may use a tensor twice,
      # but we deduplicate (with respect to in_degree) so that we can later use
      # users_of to decrement in_degree.
      if operation_name in users_of[input_name]:
        continue
      users_of[input_name].add(operation_name)
      in_degree[operation_name] += 1

  for operation_name in graph.get_all_operation_names():
    bytes_freed[operation_name] = 0
    # For each input, this operation frees memory if it is the final consumer.
    for input_name in graph.get_operation_input_names(operation_name):
      if len(users_of[input_name]) == 1 and not graph.is_tensor_final(
          input_name):
        bytes_freed[operation_name] += graph.get_tensor_size(input_name)
    # For each output, this operation will require additional bytes of memory
    # (hence negative bytes freed).
    for output_name in graph.get_operation_output_names(operation_name):
      # If the output is used (or is final), then it eats memory.
      if users_of[output_name] or graph.is_tensor_final(output_name):
        bytes_freed[operation_name] -= graph.get_tensor_size(output_name)

  for operation_name in graph.get_all_operation_names():
    if in_degree[operation_name] == 0:
      heapq.heappush(priority_queue,
                     (-bytes_freed[operation_name], operation_name))

  # Do the (greedy) topological sort.
  while priority_queue:
    neg_bytes_freed, operation_name = heapq.heappop(priority_queue)
    if bytes_freed[operation_name] != -neg_bytes_freed:
      continue
    schedule.append(operation_id[operation_name])
    bytes_freed[operation_name] = None

    for output_name in graph.get_operation_output_names(operation_name):
      for other_operation_name in users_of[output_name]:
        in_degree[other_operation_name] -= 1
        if in_degree[other_operation_name] == 0:
          heapq.heappush(priority_queue,
                         (-bytes_freed[other_operation_name],
                          other_operation_name))

    for input_name in graph.get_operation_input_names(operation_name):
      if operation_name not in users_of[input_name]:
        # Used twice by this operation and hence already removed.
        continue
      users_of[input_name].remove(operation_name)
      if len(users_of[input_name]) != 1 or graph.is_tensor_final(output_name):
        continue
      (other_operation_name,) = users_of[input_name]
      bytes_freed[other_operation_name] += graph.get_tensor_size(
          input_name)
      if in_degree[other_operation_name] > 0:
        continue
      # Push another copy into the priority queue with our updated value.
      # The original copy will be ignored since it does not match bytes_freed.
      heapq.heappush(priority_queue, (-bytes_freed[other_operation_name],
                                      other_operation_name))

  return schedule
