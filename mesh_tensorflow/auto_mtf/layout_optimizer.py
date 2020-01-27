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

"""Computes layouts for Mesh TensorFlow.

Classes and methods to encode a Mesh TensorFlow computation as a series of
Operations and then find a layout to minimize per-machine memory usage.

Sample Usage:
  mtf_graph = mtf.Graph()
  mesh = mtf.Mesh(mtf_graph, "my_mesh")
  mesh_shape = mtf.convert_to_shape("m1:2;m2:2")
  # Add some operations to mesh using Mesh TensorFlow.
  estimator = memory_estimator.MemoryEstimator(mtf_graph, mesh_shape)
  optimizer = layout_optimizer.LayoutOptimizer(estimator)
  layout = optimizer.solve()
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

from absl import logging
from mesh_tensorflow.auto_mtf import print_cp_model_solution
from mesh_tensorflow.auto_mtf import scheduler
import six
from ortools.sat.python import cp_model


class SolverError(Exception):
  pass


class LayoutOptimizer(object):
  """Tries to compute a good layout for Mesh Tensorflow.

  Given a mesh shape (see Mesh TensorFlow) and several operations, computes a
  good layout (a mapping from TensorFlow dimensions to mesh dimensions) using
  integer programming.

  More formally, suppose T is the set of TensorFlow dimensions and M is the set
  of mesh dimensions. A layout L is a map from (T) to (M union {"unassigned"}),
  designating which tensor dimensions are split using which mesh dimensions.

  We wish to compute a layout that minimizes memory usage. Unfortunately, the
  memory usage doesn't just depend on the layout, but also on how the scheduler
  orders operations; whenever an operation is being performed, there are other
  tensors in memory besides that operation's input and output tensors. The
  layout, however, affects the size of the various tensors in memory, as well as
  *the amount of temporary memory the operation uses*.

  With this in mind, our (boolean) integer program to minimize memory is:

  Variables:
    x_{t->m}: "global" (boolean) variables; takes a value of 1 if t in T is
        assigned to m in M, and a value of 0 otherwise.
    y_{assignment}: "local" (boolean) variables; for every set of TensorFlow
        dimensions used in an operation or tensor and for every (valid)
        assignment from that set of TF dimensions to (M union {"unassigned"}),
        we have one of these which takes a value of 1 if the global variables
        completely agree with that assignment and 0 otherwise.
    z: memory (continuous) variable; the peak memory usage.

  Constraints:
    Operation Constraints: for every operation, no two dimensions used in that
        operation can be mapped to the same mesh dimension (since doing so would
        cause some of its computation to be skipped entirely).
    Global Constraints: we enforce that every TensorFlow dimension is assigned
        to at most one mesh dimension (it may be unassigned).
    (Optional) Divisibility Constraints: we enforce that a TensorFlow dimension
        can only be assigned to a mesh dimension if the latter's size evenly
        divides the former's size.
    Local Constraints: we enforce that out of all assignments that share a
        domain (i.e. set of TensorFlow dimensions), exactly one is chosen.
    Global-to-Local Constraints: we enforce that assignment(t) = m, then
        x_{t->m} must be 1 for y_{assignment} to be 1. We also enforce that if
        assignment(t) = "unassigned", then x_{t->m} must be 0 for all m in M.
    Memory Constraints: for every operation i, the peak memory usage z must be
        least the memory usage during that operation. The latter can be derived
        from memory_contents[i] and the local variables relevant to those
        tensors (their new sizes) and to the operation (temporary memory
        needed).

  Objective Function:
    We want to minimize the variable z. However, we want to tiebreak by the
    number of assigned dimensions (preferring more dimensions), so our
    modified objective is (#MTF Dimensions + 1) * z - sum x_{t->m}. Note that we
    prefer more splitting because in general splits result in smaller tensors
    and less duplicated work.
  """

  def __init__(self, memory_estimator, scheduler_alg="LIST"):
    """Uses a auto_mtf.memory_estimator to set up the integer program.

    Args:
      memory_estimator: a memory_estimator.MemoryEstimator.
      scheduler_alg: an optional string, see scheduler.minimize_peak_memory.
    """
    self._estimator = memory_estimator
    self._scheduler_alg = scheduler_alg
    self._layout_validator = self._estimator.get_layout_validator()
    self._graph = self._estimator.get_graph_interface()
    self._memory_contents = None  # [frozenset(string)]

    # Initialize the model.
    self._model = cp_model.CpModel()

    self._preprocess_input()
    self._initialize_variables()
    self._add_constraints()
    self._build_objective_function()

  def _preprocess_input(self):
    """Computing useful input data structures to ease IP construction."""
    # Compute the sets of MTF dimensions used in operations/tensors.

    # a {string: frozenset(string)}, mapping operation name to MTF dimension
    # names.
    self._operation_name_to_mtf_dimension_set = {}
    # a {string: frozenset(string)}, mapping tensor name to MTF dimension names.
    self._tensor_name_to_mtf_dimension_set = {}

    for operation_name in self._graph.get_all_operation_names():
      self._operation_name_to_mtf_dimension_set[operation_name] = frozenset(
          set(self._graph.get_operation_mtf_dimension_names(
              operation_name)).intersection(
                  self._layout_validator.splittable_mtf_dimension_names))
    for tensor_name in self._graph.get_all_tensor_names():
      self._tensor_name_to_mtf_dimension_set[tensor_name] = frozenset(
          set(self._graph.get_tensor_mtf_dimension_names(tensor_name))
          .intersection(self._layout_validator.splittable_mtf_dimension_names))

    self._operation_mtf_dimension_sets = set(
        self._operation_name_to_mtf_dimension_set.values())
    self._mtf_dimension_sets = self._operation_mtf_dimension_sets | set(
        self._tensor_name_to_mtf_dimension_set.values())

    # Compute possible assignments for each set of MTF dimensions.
    self._assignments = {}  # indexed by MTF dimension set
    for mtf_dimension_set in self._mtf_dimension_sets:
      self._assignments[mtf_dimension_set] = _generate_assignments(
          mtf_dimension_set, self._layout_validator.mesh_dimension_name_to_size)

  def _initialize_variables(self):
    """Initializing the variables of the IP."""
    # Initialize global variables.
    self._global_vars = {}  # Indexed by (MTF dimension, mesh dimension)
    for mtf_dimension_name in (
        self._layout_validator.splittable_mtf_dimension_names):
      for mesh_dimension_name in (
          self._layout_validator.mesh_dimension_name_to_size):
        name = _global_var_name(mtf_dimension_name, mesh_dimension_name)
        self._global_vars[(mtf_dimension_name, mesh_dimension_name)] = (
            self._model.NewBoolVar(name))

    # Initialize local variables.
    self._local_vars = {}  # Indexed by (tensorflow dimension set), then name of
    # assignment.
    for mtf_dimension_set in self._mtf_dimension_sets:
      self._local_vars[mtf_dimension_set] = {}
      for assignment in self._assignments[mtf_dimension_set]:
        # TODO(joshuawang): Avoid hash collision no matter what dimension names
        # are; don't hash by this local var name, swap to using a tuple encoding
        # of the full assignment instead.
        name = _local_var_name(mtf_dimension_set, assignment)
        self._local_vars[mtf_dimension_set][name] = (
            self._model.NewBoolVar(name))

    # Initialize memory variable. We need a crude upper bound on memory, so we
    # use the total size of all tensors under the empty assignment.
    # NOTE(joshuawang): This bound could be improved by factoring in the
    # schedule.
    memory_upper_bound = 0
    for tensor_name in self._graph.get_all_tensor_names():
      if self._graph.is_tensor_on_canonical_device(tensor_name):
        memory_upper_bound += int(self._graph.get_tensor_size(tensor_name))
    self._memory_var = self._model.NewIntVar(0, memory_upper_bound, "z")

  def _add_constraints(self):
    """Adding constraints to the IP."""
    # Add operation constraints.
    for mesh_dimension_name in (
        self._layout_validator.mesh_dimension_name_to_size):
      for mtf_dimension_set in self._operation_mtf_dimension_sets:
        self._model.Add(
            sum(self._global_vars[(mtf_dimension_name, mesh_dimension_name)]
                for mtf_dimension_name in mtf_dimension_set) <= 1)

    # Add global constraints.
    for mtf_dimension_name in (
        self._layout_validator.splittable_mtf_dimension_names):
      self._model.Add(
          sum(self._global_vars[(mtf_dimension_name, mesh_dimension_name)]
              for mesh_dimension_name in (
                  self._layout_validator.mesh_dimension_name_to_size)) <= 1)

    # Add divisibility constraints.
    for mtf_dimension_name in (
        self._layout_validator.splittable_mtf_dimension_names):
      for mesh_dimension_name in (
          self._layout_validator.mesh_dimension_name_to_size):
        if not self._layout_validator.is_valid_assignment(mtf_dimension_name,
                                                          mesh_dimension_name):
          self._model.Add(self._global_vars[(mtf_dimension_name,
                                             mesh_dimension_name)] == 0)

    # Add local constraints.
    for mtf_dimension_set in self._mtf_dimension_sets:
      self._model.Add(
          sum(self._local_vars[mtf_dimension_set][_local_var_name(
              mtf_dimension_set, assignment)]
              for assignment in self._assignments[mtf_dimension_set]) == 1)

    # Add local-to-global constraints.
    for mtf_dimension_set in self._mtf_dimension_sets:
      for assignment in self._assignments[mtf_dimension_set]:
        name = _local_var_name(mtf_dimension_set, assignment)
        for mtf_dimension_name in mtf_dimension_set:
          if mtf_dimension_name in assignment:
            mesh_dimension_name = assignment[mtf_dimension_name]
            self._model.AddImplication(
                self._local_vars[mtf_dimension_set][name],
                self._global_vars[(mtf_dimension_name, mesh_dimension_name)])
          else:
            for mesh_dimension_name in (
                self._layout_validator.mesh_dimension_name_to_size):
              self._model.AddImplication(
                  self._global_vars[(mtf_dimension_name, mesh_dimension_name)],
                  self._local_vars[mtf_dimension_set][name].Not())

    # Add memory constraints.
    tensor_memory_sum = {}
    for tensor_name in self._graph.get_all_tensor_names():
      tensor_memory_sum[tensor_name] = 0
      mtf_dimension_set = self._tensor_name_to_mtf_dimension_set[tensor_name]

      if not self._graph.is_tensor_on_canonical_device(tensor_name):
        continue

      for assignment in self._assignments[mtf_dimension_set]:
        size_under_assignment = self._graph.get_tensor_size(
            tensor_name, assignment,
            self._layout_validator.mesh_dimension_name_to_size)

        name = _local_var_name(mtf_dimension_set, assignment)
        tensor_memory_sum[tensor_name] += (
            size_under_assignment * self._local_vars[mtf_dimension_set][name])

    for tensor_names in self._get_memory_contents():
      self._model.Add(
          sum(tensor_memory_sum[tensor_name]
              for tensor_name in tensor_names) <= self._memory_var)

  def _build_objective_function(self):
    """Builds the objective function of the IP."""
    # Break ties in favor of more assignments.
    scale = len(self._layout_validator.splittable_mtf_dimension_names) + 1
    objective = scale * self._memory_var - sum(six.itervalues(
        self._global_vars))
    self._model.Minimize(objective)

  def _get_memory_contents(self):
    """Runs the scheduler to determine memory contents at every point in time.

    Returns:
      a list of frozenset of strings, where the ith entry describes the tensors
      in memory when executing operation i (where schedule[i] is an index into
      GetAllOperationNames()).
    """
    if self._memory_contents is not None:
      return self._memory_contents

    schedule = scheduler.minimize_peak_memory(self._graph, self._scheduler_alg)
    self._memory_contents = self._graph.compute_memory_contents_under_schedule(
        schedule)

    return self._memory_contents

  def solve(self, print_solution=False):
    """Solves the current integer program and returns the computed layout.

    Args:
      print_solution: An optional boolean indicating whether to print the full
        solution in human-readable format.

    Returns:
      The computed layout (as a string).

    Raises:
      SolverError: the internal solver could not find a solution, or the
          solution found is infeasible.
    """
    # Solve and see how well the solver did.
    self._cp_solver = cp_model.CpSolver()
    status = self._cp_solver.Solve(self._model)
    if status != cp_model.OPTIMAL:
      if status == cp_model.FEASIBLE:
        logging.warning("A potentially suboptimal solution was found.")
      else:
        logging.error("Solver returned status %d.", status)
        raise SolverError("The solver could not solve the problem and returned "
                          "status {}.".format(status))

    # TODO(joshuawang): Verify the solver's solution.
    if print_solution:
      print_cp_model_solution.print_solution(self._model, self._cp_solver)

    # Reconstruct layout from solution.
    layout = []
    for mtf_dimension_name in (
        self._layout_validator.splittable_mtf_dimension_names):
      for mesh_dimension_name in (
          self._layout_validator.mesh_dimension_name_to_size):
        value = self._cp_solver.Value(self._global_vars[(mtf_dimension_name,
                                                         mesh_dimension_name)])
        if value:  # Value is integer.
          layout.append(mtf_dimension_name + ":" + mesh_dimension_name)

    layout.sort()
    return ";".join(layout)

  def evaluate_layout(self, layout):
    """The current objective value for the given layout.

    TODO(joshuawang): The current function does not check that the given
    layout is valid.

    Args:
      layout: a string, representing a layout to evaluate (e.g.
          "d_ff:m1;heads:m2").

    Returns:
      A float, the objective value.
    """
    layout_dict = {}
    if layout:
      for pair in layout.split(";"):
        mtf_dimension_name, mesh_dimension_name = pair.split(":", 1)
        if (mtf_dimension_name in
            self._layout_validator.splittable_mtf_dimension_names):
          layout_dict[mtf_dimension_name] = mesh_dimension_name
        else:
          logging.warning("Skipping unsplittable dimension %s.",
                          mtf_dimension_name)

    tensor_memory = {}  # {string: float}, size of each tensor under our layout
    for tensor_name in self._graph.get_all_tensor_names():
      if self._graph.is_tensor_on_canonical_device(tensor_name):
        tensor_memory[tensor_name] = self._graph.get_tensor_size(
            tensor_name, layout_dict,
            self._layout_validator.mesh_dimension_name_to_size)
      else:
        tensor_memory[tensor_name] = 0.0

    peak_memory_usage = 0.0
    for tensor_names in self._get_memory_contents():
      memory_usage = 0.0
      for tensor_name in tensor_names:
        memory_usage += tensor_memory[tensor_name]
      peak_memory_usage = max(peak_memory_usage, memory_usage)
    return peak_memory_usage


def _global_var_name(splittable_dimension, mesh_dimension):
  """Name for a global variable.

  Args:
    splittable_dimension: the name of a splittable dimension (string)
    mesh_dimension: the name of a mesh dimension (string)

  Returns:
    A string, the variable name.
  """
  return "x_({}:{})".format(splittable_dimension, mesh_dimension)


def _local_var_name(splittable_dimensions, assignment):
  """Name for a local variable.

  Args:
    splittable_dimensions: frozenset of names of splittable dimensions.
    assignment: dict from names of splittable dimensions to names of mesh
      dimensions.

  Returns:
    A string, the variable name.
  """
  assignment_string = []
  for splittable in sorted(splittable_dimensions):
    if splittable in assignment:
      assignment_string.append("{}:{}".format(splittable,
                                              assignment[splittable]))
    else:
      assignment_string.append("{}".format(splittable))
  return "y_(" + ",".join(assignment_string) + ")"


def _generate_assignments(splittable_dimensions, mesh_dimension_to_size):
  """Generates all ways to map splittable dimensions to mesh dimensions.

  Args:
    splittable_dimensions: a frozenset of the names of splittable dimensions.
    mesh_dimension_to_size: a dictionary from mesh dimension name to size.

  Returns:
    A list of the valid assignments. Each assignment is a dict keyed by every
        splittable dimension, whose value is either a mesh dimension or None.
  """
  assignments = []
  for assignment_size in six.moves.xrange(
      1 + min(len(splittable_dimensions), len(mesh_dimension_to_size))):
    for s_dims_chosen in itertools.combinations(splittable_dimensions,
                                                assignment_size):
      for m_dims_chosen in itertools.permutations(mesh_dimension_to_size,
                                                  assignment_size):
        assignments.append(dict(zip(s_dims_chosen, m_dims_chosen)))
  return assignments
