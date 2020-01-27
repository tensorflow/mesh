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

"""Convenience method to print information about a pywrapcp solver's solution.

Sample Usage:
    model = pywrapcp.CpModel()
    # Input variables, constraints, and objective into model.
    solver = pywrapcp.CpSolver()
    status = solver.Solve(model)
    # Check the status returned by solver.
    print_pywrapcp_solution.print_solution(model, solver)
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


def print_solution(model, solver):
  """Prints the solution associated with solver.

  If solver has already had Solve() called on it, prints the solution. This
  includes each variable and its assignment, along with the objective function
  and its optimal value.
  If solver has not had Solve() called on it, or there is no feasible solution,
  this will probably crash.

  Args:
    model: A pywrapcp.CpModel object.
    solver: A pywrapcp.CpSolver object.

  Returns:
    Nothing, but prints the solution associated with solver.
  """
  model_proto = model.Proto()
  response_proto = solver.ResponseProto()
  variables_in_objective_map = {}
  maximization = False
  if model_proto.HasField('objective'):
    objective = model_proto.objective
    for i in range(len(objective.vars)):
      variables_in_objective_map[objective.vars[i]] = objective.coeffs[i]
    if objective.scaling_factor < 0.0:
      maximization = True
  variable_assignments = []
  variables_in_objective = []
  num_vars = len(model_proto.variables)
  for var_index in range(num_vars):
    if not model_proto.variables[var_index].name:
      continue
    variable_name = model_proto.variables[var_index].name
    if var_index in variables_in_objective_map:
      coefficient = variables_in_objective_map[var_index]
      if coefficient:
        if maximization:
          coefficient *= -1
        if coefficient < 0:
          variables_in_objective.append(' - {} * {}'.format(
              -coefficient, variable_name))
        elif coefficient > 0:
          variables_in_objective.append(' + {} * {}'.format(
              coefficient, variable_name))
    variable_assignments.append('  {} = {}\n'.format(
        variable_name, response_proto.solution[var_index]))
  print(''.join(variable_assignments), end='')
  # Strip the leading '+' if it exists.
  if variables_in_objective and variables_in_objective[0][1] == '+':
    variables_in_objective[0] = variables_in_objective[0][2:]
  print('{}:{}'.format('Maximize' if maximization else 'Minimize',
                       ''.join(variables_in_objective)))
  print('Objective value: {}\n'.format(solver.ObjectiveValue()))
