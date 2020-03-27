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

"""Tests for Mesh TensorFlow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized

import mesh_tensorflow as mtf
import tensorflow.compat.v1 as tf
from tensorflow.python.framework import test_util  # pylint:disable=g-direct-tensorflow-import


class LaidOutTensor(object):
  """LaidOutTensor (see placement_mesh_impl.py, simd_mesh_impl.py) for tests."""

  def __init__(self, tensor_list):
    self.tensor_list = tensor_list


class MeshTensorFlowTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      (mtf.Dimension("x", 5),),
      (("x", 5),),
  )
  def testConvertToDimension(self, inputs):
    dimension = mtf.convert_to_dimension(inputs)
    self.assertEqual(dimension.name, "x")
    self.assertEqual(dimension.size, 5)

  def testConvertToDimensionGenericInputs(self):
    dimension = mtf.convert_to_dimension(None)
    self.assertEqual(dimension, None)
    with self.assertRaises(TypeError):
      mtf.convert_to_dimension(5)

  @parameterized.parameters(
      (mtf.Shape([mtf.Dimension("x", 4),
                  mtf.Dimension("y", 8)]),),
      ("x:4;y:8",),
      ("x:4.y:8",),
      ("x:4 y:8",),
      ("x:4,y:8",),
  )
  def testConvertToShape(self, inputs):
    shape = mtf.convert_to_shape(inputs)
    self.assertEqual(shape, mtf.Shape([mtf.Dimension("x", 4),
                                       mtf.Dimension("y", 8)]))

  def testConvertToShapeGenericInputs(self):
    shape = mtf.convert_to_shape([])
    self.assertEqual(shape.dims, [])
    shape = mtf.convert_to_shape(None)
    self.assertEqual(shape, None)
    with self.assertRaises(ValueError):
      mtf.convert_to_shape("x;4")

  @parameterized.parameters(
      (mtf.LayoutRules([("d_ff", "model"), ("heads", "model")]),),
      ("d_ff:model;heads:model",),
      ("d_ff:model.heads:model",),
      ("d_ff:model heads:model",),
      ("d_ff:model,heads:model",),
      ([("d_ff", "model"), ("heads", "model")],),
  )
  def testConvertToLayoutRules(self, inputs):
    layout_rules = mtf.convert_to_layout_rules(inputs)
    self.assertEqual(
        layout_rules._pairs,
        mtf.LayoutRules([("d_ff", "model"), ("heads", "model")])._pairs)

  def testConvertToLayoutRulesGenericInputs(self):
    with self.assertRaises(ValueError):
      mtf.convert_to_layout_rules("d_ff;heads")

  def testTensorLayout(self):
    tensor_layout = mtf.TensorLayout([0, 2, 1])
    self.assertEqual(tensor_layout.mesh_axis_to_tensor_axis(0), ())
    self.assertEqual(tensor_layout.mesh_axis_to_tensor_axis(1), (0,))
    self.assertEqual(tensor_layout.mesh_axis_to_tensor_axis(2), (0, 2))
    tensor_layout = mtf.TensorLayout([None, 0])
    self.assertFalse(tensor_layout.is_fully_replicated)
    tensor_layout = mtf.TensorLayout([None, None, None])
    self.assertTrue(tensor_layout.is_fully_replicated)

  def testGraph(self):
    graph = mtf.Graph()
    self.assertEmpty(graph.operations)
    self.assertEmpty(graph.trainable_variables)
    self.assertEmpty(graph.all_variables)
    mesh = mtf.Mesh(graph, "mesh_test")
    _ = mtf.import_tf_tensor(mesh,
                             tf_tensor=tf.constant(0.),
                             shape=mtf.Shape([]))
    self.assertLen(graph.operations, 1)
    self.assertEmpty(graph.trainable_variables)
    self.assertEmpty(graph.all_variables)
    _ = mtf.get_variable(mesh, "variable_0", mtf.Shape([]), trainable=True)
    self.assertLen(graph.operations, 2)
    self.assertLen(graph.trainable_variables, 1)
    self.assertLen(graph.all_variables, 1)
    _ = mtf.get_variable(mesh, "variable_1", mtf.Shape([]), trainable=False)
    self.assertLen(graph.operations, 3)
    self.assertLen(graph.trainable_variables, 1)
    self.assertLen(graph.all_variables, 2)

  def testGraphNames(self):
    # Standard Usage.
    graph = mtf.Graph()
    self.assertEqual(graph.unique_name("a"), "a")
    self.assertEqual(graph.unique_name("a"), "a_1")
    self.assertEqual(graph.unique_name("a"), "a_2")

    # Edge cases, the user may choose the name "a_1".
    graph = mtf.Graph()
    self.assertEqual(graph.unique_name("a"), "a")
    self.assertEqual(graph.unique_name("a"), "a_1")
    self.assertEqual(graph.unique_name("a_1"), "a_1_1")

    graph = mtf.Graph()
    self.assertEqual(graph.unique_name("a"), "a")
    self.assertEqual(graph.unique_name("a_1"), "a_1")
    self.assertEqual(graph.unique_name("a"), "a_2")

    # Case insensitive.
    graph = mtf.Graph()
    self.assertEqual(graph.unique_name("a"), "a")
    self.assertEqual(graph.unique_name("A"), "A_1")

  @test_util.run_in_graph_and_eager_modes()
  def testLowering(self):
    graph = mtf.Graph()
    mesh = mtf.Mesh(graph, "my_mesh")
    inputs = tf.constant(0.)
    mtf_inputs = mtf.import_tf_tensor(mesh,
                                      tf_tensor=inputs,
                                      shape=mtf.Shape([]))
    mesh_impl = mtf.placement_mesh_impl.PlacementMeshImpl(
        shape=[], layout={}, devices=[""])
    lowering = mtf.Lowering(graph, {mesh: mesh_impl})

    outputs = lowering.export_to_tf_tensor(mtf_inputs)
    inputs_value, outputs_value = self.evaluate([inputs, outputs])
    self.assertEqual(inputs_value, outputs_value)

    # Check that methods run without error.
    _ = lowering.copy_masters_to_slices()
    _ = lowering.copy_slices_to_masters()

  def testMesh(self):
    graph = mtf.Graph()
    mesh = mtf.Mesh(graph, "my_mesh")
    self.assertEqual(mesh.graph, graph)

  def testMeshImpl(self):
    shape = mtf.Shape([mtf.Dimension("batch", 4),
                       mtf.Dimension("model", 8)])
    layout_rules = mtf.LayoutRules([("batch", "batch"),
                                    ("d_ff", "model"),
                                    ("heads", "model")])
    mesh_impl = mtf.MeshImpl(shape=shape, layout_rules=layout_rules)
    self.assertEqual(mesh_impl.shape, shape)
    self.assertLen(shape, mesh_impl.ndims)
    self.assertEqual(mesh_impl.layout_rules, layout_rules)
    self.assertEqual(mesh_impl.size, shape.size)
    self.assertTrue(mesh_impl.supports_control_dependencies)

    batch = mtf.Dimension("batch", 128)
    length = mtf.Dimension("length", 500)
    d_ff = mtf.Dimension("d_ff", 2048)
    heads = mtf.Dimension("heads", 8)
    self.assertEqual(mesh_impl.tensor_dimension_to_mesh_axis(batch), 0)
    self.assertEqual(mesh_impl.tensor_dimension_to_mesh_axis(d_ff), 1)
    self.assertEqual(mesh_impl.tensor_dimension_to_mesh_axis(heads), 1)
    self.assertEqual(mesh_impl.tensor_layout(mtf.Shape([batch, length, d_ff])),
                     mtf.TensorLayout([0, None, 1]))


class OperationSplittabilityTest(tf.test.TestCase):

  def setUp(self):
    super(OperationSplittabilityTest, self).setUp()
    self.graph = mtf.Graph()
    self.mesh = mtf.Mesh(self.graph, "my_mesh")

    self.a_dim = mtf.Dimension("a", 5)
    self.b_dim = mtf.Dimension("b", 10)
    self.c_dim = mtf.Dimension("c", 15)

    self.ab_shape = mtf.Shape([self.a_dim, self.b_dim])
    self.x = mtf.zeros(self.mesh, self.ab_shape)

    self.batch_dim = mtf.Dimension("batch", 100)
    self.grid_h_dim = mtf.Dimension("grid_h", 10)
    self.grid_w_dim = mtf.Dimension("grid_w", 10)
    self.filter_h_dim = mtf.Dimension("filter_h", 5)
    self.filter_w_dim = mtf.Dimension("filter_w", 5)
    self.in_dim = mtf.Dimension("in", 10)
    self.out_dim = mtf.Dimension("out", 10)
    self.image = mtf.zeros(self.mesh, [self.batch_dim, self.grid_h_dim,
                                       self.grid_w_dim, self.in_dim])

  def testOperation(self):
    operation = mtf.Operation([self.x], name="operation")

    # Everything is splittable.
    self.assertEqual(
        operation._initialize_all_dimensions_as_splittable(),
        (frozenset(["a", "b"]), frozenset()))

    # Everything is unsplittable.
    self.assertEqual(
        operation._initialize_splittable_and_unsplittable_dims("unsplittable"),
        (frozenset(), frozenset(["a", "b"])))

    # Everything is unsplittable except dimension "b".
    self.assertEqual(
        operation._initialize_splittable_and_unsplittable_dims(
            "unsplittable", ["b"]),
        (frozenset(["b"]), frozenset(["a"])))

    self.assertRaises(
        ValueError,
        operation._initialize_splittable_and_unsplittable_dims,
        "invalid")

  def testSlicewiseOperationAndGenericGradOperation(self):
    slicewise_operation = mtf.SlicewiseOperation(
        tf.exp,
        [self.x],
        [self.x.shape],
        [self.x.dtype],
        splittable_dims=[self.a_dim],  # pretend only dim "a" can be split.
        grad_function=lambda op, dy: [dy * op.outputs[0]],
        name="component-wise exp")

    self.assertEqual(slicewise_operation.splittable_dims, frozenset(["a"]))
    self.assertEqual(slicewise_operation.unsplittable_dims, frozenset(["b"]))

    generic_grad_operation = mtf.GenericGradOperation(slicewise_operation,
                                                      [self.x])

    self.assertEqual(generic_grad_operation.splittable_dims,
                     frozenset(["a", "b"]))
    self.assertEqual(generic_grad_operation.unsplittable_dims,
                     frozenset())

  def testScalarMultiplyOperationandScalarAddOperation(self):
    scalar = 2.0
    scalar_multiply_operation = mtf.ScalarMultiplyOperation(self.x, scalar)
    self.assertEqual(scalar_multiply_operation.splittable_dims,
                     frozenset(["a", "b"]))
    self.assertEqual(scalar_multiply_operation.unsplittable_dims, frozenset())

    scalar_add_operation = mtf.ScalarAddOperation(self.x, scalar)
    self.assertEqual(scalar_add_operation.splittable_dims,
                     frozenset(["a", "b"]))
    self.assertEqual(scalar_add_operation.unsplittable_dims, frozenset())

  def testBinaryOpWithBroadcasting(self):
    x2 = mtf.zeros(self.mesh, mtf.Shape([self.a_dim, self.c_dim]))
    binary_op_with_broadcasting = mtf.BinaryOpWithBroadcasting(
        tf.less,
        self.x,
        x2,
        mtf.Shape([self.a_dim, self.b_dim, self.c_dim]),
        tf.bool,
        name="less with broadcasting")

    self.assertEqual(binary_op_with_broadcasting.splittable_dims,
                     frozenset(["a", "b", "c"]))
    self.assertEqual(binary_op_with_broadcasting.unsplittable_dims, frozenset())

  def testBroadcastOperation(self):
    broadcast_operation = mtf.BroadcastOperation(
        self.x, mtf.Shape([self.b_dim, self.c_dim, self.a_dim]))
    self.assertEqual(broadcast_operation.splittable_dims,
                     frozenset(["a", "b", "c"]))
    self.assertEqual(broadcast_operation.unsplittable_dims, frozenset())

  def testReduceOperation(self):
    reduce_operation = mtf.ReduceOperation(self.x, mtf.Shape([self.b_dim]),
                                           "sum")
    self.assertEqual(reduce_operation.splittable_dims, frozenset(["a", "b"]))
    self.assertEqual(reduce_operation.unsplittable_dims, frozenset())

  def testPoolOperation(self):
    reduce_operation = mtf.PoolOperation(self.image, [2, 2], [2, 2], "AVG_2D")
    self.assertEqual(reduce_operation.splittable_dims,
                     frozenset(["batch", "in"]))
    self.assertEqual(reduce_operation.unsplittable_dims,
                     frozenset(["grid_h", "grid_w"]))

  def testConcatOperation(self):
    concat_dim1 = mtf.Dimension("concat", 5)
    concat_dim2 = mtf.Dimension("concat", 7)

    x1 = mtf.zeros(self.mesh, mtf.Shape([self.a_dim, self.b_dim, concat_dim1]))
    x2 = mtf.zeros(self.mesh, mtf.Shape([self.a_dim, self.b_dim, concat_dim2]))

    concat_operation = mtf.ConcatOperation([x1, x2], "concat")
    self.assertEqual(concat_operation.splittable_dims, frozenset(["a", "b"]))
    self.assertEqual(concat_operation.unsplittable_dims, frozenset(["concat"]))

  def testSplitOperation(self):
    split_operation = mtf.SplitOperation(self.x, self.b_dim, [3, 7])
    self.assertEqual(split_operation.splittable_dims, frozenset(["a"]))
    self.assertEqual(split_operation.unsplittable_dims, frozenset(["b"]))

  def testStackOperation(self):
    stack_operation = mtf.StackOperation([self.x, self.x], "stack", axis=0)
    self.assertEqual(stack_operation.splittable_dims, frozenset(["a", "b"]))
    self.assertEqual(stack_operation.unsplittable_dims, frozenset(["stack"]))

  def testUnstackOperation(self):
    unstack_operation = mtf.UnstackOperation(self.x, self.b_dim)
    self.assertEqual(unstack_operation.splittable_dims, frozenset(["a"]))
    self.assertEqual(unstack_operation.unsplittable_dims, frozenset(["b"]))

  def testEinsumOperation(self):
    x2 = mtf.zeros(self.mesh, mtf.Shape([self.a_dim, self.c_dim]))
    einsum_operation = mtf.EinsumOperation([self.x, x2],
                                           mtf.Shape([self.b_dim, self.c_dim]))
    self.assertEqual(einsum_operation.splittable_dims,
                     frozenset(["a", "b", "c"]))
    self.assertEqual(einsum_operation.unsplittable_dims, frozenset())

  def testConv2dOperations(self):
    conv_input = mtf.zeros(
        self.mesh,
        mtf.Shape([self.batch_dim, self.grid_h_dim, self.grid_w_dim,
                   self.in_dim]))
    conv_filter = mtf.zeros(
        self.mesh,
        mtf.Shape([self.filter_h_dim, self.filter_w_dim, self.in_dim,
                   self.out_dim]))
    strides = [1, 1, 1, 1]
    padding = "SAME"

    conv2d_operation = mtf.Conv2dOperation(conv_input, conv_filter, strides,
                                           padding)
    self.assertEqual(conv2d_operation.splittable_dims,
                     frozenset(["batch", "in", "out"]))
    self.assertEqual(conv2d_operation.unsplittable_dims,
                     frozenset(["filter_h", "filter_w", "grid_h", "grid_w"]))

    output = conv2d_operation.outputs[0]
    d_output = mtf.zeros(self.mesh, output.shape)

    conv2d_backprop_input_operation = mtf.Conv2or3dBackpropInputOperation(
        2, False, conv_input.shape, conv_filter, d_output, strides, padding)
    self.assertEqual(conv2d_backprop_input_operation.splittable_dims,
                     frozenset(["batch", "filter_h", "filter_w", "grid_h",
                                "grid_w", "in", "out"]))
    self.assertEqual(conv2d_backprop_input_operation.unsplittable_dims,
                     frozenset())

    conv2d_backprop_filter_operation = mtf.Conv2or3dBackpropFilterOperation(
        2, False, conv_input, conv_filter.shape, d_output, strides, padding)
    self.assertEqual(conv2d_backprop_filter_operation.splittable_dims,
                     frozenset(["batch", "filter_h", "filter_w", "grid_h",
                                "grid_w", "in", "out"]))
    self.assertEqual(conv2d_backprop_filter_operation.unsplittable_dims,
                     frozenset())

  def testShiftOperation(self):
    shift_operation = mtf.ShiftOperation(self.x, -5, self.b_dim, wrap=True)
    self.assertEqual(shift_operation.splittable_dims, frozenset(["a", "b"]))
    self.assertEqual(shift_operation.unsplittable_dims, frozenset())

  def testSliceOperation(self):
    slice_operation = mtf.SliceOperation(self.x, begin=3, size=4,
                                         slice_dim_name="b")
    self.assertEqual(slice_operation.splittable_dims, frozenset(["a"]))
    self.assertEqual(slice_operation.unsplittable_dims, frozenset(["b"]))

  def testPadOperation(self):
    pad_operation = mtf.PadOperation(self.x, [7, 2], "a")
    self.assertEqual(pad_operation.splittable_dims, frozenset(["b"]))
    self.assertEqual(pad_operation.unsplittable_dims, frozenset(["a"]))

  def testOneHotOperation(self):
    x = mtf.zeros(self.mesh, self.ab_shape, dtype=tf.int32)
    one_hot_operation = mtf.OneHotOperation(x, self.c_dim, 1, 0, dtype=tf.bool)
    self.assertEqual(one_hot_operation.splittable_dims,
                     frozenset(["a", "b", "c"]))
    self.assertEqual(one_hot_operation.unsplittable_dims, frozenset())

  def testImportOperation(self):
    tf_x = tf.zeros([5, 10])
    import_operation = mtf.ImportOperation(self.mesh, tf_x, self.ab_shape)
    self.assertEqual(import_operation.splittable_dims, frozenset(["a", "b"]))
    self.assertEqual(import_operation.unsplittable_dims, frozenset())

  def testImportLaidOutTensorOperation(self):
    laid_out_x = LaidOutTensor([self.x])

    import_laid_out_tensor_operation = mtf.ImportLaidOutTensorOperation(
        self.mesh, laid_out_x, self.ab_shape)
    self.assertEqual(import_laid_out_tensor_operation.splittable_dims,
                     frozenset())
    self.assertEqual(import_laid_out_tensor_operation.unsplittable_dims,
                     frozenset(["a", "b"]))

  def testVariableOperations(self):
    var = mtf.Variable(self.mesh,
                       "test_variable",
                       self.ab_shape,
                       mtf.VariableDType(tf.int32, tf.int32, tf.int32),
                       initializer=tf.zeros_initializer(),
                       trainable=True)

    self.assertEqual(var.splittable_dims, frozenset(["a", "b"]))
    self.assertEqual(var.unsplittable_dims, frozenset())

    read_variable = mtf.ReadVariable(var)
    self.assertEqual(read_variable.splittable_dims, frozenset(["a", "b"]))
    self.assertEqual(read_variable.unsplittable_dims, frozenset())

    assign = mtf.Assign([var], [self.x])
    self.assertEqual(assign.splittable_dims, frozenset(["a", "b"]))
    self.assertEqual(assign.unsplittable_dims, frozenset())

    depend = mtf.Depend(read_variable.outputs[0], [assign])
    self.assertEqual(depend.splittable_dims, frozenset(["a", "b"]))
    self.assertEqual(depend.unsplittable_dims, frozenset())

  def testConstant(self):
    constant = mtf.Constant(self.mesh, 0, self.ab_shape, dtype=tf.int32)
    self.assertEqual(constant.splittable_dims, frozenset(["a", "b"]))
    self.assertEqual(constant.unsplittable_dims, frozenset())

  def testStopGradient(self):
    stop_gradient = mtf.StopGradient(self.x)
    self.assertEqual(stop_gradient.splittable_dims, frozenset(["a", "b"]))
    self.assertEqual(stop_gradient.unsplittable_dims, frozenset())

  def testPrintOperation(self):
    print_operation = mtf.PrintOperation(self.x, [self.x], "Tensor x: ")
    self.assertEqual(print_operation.splittable_dims, frozenset(["a", "b"]))
    self.assertEqual(print_operation.unsplittable_dims, frozenset())

  def testReshapeOperation(self):
    reshape_operation = mtf.ReshapeOperation(
        self.x, mtf.Shape([mtf.Dimension("x", 25), mtf.Dimension("y", 2)]))
    self.assertEqual(reshape_operation.splittable_dims,
                     frozenset(["a", "b", "x", "y"]))
    self.assertEqual(reshape_operation.unsplittable_dims, frozenset())

  def testRandomOperation(self):
    random_operation = mtf.RandomOperation(self.mesh, self.ab_shape,
                                           tf.random_uniform)
    self.assertEqual(random_operation.splittable_dims, frozenset(["a", "b"]))
    self.assertEqual(random_operation.unsplittable_dims, frozenset())

  def testWhileLoopOperation(self):
    # This test case implements the following:
    # for i in range(10):
    #   x = x * 2
    i = mtf.constant(self.mesh, 0, mtf.Shape([]))
    cond_fn = lambda i, x: mtf.less(i, 10)
    body_fn = lambda i, x: [mtf.add(i, 1), mtf.multiply(x, 2)]

    while_loop_operation = mtf.WhileLoopOperation(cond_fn, body_fn, [i, self.x])
    self.assertEqual(while_loop_operation.splittable_dims,
                     frozenset(["a", "b"]))
    self.assertEqual(while_loop_operation.unsplittable_dims, frozenset())


class NthSmallestTest(tf.test.TestCase):

  def testNthLargest(self):
    graph = mtf.Graph()
    mesh = mtf.Mesh(graph, "my_mesh")
    a_dim = mtf.Dimension("a", 6)
    b_dim = mtf.Dimension("b", 2)
    inputs = tf.constant([[1, 10],
                          [2, 9],
                          [3, 8],
                          [4, 7],
                          [5, 6],
                          [6, 5]])
    n = 1  # find second largest element (since n is zero-indexed)
    reduced_dim = a_dim
    expected_outputs = tf.constant([5, 9])

    mtf_inputs = mtf.import_tf_tensor(
        mesh, inputs, shape=mtf.Shape([a_dim, b_dim]))
    mtf_outputs = mtf.nth_largest_element(
        mtf_inputs, n, reduced_dim, "test_nth_largest")
    mesh_impl = mtf.placement_mesh_impl.PlacementMeshImpl(
        shape="all:2", layout="a:all", devices=["", ""])
    lowering = mtf.Lowering(graph, {mesh: mesh_impl})
    actual_outputs = lowering.export_to_tf_tensor(mtf_outputs)
    self.assertAllEqual(self.evaluate(actual_outputs),
                        self.evaluate(expected_outputs))

  def testNthSmallestReduceSecondDim(self):
    graph = mtf.Graph()
    mesh = mtf.Mesh(graph, "my_mesh")
    a_dim = mtf.Dimension("a", 6)
    b_dim = mtf.Dimension("b", 2)
    inputs = tf.constant([[1, 10],
                          [2, 9],
                          [3, 8],
                          [4, 7],
                          [5, 6],
                          [6, 5]])
    n = 0  # find smallest element (n is zero-indexed)
    reduced_dim = b_dim
    expected_outputs = tf.constant([1, 2, 3, 4, 5, 5])

    mtf_inputs = mtf.import_tf_tensor(
        mesh, inputs, shape=mtf.Shape([a_dim, b_dim]))
    mtf_outputs = mtf.nth_smallest_element(
        mtf_inputs, n, reduced_dim, "test_nth_smallest")
    mesh_impl = mtf.placement_mesh_impl.PlacementMeshImpl(
        shape="all:2", layout="a:all", devices=["", ""])
    lowering = mtf.Lowering(graph, {mesh: mesh_impl})
    actual_outputs = lowering.export_to_tf_tensor(mtf_outputs)
    self.assertAllEqual(self.evaluate(actual_outputs),
                        self.evaluate(expected_outputs))


class TopKTest(tf.test.TestCase):

  def testTopK(self):
    graph = mtf.Graph()
    mesh = mtf.Mesh(graph, "my_mesh")
    a_dim = mtf.Dimension("a", 6)
    b_dim = mtf.Dimension("b", 2)
    inputs = tf.constant([[1, 10],
                          [2, 9],
                          [3, 8],
                          [4, 7],
                          [5, 6],
                          [6, 5]],
                         dtype=tf.float32)
    k_dim = mtf.Dimension("k", 2)
    d_values = tf.constant([[11, 12], [13, 14]], dtype=tf.float32)
    reduced_dim = a_dim
    expected_values = tf.constant([[6, 5], [10, 9]], dtype=tf.float32)
    expected_indices = tf.constant([[5, 4], [0, 1]])
    expected_d_inputs = tf.constant([[0, 13],
                                     [0, 14],
                                     [0, 0],
                                     [0, 0],
                                     [12, 0],
                                     [11, 0]],
                                    dtype=tf.float32)

    mtf_inputs = mtf.import_fully_replicated(
        mesh, inputs, shape=mtf.Shape([a_dim, b_dim]))
    mtf_d_values = mtf.import_tf_tensor(
        mesh, d_values, shape=mtf.Shape([b_dim, k_dim]))
    mtf_values, mtf_indices = mtf.top_k(mtf_inputs,
                                        reduced_dim=reduced_dim,
                                        k_dim=k_dim,
                                        name="test_nth_smallest")
    [mtf_d_inputs] = mtf.gradients([mtf_values], [mtf_inputs], [mtf_d_values])
    mesh_impl = mtf.placement_mesh_impl.PlacementMeshImpl(
        shape="rows:2,cols:2", layout="a:rows,b:cols", devices=["", "", "", ""])
    lowering = mtf.Lowering(graph, {mesh: mesh_impl})
    actual_values = lowering.export_to_tf_tensor(mtf_values)
    actual_indices = lowering.export_to_tf_tensor(mtf_indices)
    actual_d_inputs = lowering.export_to_tf_tensor(mtf_d_inputs)
    actual_inputs = lowering.export_to_tf_tensor(mtf_inputs)
    self.assertAllEqual(self.evaluate(actual_inputs),
                        self.evaluate(inputs))
    self.assertAllEqual(self.evaluate(actual_values),
                        self.evaluate(expected_values))
    self.assertAllEqual(self.evaluate(actual_indices),
                        self.evaluate(expected_indices))
    self.assertAllEqual(self.evaluate(actual_d_inputs),
                        self.evaluate(expected_d_inputs))


class RecomputeGradTest(tf.test.TestCase):

  def testRecomputeGrad(self):
    graph = mtf.Graph()
    mesh = mtf.Mesh(graph, "my_mesh")
    # let's differentiate x^2 + x
    # dy/dx = 2x+1
    def x_squared_plus_x(x):
      return x * x + x
    x = tf.constant([5, 10], dtype=tf.float32)
    dy = tf.constant([2, 3], dtype=tf.float32)
    two = mtf.Dimension("two", 2)
    expected_y = tf.constant([30, 110], dtype=tf.float32)
    expected_dx = tf.constant([22, 63], dtype=tf.float32)
    mtf_x = mtf.import_fully_replicated(
        mesh, x, shape=mtf.Shape([two]))
    mtf_dy = mtf.import_tf_tensor(
        mesh, dy, shape=mtf.Shape([two]))
    mtf_y = mtf.recompute_grad(x_squared_plus_x, [mtf_x])
    [mtf_dx] = mtf.gradients([mtf_y], [mtf_x], [mtf_dy])
    mesh_impl = mtf.placement_mesh_impl.PlacementMeshImpl(
        shape="processors:2", layout="two:processors", devices=["", ""])
    lowering = mtf.Lowering(graph, {mesh: mesh_impl})
    actual_y = lowering.export_to_tf_tensor(mtf_y)
    actual_dx = lowering.export_to_tf_tensor(mtf_dx)
    self.assertAllEqual(self.evaluate(actual_y),
                        self.evaluate(expected_y))
    self.assertAllEqual(self.evaluate(actual_dx),
                        self.evaluate(expected_dx))


if __name__ == "__main__":
  tf.disable_v2_behavior()
  tf.enable_eager_execution()
  tf.test.main()
