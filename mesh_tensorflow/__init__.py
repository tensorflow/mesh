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

"""Mesh TensorFlow."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from mesh_tensorflow import beam_search
from mesh_tensorflow import layers
from mesh_tensorflow import optimize
from mesh_tensorflow import placement_mesh_impl
from mesh_tensorflow import simd_mesh_impl
from mesh_tensorflow import tpu_variables
from mesh_tensorflow import utils
from mesh_tensorflow.ops_with_redefined_builtins import *  # pylint: disable=wildcard-import


# TODO(trandustin): Seal module.
# from tensorflow.python.util.all_util import remove_undocumented  # pylint: disable=line-too-long
#
# _allowed_symbols = None
#
# remove_undocumented(__name__, _allowed_symbols)
