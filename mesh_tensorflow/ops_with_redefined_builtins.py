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

from mesh_tensorflow.ops import *  # pylint: disable=wildcard-import
from mesh_tensorflow.ops import mtf_abs as abs  # pylint: disable=redefined-builtin,unused-import
from mesh_tensorflow.ops import mtf_pow as pow  # pylint: disable=redefined-builtin,unused-import
from mesh_tensorflow.ops import mtf_range as range  # pylint: disable=redefined-builtin,unused-import
from mesh_tensorflow.ops import mtf_slice as slice  # pylint: disable=redefined-builtin,unused-import



# TODO(trandustin): Seal module.
# from tensorflow.python.util.all_util import remove_undocumented  # pylint: disable=line-too-long
#
# _allowed_symbols = None
#
# remove_undocumented(__name__, _allowed_symbols)
