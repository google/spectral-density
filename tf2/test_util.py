# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Functions used for testing."""

from typing import Callable, List, Union
import tensorflow.compat.v2 as tf

import tensor_list_util


# Shorthand notation: Parameters for which the autograd can compute the gradient
# with respect to are tensors or list of tensors.
Parameters = Union[tf.Tensor, List[tf.Tensor]]


@tf.function
def hessian(function: Callable[[Parameters], tf.Tensor],
            parameters: Parameters) -> Parameters:
  """Computes the Hessian of a given function.

  Useful for testing, although scales very poorly.

  Args:
    function: A function for which we want to compute the Hessian.
    parameters: Parameters with respect to the Hessian should be computed.

  Returns:
    A tensor or list of tensors of same nested structure as `Parameters`,
      representing the Hessian.
  """
  with tf.GradientTape() as outer_tape:
    with tf.GradientTape() as inner_tape:
      value = function(parameters)
    grads = inner_tape.gradient(value, parameters)
    grads = tensor_list_util.tensor_list_to_vector(grads)
  return outer_tape.jacobian(grads, parameters)


def hessian_as_matrix(function: Callable[[Parameters], tf.Tensor],
                      parameters: Parameters) -> tf.Tensor:
  """Computes the Hessian of a given function.

  Same as `hessian`, although return a matrix of size [w_dim, w_dim], where
  `w_dim` is the number of parameters, which makes it easier to work with.

  Args:
    function: A function for which we want to compute the Hessian.
    parameters: Parameters with respect to the Hessian should be computed.

  Returns:
    A tensor of size [w_dim, w_dim] representing the Hessian.
  """
  hessian_as_tensor_list = hessian(function, parameters)
  hessian_as_tensor_list = [
      tf.reshape(e, [e.shape[0], -1]) for e in hessian_as_tensor_list]
  return tf.concat(hessian_as_tensor_list, axis=1)
