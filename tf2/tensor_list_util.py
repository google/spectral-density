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
"""Utility functions to convert between lists of tensors and vectors."""

from typing import List
import numpy as np
import tensorflow.compat.v2 as tf


def tensor_list_to_vector(tensor_list: List[tf.Tensor]) -> tf.Tensor:
  """Convert a list of tensors into a single (vertical) tensor.

  Useful to convert a model's `trainable_parameters` (list of tf.Variables) into
  a single vector (more convenient for Lanczos tridiagonalization).

  Args:
    tensor_list: List of tensorflow tensors to convert to a single vector.

  Returns:
    A vector obtained by concatenation and reshaping of the original tensors.
      The shape of the vector will be [w_dim, 1] where w_dim is the total number
      of scalars in the list of tensors.
  """
  return tf.concat([tf.reshape(p, [-1, 1]) for p in tensor_list], axis=0)


def vector_to_tensor_list(vector: tf.Tensor,
                          structure: List[tf.Tensor]) -> List[tf.Tensor]:
  """Inverse of `tensor_list_to_vector`.

  Convert a (vertical) vector into a list of tensors, following the shapes of
  the tensors in `structure`. For instance:

  ```
  model_weights = model.trainable_variables
  weights_as_vector = tensor_list_to_vector(model_weights)
  reshaped_weights = vector_to_tensor_list(
    weights_as_vector, structure=model_weights)
  assertShapes(model_weights, reshaped_weights)
  ```

  Args:
    vector: A tensor of shape [w_dim, 1], where w_dim is the total number
      of scalars in `structure`.
    structure: List of tensors defining the shapes of the tensors in the
      returned list. The actual values of the tensors in `structure` don't
      matter.

  Returns:
    A list of tensors of the same shape as the tensors in `structure`

  Raises:
    InvalidArgumentError: If the number of scalars in `vector` doesn't match the
      number of scalars in `structure`.
  """
  current_index = 0
  reshaped_tensors = []
  for example_tensor in structure:
    required_size = np.prod(example_tensor.shape)
    sliced = vector[current_index: current_index + required_size]
    reshaped_tensors.append(tf.reshape(sliced, example_tensor.shape))
    current_index += required_size
  return reshaped_tensors
