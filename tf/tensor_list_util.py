# Copyright 2019 Google LLC
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
"""Utilities for manipulating lists of tensors like Tensorflow gradients."""


import numpy as np
import tensorflow as tf


def inner_product(x, y):
  """Computes an inner product of lists of matching shape tensors.

  Args:
    x: A list of tensors.
    y: A list of tensors of shape matching x.

  Returns:
    The inner (dot) product of x and y, viewing x and y as members of a space
    endowed with the natural component-wise inner product.
  """
  assert len(x) == len(y)
  return tf.reduce_sum([tf.reduce_sum(a * b) for a, b in zip(x, y)])


def add(x, y):
  """Adds two lists of matching shape tensors.

  Args:
    x: A list of tensors.
    y: A list of tensors of shape matching x.

  Returns:
    The component-wise sum of x and y.
  """
  assert len(x) == len(y)
  return [tf.add(a, b) for a, b in zip(x, y)]


def subtract(x, y):
  """Subtracts two lists of matching shape tensors.

  Args:
    x: A list of tensors.
    y: A list of tensors of shape matching x.

  Returns:
    The component-wise difference of x and y.
  """
  assert len(x) == len(y)
  return [tf.subtract(a, b) for a, b in zip(x, y)]


def multiply(x, y):
  """Multiplies two lists of matching shape tensors.

  Args:
    x: A list of tensors.
    y: A list of tensors of shape matching x.

  Returns:
    The component-wise product of x and y.
  """
  assert len(x) == len(y)
  return [tf.multiply(a, b) for a, b in zip(x, y)]


def scale(x, y):
  """Multiplies list of tensors x by scalar y.

  Args:
    x: A scalar tensor, more generally, any tensor which can be broadcast as all
      components of y.
    y: A list of tensors.

  Returns:
    The broadcast multiplication of x and y's components.
  """
  return [tf.multiply(x, a) for a in y]


def l2_squared(x):
  """Computes sum of squares of list of tensors.


  Args:
    x: A list of tensors.

  Returns:
    The l2 squared norm of x.
  """
  return tf.reduce_sum([tf.reduce_sum(tf.square(a)) for a in x])


class GradientPacker(object):
  """Helper class for packing/unpacking heterogenous tensors into a vector.
  """

  def __init__(self, loss, gradients=None, var_list=None):
    """"Construct a new gradient packer object.

    This allows for conversion between lists of tensors and a single flat tensor
    representation more amenable to linear algebra.

    Args:
      loss: A tensor representing the loss that we want to differentiate.
      gradients: An (optional) list of tensors computed by tf.gradients. If this
        is None, then we will manually compute it, at the cost of many ops.
      var_list: An (optional) list of variables that we want to differentiate
        the loss with respect to; if gradients is None, and this is unspecified,
        this defaults to tf.trainable_variables().
    """
    # Dependency injecting gradients potentially saves very many ops.
    if gradients is None:
      if var_list is None:
        var_list = tf.trainable_variables()
      gradients = tf.gradients(loss, var_list)

    self.gradient_length = len(gradients)

    self.real_indices = [i for i, x in enumerate(gradients) if x is not None]

    self.gradient_shapes = [x.shape.as_list() for x in gradients
                            if x is not None]
    self._component_sizes = [np.prod(x) for x in self.gradient_shapes]

  def pack(self, gradients):
    """Converts gradients into a big [1, d] vector.

    Warning: Do not use as inputs grads_and_vars.

    Args:
      gradients: A list of tensors or None values (e.g. the result of
        tf.gradients).

    Returns:
      A [1, d] tensor containing all the entries of the input flattened and
      concatenated into one enormous tensor.
    """
    df = tf.concat([tf.reshape(x, [-1]) for x in gradients if x is not None], 0)
    return tf.expand_dims(df, 0)

  def unpack(self, vec_df, full=False):
    """Converts a (gradient) vector into a list of tensors.

    Args:
      vec_df: A packed [1, d] gradient vector (e.g. the result of pack above).
      full:  A boolean indicating whether to include None entries in the output.

    Returns:
      A list of tensors of the same length and shape as the original gradient.
    """
    vec_df = tf.reshape(vec_df, [1, -1])

    grad_parts = tf.split(vec_df, self._component_sizes, axis=1)
    grad_parts = [tf.reshape(grad, shape) for grad, shape in
                  zip(grad_parts, self.gradient_shapes)]

    if full:
      components = [None] * self.gradient_length

      for i in range(len(self.gradient_shapes)):
        components[self.real_indices[i]] = grad_parts[i]
      return components
    else:
      return grad_parts

  @property
  def gradient_size(self):
    """Number of entries in the gradient."""
    return np.sum(self._component_sizes)
