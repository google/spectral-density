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
"""Efficient Hessian vector products."""

from typing import Callable, List, Text, Tuple, Union

import tensor_list_util
import tensorflow.compat.v2 as tf

# Shorthand notation: Parameters for which the autograd can compute the gradient
# with respect to are tensors or list of tensors.
Parameters = Union[tf.Tensor, List[tf.Tensor]]


def _hessian_vector_product(
    function: Callable[[Parameters], tf.Tensor],
    parameters: Parameters,
    v: Parameters) -> Parameters:
  """Computes Hessian-vector products.

  Computes the product H.v where v is an arbitrary vector and H is the Hessian
  of a function evaluated at `parameters`.

  The result is the same as if the Hessian was computed explicitly and
  multiplied the vector. However, this function uses the autograd in backward
  then forward mode in order to compute this Hessian vector product without
  having to explicitly compute the Hessian.

  Args:
    function: A (twice) differentiable function that takes as input a tensor or
      a list of tensors and returns a scalar.
    parameters: The parameters with respect to which we want to compute the
      Hessian for the hessian vector product.
    v: An arbitrary vector or list of vectors of the same nested structure as
      `parameters`.

  Returns:
    A vector or list of vectors of the same nested structure as
      `parameters`, equal to H.v.
  """
  with tf.autodiff.ForwardAccumulator(
      primals=parameters, tangents=v) as acc:
    with tf.GradientTape() as tape:
      tape.watch(parameters)
      value = function(parameters)
    backward = tape.gradient(value, parameters)
  return acc.jvp(backward)


def _reduce_function_over_dataset(
    function: Callable[[Tuple[tf.Tensor, tf.Tensor]], Parameters],
    dataset: tf.data.Dataset,
    reduce_op: Text = "MEAN") -> Parameters:
  """Averages or sums f(x) over x in a dataset, for any arbitrary function f.

  Args:
    function: A function that take as input examples sampled from the dataset,
      and return a Tensor or list of Tensors.
    dataset: A dataset that yield the inputs to `function` over which the
      outputs of `function` should be averaged or summed.
    reduce_op: Whether to average over the dataset (if set to `MEAN`) or
      to simply sum the output tensors (if set to `SUM`).

  Returns:
    Output of `function` averaged or summed over the dataset.
  """
  assert reduce_op in ["MEAN", "SUM"]
  dataset = iter(dataset)
  # We loose a bit of generality by assuming that the dataset yield tuple of
  # tensors instead of anything that the function can take as input, only to
  # be able to get the batch size. Fine for now, maybe change later if this ever
  # becomes a restriction.
  x, y = next(dataset)
  acc = function((x, y))
  acc = [acc] if not isinstance(acc, list) else acc
  accumulated_obs = x.shape[0]
  for x, y in dataset:
    new_val = function((x, y))
    new_obs = x.shape[0]
    w_old = accumulated_obs / (accumulated_obs + new_obs)
    w_new = new_obs / (accumulated_obs + new_obs)
    new_val = [new_val] if not isinstance(new_val, list) else new_val
    for i, value in enumerate(new_val):
      if reduce_op == "SUM":
        acc[i] = acc[i] + value
      else:
        acc[i] = w_old * acc[i] + w_new * value
    accumulated_obs += new_obs
  return acc


def model_hessian_vector_product(
    loss_function: Callable[[tf.keras.Model, Tuple[tf.Tensor, tf.Tensor]],
                            tf.Tensor],
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
    v: tf.Tensor,
    reduce_op: Text = "MEAN") -> tf.Tensor:
  """Computes the product of a model's Hessian with an arbitrary vector.

  The Hessian is defined as the second order derivative of the loss summed (or
  averaged) over the dataset, with respect to the model's parameters.

  Args:
    loss_function: Function that takes as input a model and an (input, output)
      tuple representing a batch of examples, an returns a scalar.
    model: The Keras model for which we want to compute the Hessian.
    dataset: Dataset containing the examples over which the loss should be
      computed.
    v: Arbitrary vector of size [w_dim, 1], where `w_dim` is the number of
      parameters in the model, for which we want to compute the Hessian vector
      product.
    reduce_op: Whether to average the loss value over the dataset (if set to
      `MEAN`) or to simply sum it (if set to `SUM`).

  Returns:
    A vector of size [w_dim, 1], product of the model's Hessian and `v`.
  """
  if reduce_op not in ["MEAN", "SUM"]:
    raise ValueError(
        "`reduce_op` must be in 'MEAN' or 'SUM', but got {}".format(reduce_op))
  v = tensor_list_util.vector_to_tensor_list(v, model.trainable_variables)

  @tf.function
  def loss_hessian_vector_product(inputs):
    return _hessian_vector_product(
        lambda _: loss_function(model, inputs),
        model.trainable_variables,
        v)
  mvp_as_list_of_tensors = _reduce_function_over_dataset(
      loss_hessian_vector_product,
      dataset,
      reduce_op=reduce_op)
  return tensor_list_util.tensor_list_to_vector(mvp_as_list_of_tensors)
