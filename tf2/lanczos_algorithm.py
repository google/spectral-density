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
"""Implementation of the Lanczos algorithm."""

import time
from typing import Any, Callable, Text, Tuple
import warnings

import matrix_vector_product
import numpy as np
import tensorflow.compat.v2 as tf


class DeviceSelector(object):
  """Helper class to select GPU if available."""

  def __init__(self, only_gpu):
    self.default = "GPU" if self.has_gpu() and only_gpu else "CPU"
    self.accelerator = "CPU" if not self.has_gpu() else "GPU"

  def has_gpu(self):
    return bool(tf.config.experimental.list_physical_devices("GPU"))


def lanczos_algorithm(mvp_fn: Callable[[tf.Tensor], tf.Tensor],
                      dim: int,
                      order: int,
                      random_seed: int = 0,
                      only_gpu: bool = True) -> Tuple[tf.Tensor, tf.Tensor]:
  """Estimates an Hermitian matrix by using its product with arbitrary vectors.

  The Lanczos algorithm is described here:
  https://en.wikipedia.org/wiki/Lanczos_algorithm

  Args:
    mvp_fn: Matrix-vector product function. Function that takes as input a
      tensor of shape [`dim`, 1] and returns another tensor of the same shape.
      The returned tensor should be equal to Hv where v is the input vector and
      H is the symmetric matrix to estimate.
    dim: Dimension of the problem (number of columns and rows of the matrix to
      estimate.)
    order: Rank of the approximation to compute. `mvp_fn` will be called `order`
      times.
    random_seed: Random seed used for sampling the initial vector.
    only_gpu: Whether to use available GPUs for both the matrix vector product
      and the orthogonalization (if set to false, CPU will be used for
      orthogonalization). It is recommended to set this parameter to true and
      change it only if a memory error occurs.

  Returns:
    An estimation of the matrix defined by the matrix vector product function
      given. The matrix is returned as a tuple of two tensors (V,T) of shape
      [dim, order] and [order, order], where T is tridiagonal. The approximation
      of the matrix is then A = V T V^*.
  """
  device_selector = DeviceSelector(only_gpu)

  # Lanczos runs on CPU to save accelerator memory. Most of the computational
  # load takes place in the matrix vector function, which is still computed
  # on GPU if available.
  with tf.device(device_selector.default):
    # Runs Lanczos in float64 as numerical stability is an issue and the
    # bottleneck is calling `mvp_fn`.
    float_dtype = tf.float64
    tridiag = tf.Variable(tf.zeros((order, order), dtype=float_dtype))
    vecs = tf.Variable(tf.zeros((dim, order), dtype=float_dtype))
    init_vec = tf.random.uniform(
        (dim, 1), minval=-1, maxval=1, dtype=float_dtype, seed=random_seed)
    init_vec = init_vec / tf.math.reduce_euclidean_norm(init_vec)
    vecs[:, 0:1].assign(init_vec)
    beta = 0
    v_old = tf.zeros((dim, 1), dtype=float_dtype)

    for i in range(order):
      ts = time.time()
      v = vecs[:, i:i+1]
      with tf.device(device_selector.accelerator):
        tss = time.time()
        w = tf.cast(mvp_fn(tf.cast(v, tf.float32)), float_dtype)
        time_mvp = time.time() - tss
      w = w - beta * v_old
      alpha = tf.matmul(w, v, transpose_a=True)
      tridiag[i:i+1, i:i+1].assign(alpha)
      w = w - alpha * v

      # Reorthogonalization
      for j in range(i):
        tau = vecs[:, j:j+1]
        coeff = tf.matmul(w, tau, transpose_a=True)
        w = w - coeff * tau

      beta = tf.math.reduce_euclidean_norm(w)
      if beta < 1e-6:
        warning_msg = ("Possible numerical stability issues in Lanczos: "
                       "got beta = {} in iteration {}".format(beta.numpy(), i))
        warnings.warn(warning_msg)

      if i + 1 < order:
        tridiag[i, i+1].assign(beta)
        tridiag[i+1, i].assign(beta)
        vecs[:, i+1:i+2].assign(w / beta)

      v_old = v

      info = "Iteration {}/{} done in {:.2f}s (MVP: {:.2f}s).".format(
          i, order,
          time.time() - ts, time_mvp)
      print(info)

  return vecs, tridiag


def approximate_hessian(model: tf.keras.Model,
                        loss_function: Callable[[tf.keras.Model, Any],
                                                tf.Tensor],
                        dataset: tf.data.Dataset,
                        order: int,
                        reduce_op: Text = "MEAN",
                        random_seed: int = 0,
                        only_gpu: bool = True) -> Tuple[tf.Tensor, tf.Tensor]:
  """Approximates the Hessian of a model using Lanczos algorithm.

  Will return an approximation of rank `order` as a tuple of vectors and
  tridiagonal matrices (V, T) such that H = V T V^*. The loss will be
  computed on the entire dataset `order` times.

  Args:
    model: The model for which we want to compute the Hessian.
    loss_function: Loss function used to train the model. Takes as input a Keras
      model and a batch (any object yield by iterating on the dataset), and
      returns a scalar.
    dataset: Dataset on which the model is trained.
    order: Rank of the approximation of the Hessian. Setting order to the number
      of parameters recovers the full Hessian, modulo numerical errors.
    reduce_op: Whether the loss function averages or sum the per sample loss.
      Should be "MEAN" or "SUM".
    random_seed: Seed to use to sample the first vector in the Lanczos
      algorithm.
    only_gpu: Whether to use available GPUs for both the model's computation
      and the orthogonalization (if set to false, CPU will be used for
      orthogonalization). It is recommended to set this parameter to true and
      change it only if a memory error occurs.

  Returns:
    A tuple of tensors (V, T) such that H = V T V^* is an approximation of the
      Hessian.
  """
  def hessian_vector_product(v: tf.Tensor):
    return matrix_vector_product.model_hessian_vector_product(
        loss_function, model, dataset, v, reduce_op=reduce_op)

  w_dim = sum((np.prod(w.shape) for w in model.trainable_variables))
  return lanczos_algorithm(
      hessian_vector_product, w_dim, order, random_seed=random_seed,
      only_gpu=only_gpu)
