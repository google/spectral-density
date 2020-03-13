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
"""Tests lanczos.matrix_vector_product."""


from typing import Text

from absl.testing import absltest
from absl.testing import parameterized
import matrix_vector_product
import test_util
import numpy as np
import tensorflow.compat.v2 as tf


class MatrixVectorProductTest(tf.test.TestCase, parameterized.TestCase):

  def test_HessianVectorProduct(self):
    """Tests the HVP for a simple quadratic function."""
    Q = tf.linalg.diag([1.0, 2.0, 3.0])  # pylint:disable=invalid-name
    def quadratic(x):
      return 0.5 * tf.matmul(x, tf.matmul(Q, x), transpose_a=True)
    x = tf.ones((3, 1))
    v = tf.constant([[0.2], [0.5], [-1.2]])
    # Computes (d2/dx2  1/2 * x^TQx).v = Q.v
    hvp = matrix_vector_product._hessian_vector_product(quadratic, x, v)
    self.assertAllClose(hvp, tf.matmul(Q, v))

  @parameterized.named_parameters(('sum', 'SUM'), ('mean', 'MEAN'))
  def test_ModelHessianOnDataset(self, reduce_op: Text):
    """Tests the HVP for a neural network."""
    x = tf.random.uniform([32, 3], minval=-1, maxval=1, seed=0)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(7, activation='relu'),
        tf.keras.layers.Dense(2),
    ])
    _ = model(x)  # call first time to initialize the weights
    labels = tf.random.uniform((32, 2), minval=-1, maxval=1, seed=1)

    # Computes a reference hessian vector product by computing the hessian
    # explicitly.
    mse_loss = tf.keras.losses.MeanSquaredError(
        reduction=(tf.keras.losses.Reduction.SUM if reduce_op ==
                   'SUM' else tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE))
    def loss_on_full_dataset(parameters):
      del parameters  # Used implicitly when calling the model.
      return mse_loss(labels, model(x))
    model_hessian = test_util.hessian_as_matrix(
        loss_on_full_dataset, model.trainable_variables)
    num_params = sum((np.prod(w.shape) for w in model.trainable_variables))
    v = tf.random.uniform((num_params, 1), minval=-1, maxval=1, seed=2)
    hvp_ref = tf.matmul(model_hessian, v)

    # Compute the same HVP without computing the Hessian
    def loss_fn(model, inputs):
      x, y = inputs
      preds = model(x)
      return mse_loss(preds, y)
    hvp_to_test = matrix_vector_product.model_hessian_vector_product(
        loss_fn,
        model,
        tf.data.Dataset.from_tensor_slices((x, labels)).batch(5),
        v,
        reduce_op=reduce_op)
    self.assertAllClose(hvp_to_test, hvp_ref)


if __name__ == '__main__':
  tf.enable_v2_behavior()
  absltest.main()
