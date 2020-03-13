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
"""Tests lanczos.lanczos_algorithm."""

from absl.testing import absltest
import lanczos_algorithm
import numpy as np
import tensorflow.compat.v2 as tf
import test_util


# pylint:disable=invalid-name


class LanczosAlgorithmTest(tf.test.TestCase):

  def test_FullOrderRecovery(self):
    """Matrix Q should be recovered by running Lanczos with dim=order."""
    Q = tf.linalg.diag([1.0, 2.0, 3.0])
    def Qv(v):
      return tf.matmul(Q, v)
    V, T = lanczos_algorithm.lanczos_algorithm(Qv, 3, 3)
    Q_lanczos = tf.matmul(tf.matmul(V, T), V, transpose_b=True)
    self.assertAllClose(Q_lanczos, Q, atol=1e-7)

  def test_FullOrderRecoveryOnModel(self):
    """Hessian should be recovered by running Lanczos with order=dim."""
    x = tf.random.uniform((32, 3), minval=-1, maxval=1, seed=0)
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(7, activation='relu'),
        tf.keras.layers.Dense(2),
    ])
    _ = model(x)  # Call first time to initialize the weights
    labels = tf.random.uniform((32, 2), minval=-1, maxval=1, seed=1)

    # Compute Hessian explicitly:
    mse_loss = tf.keras.losses.MeanSquaredError()
    def loss_on_full_dataset(parameters):
      del parameters  # Used implicitly when calling the model.
      return mse_loss(labels, model(x))
    model_hessian = test_util.hessian_as_matrix(
        loss_on_full_dataset, model.trainable_variables)

    # Compute a full rank approximation of the Hessian using Lanczos, that
    # should then be equal to the Hessian.
    def loss_fn(model, inputs):
      x, y = inputs
      preds = model(x)
      return mse_loss(preds, y)
    w_dim = sum((np.prod(w.shape) for w in model.trainable_variables))
    V, T = lanczos_algorithm.approximate_hessian(
        model,
        loss_fn,
        tf.data.Dataset.from_tensor_slices((x, labels)).batch(5),
        order=w_dim)
    model_hessian_lanczos = tf.matmul(tf.matmul(V, T), V, transpose_b=True)
    self.assertAllClose(model_hessian_lanczos, model_hessian)


if __name__ == '__main__':
  tf.enable_v2_behavior()
  absltest.main()
