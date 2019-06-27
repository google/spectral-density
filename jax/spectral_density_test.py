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

"""Tests for Hessian density esimate library."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
from absl import flags
from absl.testing import absltest
from jax import test_util as jtu
from jax.config import config as jax_config
from jax.experimental import stax
from jax.experimental.stax import Dense
from jax.experimental.stax import LogSoftmax
from jax.experimental.stax import Relu
from jax.lib.xla_bridge import canonicalize_dtype
import jax.numpy as np
import jax.random as random
import jax.tree_util as tree_util
import numpy as onp
import density as density_lib
import hessian_computation
import lanczos


FLAGS = flags.FLAGS

jax_config.parse_flags_with_absl()


def to_onehot(x, num_class):
  onehot = np.eye(num_class)[x]
  return onehot


def get_batch(input_size, output_size, batch_size, key):
  key, split = random.split(key)
  # jax.random will always generate float32 even if jax_enable_x64==True.
  xs = random.normal(split, shape=(batch_size, input_size),
                     dtype=canonicalize_dtype(onp.float64))
  key, split = random.split(key)
  ys = random.randint(split, minval=0, maxval=output_size, shape=(batch_size,))
  ys = to_onehot(ys, output_size)
  return (xs, ys), key


def prepare_single_layer_model(input_size, output_size, width, key):
  init_random_params, predict = stax.serial(Dense(width), Relu,
                                            Dense(output_size), LogSoftmax)

  key, split = random.split(key)
  _, params = init_random_params(split, (-1, input_size))

  cast = lambda x: x.astype(canonicalize_dtype(onp.float64))
  params = tree_util.tree_map(cast, params)
  return predict, params, key


def loss(y, y_hat):
  return -np.sum(y * y_hat)


class SpectralDensityTest(jtu.JaxTestCase):

  def testHessianVectorProduct(self):
    onp.random.seed(100)
    key = random.PRNGKey(0)
    input_size = 4
    output_size = 2
    width = 10
    batch_size = 5

    # The accuracy of the approximation will be degraded when using lower
    # numberical precision (tpu is float16).
    if FLAGS.jax_test_dut == 'tpu':
      error_tolerance = 1e-4
    else:
      error_tolerance = 1e-6

    predict, params, key = prepare_single_layer_model(input_size,
                                                      output_size, width, key)

    b, key = get_batch(input_size, output_size, batch_size, key)

    def batches():
      yield b
    def loss_fn(params, batch):
      return loss(predict(params, batch[0]), batch[1])

    # isolate the function v -> Hv
    hvp, _, num_params = hessian_computation.get_hvp_fn(loss_fn, params,
                                                        batches)

    # compute the full hessian
    loss_cl = functools.partial(loss_fn, batch=b)
    hessian = hessian_computation.full_hessian(loss_cl, params)

    # test hvp
    v = np.ones((num_params))
    v_hvp = hvp(params, v)

    v_full = np.dot(hessian, v)

    self.assertArraysAllClose(v_hvp, v_full, True, atol=error_tolerance)

  def testHessianSpectrum(self):
    # TODO(gilmer): It appears that tightness of the lanczsos fit can vary.
    # While most time this unit test will pass, I find that on some seeds the
    # test will fail (though the approximation is still reasonable). It would be
    # best to understand the source of this imprecision, (seed 0 will fail for
    # example). It's possible that double precision is required to get really
    # tight estimates of the spectrum.
    onp.random.seed(100)
    key = random.PRNGKey(0)
    input_size = 2
    output_size = 2
    width = 5
    batch_size = 5
    sigma_squared = 1e-2

    if FLAGS.jax_test_dut == 'tpu':
      atol_e = 1e-2
      atol_density = .5
    else:
      atol_e = 1e-6
      atol_density = .5

    predict, params, key = prepare_single_layer_model(input_size, output_size,
                                                      width, key)

    b, key = get_batch(input_size, output_size, batch_size, key)

    def batches():
      yield b
    def loss_fn(params, batch):
      return loss(predict(params, batch[0]), batch[1])

    # isolate the function v -> Hv
    hvp, _, num_params = hessian_computation.get_hvp_fn(loss_fn, params,
                                                        batches)
    hvp_cl = lambda x: hvp(params, x)  # match the API expected by lanczos_alg

    # compute the full hessian
    loss_cl = functools.partial(loss_fn, batch=b)
    hessian = hessian_computation.full_hessian(loss_cl, params)

    def get_tridiag(key):
      return lanczos.lanczos_alg(hvp_cl, num_params, 72, key)
    tridiag = get_tridiag(key)[0]

    eigs_triag, _ = onp.linalg.eigh(tridiag)
    eigs_true, _ = onp.linalg.eigh(hessian)

    density, grids = density_lib.eigv_to_density(
        np.expand_dims(eigs_triag, 0), sigma_squared=sigma_squared)
    density_true, grids = density_lib.eigv_to_density(
        onp.expand_dims(eigs_true, 0), grids=grids, sigma_squared=sigma_squared)

    density = density.astype(canonicalize_dtype(onp.float64))
    density_true = density_true.astype(canonicalize_dtype(onp.float64))
    self.assertAlmostEqual(np.max(eigs_triag), np.max(eigs_true), delta=atol_e)
    self.assertAlmostEqual(np.min(eigs_triag), np.min(eigs_true), delta=atol_e)
    self.assertArraysAllClose(density, density_true, True, atol=atol_density,
                              rtol=1e-1)


if __name__ == '__main__':
  absltest.main()
