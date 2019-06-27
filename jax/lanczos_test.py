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

"""Tests for Hessian density estimate libraries."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import absltest
from absl.testing import parameterized
from jax import test_util as jtu
from jax.api import jit
from jax.config import config as jax_config
import jax.numpy as np
import jax.random as random
import numpy as onp
from spectral_density import density as density_lib
from spectral_density import lanczos
# TODO(yingxiao): factor out the tests for density_lib and lanczos separately.

jax_config.parse_flags_with_absl()
MATRIX_SHAPES = [(10, 10), (15, 15)]


class LanczosTest(jtu.JaxTestCase):
  # pylint: disable=g-complex-comprehension

  # Test lanczos on small Gaussian matrices.
  # This test is not intended to test the scalability to large matrices or
  # rigorously evaluate the numerical precision of the lanczos algorithm.
  # This first test examines a special case where order=matrix_shape. In this
  # case we expect the tridiagonal matrix to have the exact same eigenvalues
  # as the full matrix. For typical use cases (order << matrix_shape we will not
  # directly use the eigenvales of tridiag to approximate the hessian, instead a
  # kernel density estimate of the true spectrum will be calculated using the
  # Gaussian Quadrature Approximation (https://arxiv.org/pdf/1901.10159.pdf).
  @parameterized.named_parameters(
      jtu.cases_from_list({
          'testcase_name': '_{}'.format(shape),
          'shape': shape
      } for shape in MATRIX_SHAPES))
  def testTridiagEigenvalues(self, shape):
    onp.random.seed(100)
    sigma_squared = 1e-2

    # if order > matrix shape, lanczos may fail due to linear dependence.
    order = min(70, shape[0])

    atol = 1e-5

    key = random.PRNGKey(0)
    matrix = random.normal(key, shape)
    matrix = np.dot(matrix, matrix.T)  # symmetrize the matrix
    mvp = jit(lambda v: np.dot(matrix, v))

    eigs_true, _ = onp.linalg.eigh(matrix)

    @jit
    def get_tridiag(key):
      return lanczos.lanczos_alg(mvp, matrix.shape[0], order, rng_key=key)[0]

    tridiag_matrix = get_tridiag(key)
    eigs_tridiag, _ = onp.linalg.eigh(tridiag_matrix)
    density, grids = density_lib.eigv_to_density(
        onp.expand_dims(eigs_tridiag, 0), sigma_squared=sigma_squared)
    density_true, _ = density_lib.eigv_to_density(
        onp.expand_dims(eigs_true, 0), grids=grids, sigma_squared=sigma_squared)

    self.assertAlmostEqual(np.max(eigs_tridiag), np.max(eigs_true), delta=atol)
    self.assertAlmostEqual(np.min(eigs_tridiag), np.min(eigs_true), delta=atol)
    self.assertArraysAllClose(density, density_true, True, atol=atol)

  @parameterized.named_parameters(
      jtu.cases_from_list({
          'testcase_name': '_{}'.format(shape),
          'shape': shape
      } for shape in MATRIX_SHAPES))
  def testDensity(self, shape):
    # This test is quite similar to previous, but additionally calls the
    # tridiag_to_density function (with 5 independent draws of the lanczos alg).
    # tridiag_to_density will call density_lib.eigv_to_density but will
    # additionally supply the lanczos weighting of the eigenvalues. This is a
    # silly thing to do in this small case where order=dim (in which case the
    # correct weighting is uniform). So in this case the approximation will be
    # worse in comparison to directly using the eigenvalues of the tridiagonal
    # matrix. However, in most applications order << dim, in which case the
    # weighting will be crucial to get a good approximation. However, this unit
    # test is not designed to rigorously test the numerical precision of the
    # lanczos approximation.

    onp.random.seed(100)
    sigma_squared = 1e-2
    num_trials = 5

    # if order > matrix shape, lanczos may fail due to linear dependence.
    order = min(70, shape[0])

    # matrix and num_draws is too small to expect tight agreement in this
    # setting.
    atol = 5e-2

    key = random.PRNGKey(0)
    matrix = random.normal(key, shape)
    matrix = np.dot(matrix, matrix.T)  # symmetrize the matrix
    mvp = jit(lambda v: np.dot(matrix, v))

    eigs_true = onp.linalg.eigvalsh(matrix)
    tridiag_list = []

    @jit
    def get_tridiag(key):
      return lanczos.lanczos_alg(mvp, matrix.shape[0], order, rng_key=key)[0]

    for _ in range(num_trials):
      key, split = random.split(key)
      tridiag = get_tridiag(split)
      tridiag_list.append(tridiag)

    density, grids = density_lib.tridiag_to_density(
        tridiag_list, sigma_squared=sigma_squared)
    density_true, _ = density_lib.eigv_to_density(
        onp.expand_dims(eigs_true, 0), grids=grids, sigma_squared=sigma_squared)

    self.assertAllClose(density, density_true, True, .3)

    # Measure the statistical distance between the two distributions.
    self.assertLess(np.mean(np.abs(density-density_true)), 5e-2)

if __name__ == '__main__':
  absltest.main()
