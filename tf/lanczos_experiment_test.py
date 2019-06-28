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
"""Tests for lanczos_experiment."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import tempfile
import tensorflow as tf
import time

import lanczos_experiment

class LanczosExperimentTest(tf.test.TestCase):

  def _generate_quadratic(self, num_batches=1000, batch_size=2, dim=500):
    std = 1.0 / np.sqrt(dim)
    x = tf.get_variable(
        'x',
        shape=[dim, 1],
        initializer=tf.random_normal_initializer(stddev=std, dtype=tf.float32),
        dtype=tf.float32)
    data = np.random.normal(size=(num_batches * batch_size,
                                  dim)).astype(np.float32)
    std = np.array(np.random.exponential(size=(dim,))).astype(np.float32)
    data = data * std
    full_hessian = np.dot(data.T, data) / (num_batches * batch_size + 0.0)
    ds = tf.data.Dataset.from_tensor_slices(data).batch(batch_size).repeat()
    self._iterator = ds.make_initializable_iterator()
    elem = self._iterator.get_next()

    loss = 0.5 * tf.reduce_sum(tf.matmul(elem, x)**2) / tf.to_float(batch_size)
    temp_dir = tempfile.mkdtemp()
    return x, elem, loss, temp_dir, full_hessian, data

  def test_lanczos_quadratic(self):
    num_batches = 100
    lanczos_steps = 10
    num_draws = 1
    batch_size = 50
    n = 400
    tol = 1.0e-4
    self._counter = 0

    def end_of_input(sess, train_op):
      sess.run(train_op)
      self._counter += 1
      end_of_input = (self._counter >= num_batches)
      if end_of_input:
        self._counter = 0
      return end_of_input

    with self.test_session() as sess:
      x, _, loss, temp_dir, exact_hessian, _ = \
      self._generate_quadratic(num_batches, batch_size, n)
      x_saver = tf.train.Saver([x])
      experiment = lanczos_experiment.LanczosExperiment(
          loss,
          0,
          1,
          temp_dir,
          end_of_input,
          lanczos_steps,
          num_draws,
          normalize_model=False,
          test_mode=True)
      # We'll populate this checkpoint with x_saver below.
      experiment.get_saver(os.path.join(temp_dir, 'x-0'))
      init_fn = experiment.get_init_fn()
      train_fn = experiment.get_train_fn()
      # The usual sequence is init_op and then init_fn.
      sess.run(tf.global_variables_initializer())
      sess.run(self._iterator.initializer)
      x_saver.save(sess, os.path.join(temp_dir, 'x'), global_step=0)
      init_fn(sess)
      while True:
        try:
          train_fn(sess, None, None)
        except tf.errors.OutOfRangeError:
          tf.logging.info('Quadratic Computation is Done')
          tf.logging.info('Number of Draws Calculated: %d' % (
              experiment._curr_draw))
          tf.logging.info('Current Lanczos Steps: %d' % (
              experiment._curr_lan_step))
          break

    tridiag, lanczos_vecs = self._lanczos(exact_hessian, lanczos_steps, 0)
    tf.logging.info('Summary of Discrepancy:')
    tf.logging.info('ell_1 error tri-diagonal: %f' % (np.sum(np.abs(
        tridiag - experiment._tridiag[:, :, 0]))))
    tf.logging.info('ell_1 error Lanczos vectors:')
    tf.logging.info(np.sum(np.abs(lanczos_vecs - experiment._lancz_vec),
                           axis=0))
    self.assertAllClose(tridiag, experiment._tridiag[:, :, 0], atol=tol)
    self.assertAllClose(lanczos_vecs, experiment._lancz_vec, atol=tol)

  def _generate_cubic(self, num_batches=1000, batch_size=2):
    dims = [19, 100, 100]
    p = np.sum(dims)
    num_vars = 3
    std = 1.0 / np.sqrt(100)
    var_list = [None] * num_vars
    vals = [None] * num_vars
    n = num_batches * batch_size + 0.0
    for i in range(num_vars):
      vals[i] = np.random.normal(size=(dims[i], 1)).astype(np.float32)
      var_list[i] = tf.get_variable(
          'x_' + str(i), initializer=vals[i], dtype=tf.float32)
    data = np.random.normal(size=(int(n), 100)).astype(np.float32)
    std = np.array(np.random.exponential(size=(100,))).astype(np.float32)
    std = std + 0.0001
    data = data * (std**0.5)

    full_hessian = np.zeros((p, p), dtype='float32')
    # ranges = [(0, 19), (19, 119), (119, 219)]
    a = np.random.normal(size=(dims[0], 1)).astype(np.float32)
    b = np.random.normal(size=(dims[0], 1)).astype(np.float32)
    zeta = np.dot(data.T, data) / n
    full_hessian[19:119, 19:119] = np.dot(a.T, vals[0]) * zeta
    full_hessian[119:, 119:] = np.dot(b.T, vals[0]) * zeta
    full_hessian[:19, 19:119] = 2 * np.dot(a, np.dot(zeta, vals[1]).T)
    full_hessian[:19, 119:219] = 2 * np.dot(b, np.dot(zeta, vals[2]).T)
    full_hessian = full_hessian + full_hessian.T

    ds = tf.data.Dataset.from_tensor_slices(data).batch(batch_size).repeat()
    self._iterator = ds.make_initializable_iterator()
    elem = self._iterator.get_next()
    # Loss is: Ave (a^T * var_0)(x^T * var_1)^ 2 + (b^T * var_0)(x^T * var_2)^ 2
    loss = 0
    loss += tf.matmul(a.T, var_list[0]) * tf.reduce_mean(
        tf.matmul(elem, var_list[1])**2)
    loss += tf.matmul(b.T, var_list[0]) * tf.reduce_mean(
        tf.matmul(elem, var_list[2])**2)
    temp_dir = tempfile.mkdtemp()
    return var_list, elem, loss, temp_dir, full_hessian, data

  def test_lanczos_cubic(self):
    num_batches = 1000
    lanczos_steps = 10
    num_draws = 1
    batch_size = 50
    tol = 1.0e-4
    self._counter = 0

    def end_of_input(sess, train_op):
      sess.run(train_op)
      self._counter += 1
      end_of_input = (self._counter >= num_batches)
      if end_of_input:
        self._counter = 0
      return end_of_input

    with self.test_session() as sess:
      _, _, loss, temp_dir, exact_hessian, _ = \
      self._generate_cubic(num_batches, batch_size)
      x_saver = tf.train.Saver(tf.global_variables())
      experiment = \
      lanczos_experiment.LanczosExperiment(loss, 0, 1, temp_dir, end_of_input,
                                           lanczos_steps, num_draws,
                                           normalize_model=False,
                                           test_mode=True)
      # We'll populate this checkpoint with x_saver below.
      experiment.get_saver(os.path.join(temp_dir, 'x-0'))
      init_fn = experiment.get_init_fn()
      train_fn = experiment.get_train_fn()
      # The usual sequence is init_op and then init_fn.
      sess.run(tf.global_variables_initializer())
      sess.run(self._iterator.initializer)
      x_saver.save(sess, os.path.join(temp_dir, 'x'), global_step=0)
      init_fn(sess)
      t0 = time.time()
      while True:
        try:
          train_fn(sess, None, None)
        except tf.errors.OutOfRangeError:
          tf.logging.info('Cubic Computation is Done')
          tf.logging.info('Number of Draws Calculated: %d' % (
              experiment._curr_draw))
          tf.logging.info('Current Lanczos Steps: %d' % (
              experiment._curr_lan_step))
          break
      t1 = time.time()
    tf.logging.info('Time taken for the cubic experimentin TF: %f' % (t1 - t0))

    exact_hessian = exact_hessian.astype(np.float32)
    tridiag, lanczos_vecs = self._lanczos(exact_hessian, lanczos_steps, 0)
    tf.logging.info('Summary of Discrepancy:')
    tf.logging.info('ell_1 error tri-diagonal: %f' % (np.sum(np.abs(
        tridiag - experiment._tridiag[:, :, 0]))))
    tf.logging.info('ell_1 error Lanczos vectors:')
    tf.logging.info(np.sum(np.abs(lanczos_vecs - experiment._lancz_vec),
                           axis=0))
    self.assertAllClose(tridiag, experiment._tridiag[:, :, 0], atol=tol)
    self.assertAllClose(lanczos_vecs, experiment._lancz_vec, atol=tol)

  def _lanczos(self, hessian, order, seed=None):
    """Lanczos algorithm for testing purposes.

    This function does Lanczos algorithm of the given order. hessian is a
    symmetric real matrix. This function does full reorthogonalization.

    Args:
      hessian: A symmetric real positive definite matrix coded as a numpy array.
      order: An integer corresponding to the number of Lanczos steps to take.
      seed: The seed for the initial Lanczos vector.

    Returns:
      tridiag: A tridiagonal matrix of size (order, order).
      vecs: A numpy array of size (n, order) corresponding to the Lanczos
      vectors.

    Raises:
      ZeroDivisionError: If the Lanczos vectors become linearly dependent.
    """
    n = hessian.shape[0]
    tridiag = np.zeros((order, order))
    vecs = np.zeros((n, order), dtype='float32')
    hessian = hessian.astype(np.float32)
    if seed is not None:
      np.random.seed(seed)
    vecs[:, 0] = np.random.normal(size=(n,)).astype(np.float32)
    vecs[:, 0] = vecs[:, 0] / np.linalg.norm(vecs[:, 0])

    beta = 0
    for i in range(order):
      v = vecs[:, i].reshape((n, 1))
      if i == 0:
        v_old = 0
      else:
        v_old = vecs[:, i - 1].reshape((n, 1))
      w = np.dot(hessian, v) - beta * v_old
      alpha = np.dot(w.T, v)
      tridiag[i, i] = alpha
      w = w - alpha * v

      # Full Reorthogonalization
      for j in range(i):
        tau = vecs[:, j].reshape((n, 1))
        coeff = np.dot(w.T, tau)
        w += -coeff * tau

      beta = np.linalg.norm(w)
      if beta < 10**-6:
        raise ZeroDivisionError
      if i + 1 < order:
        tridiag[i, i + 1] = beta
        tridiag[i + 1, i] = beta
        vecs[:, i + 1] = w[:, 0] / beta
    return (tridiag, vecs)

  def test_iterator(self):
    num_batches = 100
    batch_size = 10
    n = 10
    with self.test_session() as sess:
      x, _, loss, _, _, dat = self._generate_quadratic(num_batches, batch_size,
                                                       n)
      # The usual sequence is init_op and then init_fn.
      sess.run(tf.global_variables_initializer())
      sess.run(self._iterator.initializer)
      xval = sess.run(x)
      xval = xval.reshape((n, 1))
      counter = 0
      while counter < 300:
        loss_val = sess.run(loss)
        ind = counter % num_batches
        dat_val = dat[ind * batch_size:(ind + 1) * batch_size, :]
        loss_val_check = np.mean(np.dot(dat_val, xval)**2, axis=0) * 0.5
        loss_val_check = loss_val_check[0]
        self.assertAllClose(loss_val_check, loss_val, atol=0.0001)
        counter += 1


if __name__ == '__main__':
  tf.test.main()
