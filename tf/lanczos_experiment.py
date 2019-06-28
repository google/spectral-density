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
"""The class computes Gaussian Quadrature for the Hessian eigen distribution.

This class computes and saves tridiagonal matrices that yield nodes and weights
of the Gaussian Quadrature for the eigenvalue distribution of the Hessian. For
more information, look up: https://arxiv.org/pdf/1706.06610.pdf section 4.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import os
import sys
import tensorflow as tf

import experiment_utils
import hessian_experiment

class LanczosExperiment(hessian_experiment.HessianExperiment):
  """Calculates Gaussian Quadrature for the Hessian eigenvalues using Lanczos.

  This class computes a series of tridiagonal matrices that are results of
  performing Lanczos procedure on the Hessian with different random starting
  points. We use Hessian-vector products (as computed in HessianExperiment
  class) for this computation.

  WARNING: We assume that all mini-batches are of uniform size.

  Reference: https://arxiv.org/pdf/1706.06610.pdf section 4.
  """

  def __init__(self,
               loss,
               worker,
               num_workers,
               save_path,
               end_of_input,
               lanczos_steps=80,
               num_draws=10,
               normalize_model=False,
               test_mode=False,
               output_address=None):
    """Initializes the LanczosExperiment class.

    This function initializes the parameters required for Lanczos algorithm. In
    addition, through HessianExperiment class, the tensorflow variables/ops
    required for Hessian vector products are initialized/defined.

    Args:
      loss: The loss tensor associated with the model.
      worker: The worker ID. Note that zero corresponds to the chief worker.
      num_workers: Number of workers.
      save_path: Directory where checkpoints are to be saved.
      end_of_input: A function that takes a raw session and an unrelated op,
        that must be run at some point inside its body, and returns whether
        it's the end of a phase.
      lanczos_steps: The number of Lanczos iterations we intend to do per random
        draw. For our experiments, we usually want 80 iterations.
      num_draws: The number of random draws of the initial Lanczos vector we are
        using.
      normalize_model: A boolean indicating if the filters should be normalized.
        This option is relevant for scale invariant models that use Batch-Norm.
      test_mode: A boolean indicating whether we are in the test mode or not. If
        set to True, the numpy seed for the first lanczos vector in draw #i is
        set to i.
      output_address: The file address in which the output of the algorithm is
        saved. If None, nothing is saved.
    """
    super(LanczosExperiment,
          self).__init__(loss, worker, num_workers, save_path, end_of_input,
                         sys.maxsize, normalize_model)
    self._lanczos_steps = lanczos_steps
    self._num_draws = num_draws
    self._curr_lan_step = 0
    self._curr_draw = 0
    self._test_mode = test_mode

    # The dimension of the Hessian.
    self._num_params = 0
    self._beta = 0.0
    # Flag that indicates the beginning of a draw.
    self._new_draw_flag = True
    self._assignment_obj = experiment_utils.AssignmentHelper(
        self._v_dict.values())
    self._num_params = self._assignment_obj.total_num_params()

    # Allocates the appropriate numpy array for the Lanczos calculation.
    if self._worker == 0:
      self._tridiag = np.zeros((self._lanczos_steps, self._lanczos_steps,\
                                self._num_draws), dtype='float64')
      # This corresponds to the Lanczos vectors of the current draw.
      self._lancz_vec = np.zeros((self._num_params, self._lanczos_steps),\
                                 dtype='float64')
    self._file_address = output_address

    # Defines the vectorization operation for self._accumulator_dict.
    self._vectorize = tf.concat([
        tf.reshape(self.accumulator('hessian').final_value(x), [-1])
        for x in self._v_dict.keys()
    ], 0)
    self._vectorize_resize = tf.expand_dims(self._vectorize, 1)

  # We have to create all the ops in the graph before we actually run
  # the train_fn.  The following methods create the necessary lambdas.
  def _build_new_phase_callback(self):
    def new_phase(raw_sess, current_step, started):
      """Wrapper for Lanczos Quadrature Calculation.

      This function is called in the _train_fn function below after the
      auxiliary workers are done. Lanczos iterations are done within this
      function. Intermediate Lanczos vectors are saved here. The current Lanczos
      iteration and the current draw number are recorded here.

      Arguments:
        raw_sess: The raw session.
        current_step: An integer corresponding to the current phase step.
        started: A boolean corresponding to whether the whole algorithm has
          started to run or not.

      Returns:
        A boolean signaling if the operation is done.
      """
      del current_step, started
      # This condition signals that one Lanczos draw is completely done and
      # another one should start.
      if self._curr_lan_step >= self._lanczos_steps:
        self._curr_lan_step = 0
        self._curr_draw += 1
        self._new_draw_flag = True
        self._save_lanczos_matrix()
      # This signals that the whole procedure is done.
      if self._curr_draw >= self._num_draws:
        return False

      if self._new_draw_flag:
        # Initialize the Lanczos vector.
        if self._test_mode:
          np.random.seed(self._curr_draw)
        new_v = np.random.normal(size=(self._num_params,))
        new_v = new_v / np.linalg.norm(new_v)
        self._lancz_vec[:, 0] = new_v
        self._new_draw_flag = False
        # This variable holds the value of the sub-diagonal element of the
        # tridiagonal matrix at the current Lanczos iteration.
        self._beta = 0.0
      else:
        # v is the result of accumulation in float64.
        v = raw_sess.run(self._vectorize_resize)
        assert v.dtype == np.float64
        # This function performes one step Of Lanczos calculation.
        new_v = self._lanczos(v)
        self._curr_lan_step += 1
      # Assigns new_v to self._v_dict.
      self._assignment_obj.assign(new_v.astype(np.float32), raw_sess)
      return True

    return new_phase

  def _save_lanczos_matrix(self):
    """This function saves the output of the Lanczos iterations on disk.

    This function saves the tridiagonal matrices and Lanczos vectors
    to disk after each draw is completed. The tridiagonal matrices are saved as
    a numpy array of size (self._lanczos_steps, self._lanczos_steps,
    self._num_draws). The Lanczos vectors are saved as numpy arrays of size
    (self._num_params, self._lanczos_steps). If self._file_address is None,
    nothing is saved.
    """
    if self._file_address is not None:
      file_name = 'tridiag_%d'%(self._curr_draw)
      file_address = os.path.join(self._file_address, file_name)
      with open(file_address, 'wb') as f:
        np.save(f, self._tridiag)
      file_name = 'lanczos_vec_%d'%(self._curr_draw)
      file_address = os.path.join(self._file_address, file_name)
      with open(file_address, 'wb') as f:
        np.save(f, self._lancz_vec)

  def _lanczos(self, vec):
    """One step of Lanczos Algorithm.

    This function takes in the output of the Hessian-vector product and performs
    one step of Lanczos algorithm. After appropraitely populating the
    tridiagonal matrix and the Lanczos vector matrix, the function returns a
    numpy array for the new Hessian-vector product.

    Args:
      vec: A numpy array of size [self._num_params, 1]

    Returns:
      new_vec: A numpy array of size [self._num_params, 1] for the next step of
      Lanczos.

    Raises:
      ZeroDivisionError: If the Lanczos vectors become linearly dependent.
    """
    m = self._curr_lan_step
    n = self._num_params
    k = self._curr_draw
    beta = self._beta
    if m == 0:
      v_old = 0
    else:
      v_old = self._lancz_vec[:, m - 1].reshape((n, 1))
    v = self._lancz_vec[:, m].reshape((n, 1))
    w = vec - beta * v_old
    alpha = np.dot(w.T, v)
    self._tridiag[m, m, k] = alpha
    w = w - alpha * v
    # Full reorthogonalization with modified GS
    for j in range(m):
      tau = self._lancz_vec[:, j].reshape((n, 1))
      coeff = np.dot(w.T, tau)
      w = w - coeff * tau
    beta = np.linalg.norm(w)
    if beta < 10**-6:
      raise ZeroDivisionError('The value of beta is 0')
    new_vec = w / (beta + 0.0)
    self._beta = beta
    if m + 1 < self._lanczos_steps:
      self._tridiag[m, m + 1, k] = beta
      self._tridiag[m + 1, m, k] = beta
      self._lancz_vec[:, m + 1] = new_vec[:, 0]
    return new_vec
