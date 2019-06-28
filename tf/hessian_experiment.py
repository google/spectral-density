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
"""Classes to accumulate large/full batch hessians over tensorflow models."""

import collections
import itertools
import os
import sys
import tensorflow as tf

import experiment_utils
import tensor_list_util


class PhasedExperiment(experiment_utils.AccumulationExperiment):
  """An experiment that allows for synchronization of phases.

  This class performs the setup and manages the state necessary for
  running experiments that have phases that proceed in lockstep. See
  HessianExperiment for an example.

  WARNING: We assume that all mini-batches are of uniform size.
  """

  def __init__(self, loss, worker, num_workers,
               save_path, end_of_input, max_num_phases=sys.maxsize,
               normalize_model=False):
    """Constructor for PhasedExperiment class.

    The chief signals to all workers that they should run worker computation;
    they do so until `end_of_input` function triggers. At this point, the
    workers signal to the chief that they are done with a phase. When all
    workers are done in a phase, the chief starts a new phase. When we have
    performed `max_num_phases`, the chief signals termination, and all workers
    and the chief will throw OutOfRangeError.

    Args:
      loss: A scalar tensor. If this is not averaged by mini-batch, then
         set `loss_averaged_by_batch` to False.
      worker: An int denoting which worker this is (often `FLAGS.brain_task`).
      num_workers: An int denoting How many workers there are
        (often `FLAGS.num_train_tasks`).
      save_path: Directory where checkpoints are to be saved.
      end_of_input: A function that takes a raw session and an unrelated op,
        that must be run at some point inside its body, and returns whether
        it's the end of a phase.
      max_num_phases: An optional int denoting how many phases to compute.
      normalize_model: Whether to normalize a scale invariant model (like
        Inception V3).
    """
    self._loss = loss
    self._worker = worker
    self._num_workers = num_workers
    # TODO(yingxiao): Move these to get_saver?
    self._save_path = save_path
    self._end_of_input = end_of_input
    self._max_num_phases = max_num_phases
    self._normalize_model = normalize_model

    self._restart_on_preemption = False

    # We do not save trainable variables, so we use a special collection.
    self._experiment_collections = [
        tf.GraphKeys.GLOBAL_VARIABLES, self.saved_variables_collection]
    self._saver = None

    # List of model (all trainable) variables.
    self._trainable_variables = tf.trainable_variables()

    # Restore all variables created up to this point.
    self._model_variables = (tf.global_variables() +
                             tf.get_collection(tf.GraphKeys.SAVEABLE_OBJECTS))

    # Used to indicate the first iteration.
    self._started = tf.get_variable(
        'first_time', shape=[], dtype=tf.bool, trainable=False,
        initializer=tf.zeros_initializer(dtype=tf.bool),
        collections=self._experiment_collections)
    self._mark_started = tf.assign(self._started, True)

    # Shared memory used to indicate to all workers that the chief is done.
    self._finished = tf.get_variable(
        'finished', shape=[], dtype=tf.bool, trainable=False,
        initializer=tf.zeros_initializer(dtype=tf.bool),
        collections=self._experiment_collections)
    self._mark_finished = tf.assign(self._finished, True)

    # Ops to run at the end of each phase.
    self._finalize_ops = []

    # Tokens used to communicate between the chief and the workers. If
    # should_run_token[i] is 1, then worker i should be running computation.
    self._should_run_tokens = []
    self._clear_run_tokens = []
    self._raise_run_tokens = []

    for i in range(self._num_workers):
      should_run = tf.get_variable(
          'should_run_%i' % i, shape=[], trainable=False,
          dtype=tf.int32, collections=self._experiment_collections,
          initializer=tf.zeros_initializer())
      self._should_run_tokens.append(should_run)
      self._clear_run_tokens.append(tf.assign(should_run, 0))
      self._raise_run_tokens.append(tf.assign(should_run, 1))

    # In case we're pre-empted, we clear all experiment variables.
    # We do not use this in the derived class yet.
    # TODO(yingxiao): Add robustness to preemption.
    self._clear_ops = []

    if self._normalize_model:
      self._model_normalization_ops = experiment_utils.normalize_all_filters(
          self._trainable_variables)
    else:
      self._model_normalization_ops = []

  # We have to create all the ops in the graph before we actually run
  # the train_fn.  The following methods create the necessary lambdas.
  def _build_new_phase_callback(self):
    # Subclasses should combine this saving below with further computation.
    def new_phase(raw_sess, current_step, started):
      del raw_sess, started
      return current_step < self._max_num_phases

    return new_phase

  def get_train_fn(self):
    """Returns a function for training.

    Returns:
      A train_fn that encapsulates worker and chief computation.

    Raises:
      ValueError: In case get_saver hasn't been called yet.
    """
    if self._saver is None:
      raise ValueError('Need to call get_saver first.')

    self._phase_counter = 0
    self._local_step_counter = 0

    # All ops have to be created before entering train_fn.
    train_op = self._get_train_ops()
    new_phase_callback = self._build_new_phase_callback()

    def _train_fn(sess, unused_train_op, unused_global_step,
                  train_step_kwargs=None):
      """Train function run by tf.Supervisor etc."""
      del train_step_kwargs
      # In some instances, we need to avoid running checkpoints,
      # summaries, other hooks etc.
      raw_sess = experiment_utils.get_raw_sess(sess)

      # Global termination condition.
      if raw_sess.run(self._finished):
        raise tf.errors.OutOfRangeError(None, None, 'Worker done.')

      # Mark started.
      if self._worker == 0:
        should_run, started = sess.run([self._should_run_tokens, self._started])

        if not started or not any(should_run):
          tf.logging.info('New phase %i' % self._phase_counter)

          # Run post phase computation.
          raw_sess.run(self._finalize_ops)

          new_phase_status = new_phase_callback(
              raw_sess, self._phase_counter, started)

          # Intermediate saves, and reset the accumulators.
          if started:
            tf.logging.info('Intermediate save %i.' % self._phase_counter)

            self._saver.save(raw_sess, os.path.join(
                self._save_path, 'phased_save'))
          raw_sess.run(self._clear_ops)

          if new_phase_status:
            raw_sess.run(self._mark_started)
            raw_sess.run(self._raise_run_tokens)
          else:
            raw_sess.run(self._mark_finished)
            raise tf.errors.OutOfRangeError(None, None, 'Chief done.')

          self._phase_counter += 1

      # All workers run the following code.
      if raw_sess.run(self._should_run_tokens[self._worker]):
        if self._local_step_counter % 100 == 0:
          tf.logging.info('Local step %i.' % self._local_step_counter)
        end_of_input = self._end_of_input(raw_sess, train_op)
        self._local_step_counter += 1

        if end_of_input:
          raw_sess.run(self._clear_run_tokens[self._worker])
          self._local_step_counter = 0
      return 0.0, False
    return _train_fn


class HessianExperiment(PhasedExperiment):
  """Accumulates hessian vector products over many mini-batches.

  WARNING: We assume that all mini-batches are of uniform size.

  This class performs the setup and manages the state necessary for
  accumulating many mini-batch hessians. This is compatible with both
  the new MonitoredSession, and the older tf.Supervisor, and is meant to
  replace the optimizer and create_train_op idiom (it does not play well with
  the hook paradigm).

  The main change over GradientExperiment is that the computation has phases,
  i.e., this can be used for a Lanczos or Conjugate Gradient algorithm where the
  each phase of hessian vector product corresponds to one matrix-vector product
  or one iteration of the iterative method.

  For example, if we had a model that produced some smooth loss:

    loss = ...

    hessian_experiment = HessianExperiment(
      loss, worker_id, num_workers, save_path, end_of_input_function)

    train_op = hessian_experiment.get_train_op()
    saver = hessian_experiment.get_saver(save_path)
    init_fn = hessian_experiment.get_init_fn()
    train_fn = hessian_experiment.get_train_fn()

    tf.contrib.slim.learning.train(
      train_op, train_step_fn=train_fn, init_fn=init_fn, saver=saver)
  """

  def __init__(self, loss, worker, num_workers,
               save_path, end_of_input, max_num_phases=sys.maxsize,
               normalize_model=False, matrix_type='hessian'):
    """Constructor for HessianExperiment class.

    The chief signals to all workers that they should run hessian computation;
    they do so until `end_of_input` function triggers. At this point, the
    workers signal to the chief that they are done with a phase. When all
    workers are done in a phase, the chief starts a new phase. When we have
    performed `max_num_phases`, the chief signals termination, and all workers
    and the chief will throw OutOfRangeError.

    Args:
      loss: A scalar tensor. If this is not averaged by mini-batch, then
         set `loss_averaged_by_batch` to False.
      worker: An int denoting which worker this is (often `FLAGS.brain_task`).
      num_workers: An int denoting How many workers there are
        (often `FLAGS.num_train_tasks`).
      save_path: Directory where checkpoints are to be saved.
      end_of_input: A function that takes a raw session and an unrelated op,
        that must be run at some point inside its body, and returns whether
        it's the end of a phase.
      max_num_phases: An optional int denoting how many phases to compute.
      normalize_model: Whether to normalize a scale invariant model (like
        Inception V3).
      matrix_type: String denoting what type of matrix product to apply. Choices
        are `hessian` and `fisher`.

    Raises:
      ValueError: in case matrix_type is not `hessian` or `fisher`.
    """
    super(HessianExperiment, self).__init__(
        loss, worker, num_workers, save_path, end_of_input,
        max_num_phases, normalize_model)

    # Create the variables needed for hessian accumulation.
    self._v_dict = collections.OrderedDict()
    self._counter_dict = collections.OrderedDict()
    self._accumulator_dict = collections.OrderedDict()

    # DANGER DANGER: note the dodgy initialization of the v corresponding to the
    # all 1's vector. We expect subclasses to explicitly set v in the train
    # loop.
    with tf.variable_scope('v'):
      for variable in self._trainable_variables:
        v_variable = tf.get_variable(
            variable.op.name, shape=variable.shape,
            initializer=tf.ones_initializer(tf.float32), trainable=False,
            dtype=tf.float32, collections=self._experiment_collections)

        self._v_dict[variable.op.name] = v_variable

    # Build the ops needed for accumulation.
    if matrix_type == 'hessian':
      loss, gradients, matrix_components = self._get_hessian_components()
    elif matrix_type == 'fisher':
      loss, gradients, matrix_components = self._get_fisher_components()
    else:
      raise ValueError('Unknown matrix type: %s' % matrix_type)

    # This is an ordered dict to prevent non-determinism in graph construction.
    self._accumulators = collections.OrderedDict(zip(
        ['loss', 'gradient', matrix_type],
        [
            experiment_utils.Accumulator(
                'loss', [loss],
                variable_collections=self._experiment_collections,
                keys=['loss']),
            experiment_utils.Accumulator(
                'gradients', gradients,
                variable_collections=self._experiment_collections,
                keys=[x.op.name for x in self._trainable_variables]),
            experiment_utils.Accumulator(
                matrix_type, matrix_components,
                variable_collections=self._experiment_collections,
                keys=[x.op.name for x in self._trainable_variables])]))

    self._finalize_ops.extend(list(itertools.chain.from_iterable(
        x.finalize_ops for x in self._accumulators.values())))
    self._clear_ops.extend(list(itertools.chain.from_iterable(
        x.clear_ops for x in self._accumulators.values())))
    self._train_ops = list(itertools.chain.from_iterable(
        x.update_ops for x in self._accumulators.values()))

  def _get_train_ops(self):
    return tf.group(*self._train_ops)

  def _get_hessian_components(self):
    gradients = tf.gradients(self._loss, self._trainable_variables)

    v = [self._v_dict[k.op.name] for k in self._trainable_variables]
    return (self._loss, gradients,
            tf.gradients(gradients, self._trainable_variables, grad_ys=v))

  def _get_fisher_components(self):
    gradients = tf.gradients(self._loss, self._trainable_variables)
    v = [self._v_dict[k.op.name] for k in self._trainable_variables]

    inner_product = tensor_list_util.inner_product(gradients, v)
    return (self._loss, gradients,
            tensor_list_util.scale(inner_product, gradients))

  def accumulator(self, key):
    return self._accumulators[key]
