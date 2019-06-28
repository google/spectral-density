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
"""Various utilities for gradient and hessian experiments."""

import collections
import numpy as np
import re
import tensorflow as tf


# List of variables, and a checkpoint to restore them from.
RestoreSpec = collections.namedtuple('RestoreSpec', ['variables', 'checkpoint'])


class AsymmetricSaver(tf.train.Saver):
  """A tf.train.Saver that saves and restores different sets of variables."""

  def __init__(self, save_var_list, restore_specs, *args, **kwargs):
    """Creates a tf.train.Saver that saves and restores different variables.

    Args:
      save_var_list: List of variables to be saved.
      restore_specs: A list of RestoreSpecs specifying what models and
        checkpoints to restore from. We do not assume disjointness in the
        variables, and hence they will be restored in order.
      *args: Other arguments to the underlying tf.train.Saver.
      **kwargs: Other named arguments to the underlying tf.train.Saver.
    """
    super(AsymmetricSaver, self).__init__(save_var_list, *args, **kwargs)

    # The savers have to be built here, because the graph is usually finalized
    # before the restore functions are called.
    self._restore_savers = [
        (tf.train.Saver(var_list=x.variables), x.checkpoint) for x in restore_specs]

  def restore(self, sess, save_path):
    """Restore all variables (both from save_path and restore specs).

    Note that we restore from restore specs, and then from `save_var_list`.

    Args:
      sess: A `Session` in which to restore the parameters.
      save_path: Path to load `save_var_list` from.
    """
    for saver, path in self._restore_savers:
      saver.restore(sess, path)

    super(AsymmetricSaver, self).restore(sess, save_path)

  def get_init_fn(self):
    """Creates an init_fn.

    This is necessary because initialization and save restoration are on
    mutually exclusive paths inside e.g., tf.Supervisor.

    Returns:
      An initialization function to run saver the init_op.
    """
    def _init_fn(sess):
      for saver, path in self._restore_savers:
        saver.restore(sess, path)
    return _init_fn


def get_raw_sess(derived_session):
  """Returns the raw tf.Session in a wrapped (monitored etc) session."""
  if isinstance(derived_session, tf.Session):
    return derived_session
  else:
    return get_raw_sess(
        derived_session._sess)  # pylint:  disable=protected-access


def normalize_all_filters(variables):
  """Normalizes each convolutional filter to have norm 1.

  Some models are scale invariant in that f(tx) = f(x). In these cases, we
  want to fix the norm of the gradient. For Inception V3, the most reasonable
  way is to normalize each linear map (i.e., filter of a convolution).

  This code works with the learning/brain/contrib/slim/examples/Inception model
  but the variable structure means it may not work with other layer functions.

  Args:
    variables: A list of variables that may need to be normalized.

  Returns:
    A list of ops for normalizing all linear/affine maps.
  """
  # First pass: pair variables of the form y = Wx + b.
  regex = '^(.*)(weights|biases)$'
  prefixes = collections.defaultdict(int)

  for variable in variables:
    matches = re.search(regex, variable.op.name)
    if matches is not None:
      prefixes[matches.groups()[0]] += 1

  paired_variables = collections.defaultdict(lambda: {})
  unpaired_variables = []

  for variable in variables:
    matches = re.search(regex, variable.op.name)
    if matches is not None:
      if prefixes[matches.groups()[0]] > 1:
        paired_variables[matches.groups()[0]][matches.groups()[1]] = variable
      else:
        unpaired_variables.append(variable)

  # Create ops to normalize both paired and unpaired variables.
  ops = []
  for variable in unpaired_variables:
    ops.append(
        tf.assign(variable,
                  variable * tf.rsqrt(tf.reduce_sum(tf.square(variable),
                                                    [0, 1, 2]))))

  for variables in paired_variables.values():
    normalizer = tf.rsqrt(tf.reduce_sum(
        tf.square(variables['weights']), [0, 1, 2]))
    ops.extend([
        tf.assign(variables['weights'],
                  variables['weights'] * normalizer),
        tf.assign(variables['biases'],
                  variables['biases'] * normalizer)])
  return ops


class Accumulator(object):
  """Accumulates values for a list of tensors over many mini-batches."""

  def __init__(self, name, list_of_tensors, variable_collections=None,
               keys=None):
    """Create an accumulator for a list of tensors.

    Args:
      name: The variables scope name under which to nest all variables.
      list_of_tensors: A list of tensors to accumulate.
      variable_collections: A list of collections where the created variables
        are added. If None, this defaults to global variables.
      keys: A list of convenient names for the elements of the list of tensors.
        If None, we use the tensor op name.
    """
    if variable_collections is None:
      variable_collections = [tf.GraphKeys.GLOBAL_VARIABLES]

    if keys is None:
      keys = [x.op.name for x in list_of_tensors]

    with tf.variable_scope(name):
      # Everything is ordered to make iteration deterministic.
      self._accumulator_dict = collections.OrderedDict()
      self._counter_dict = collections.OrderedDict()
      self._final_value_dict = collections.OrderedDict()

      self._clear_ops = []
      self._update_ops = []
      self._finalize_ops = []

      for var_name, tensor in zip(keys, list_of_tensors):
        # Early exit for None tensors (i.e., generated by tf.gradients).
        if tensor is None:
          continue

        with tf.variable_scope('counters'):
          counter = tf.get_variable(
              var_name, shape=[],
              initializer=tf.zeros_initializer(tf.float64),
              trainable=False, dtype=tf.float64,
              collections=variable_collections)
          self._counter_dict[var_name] = counter
          self._clear_ops.append(counter.initializer)

        with tf.variable_scope('variables'):
          # Accumulate in doubles: prophylactic measure against
          # floating point error.
          accumulator = tf.get_variable(
              var_name, shape=tensor.shape,
              initializer=tf.zeros_initializer(tf.float64), trainable=False,
              dtype=tf.float64, collections=variable_collections)
          self._accumulator_dict[var_name] = accumulator
          self._clear_ops.append(accumulator.initializer)

        with tf.variable_scope('final_values'):
          final_value = tf.get_variable(
              var_name, shape=tensor.shape,
              initializer=tf.zeros_initializer(tf.float64), trainable=False,
              dtype=tf.float64, collections=variable_collections)
          self._final_value_dict[var_name] = final_value

          with tf.colocate_with(final_value):
            update_final_value = tf.assign(final_value, accumulator / counter)
          self._finalize_ops.append(update_final_value)

        # Note the sequencing of the updates here. This is to prevent the
        # counter update going through even if the value update throws
        # OutOfRangeError.
        with tf.colocate_with(accumulator):
          update_value = tf.assign_add(accumulator, tf.cast(tensor, tf.float64),
                                       use_locking=True)

        with tf.control_dependencies([update_value]):
          with tf.colocate_with(counter):
            update_counter = tf.assign_add(counter, 1.0, use_locking=True)

        self._update_ops.append(update_counter)

  @property
  def finalize_ops(self):
    return self._finalize_ops

  @property
  def clear_ops(self):
    return self._clear_ops

  @property
  def update_ops(self):
    return self._update_ops

  def accumulator_value(self, key):
    return self._accumulator_dict[key]

  def final_value(self, key):
    return self._final_value_dict[key]

  def counter(self, key):
    return self._counter_dict[key]


class AccumulationExperiment(object):
  """Manages the computation for accumulation over batches."""

  def init(self):
    self._saver = None
    self._restart_on_preemption = False
    self._started = None
    self._clear_ops = []
    self._model_normalization_ops = []

  @property
  def saved_variables_collection(self):
    return 'experiment_collection'

  def get_train_op(self):
    """Returns dummy train op. This gets ignored."""
    return tf.no_op(name='dummy_train_op')

  def get_init_fn(self):
    """Returns init function for asymmetric saving and loading."""
    if self._saver is None:
      raise ValueError('Must call get_saver before get_init_fn')

    # This logic needs to be in both the init_fn and the saver since
    # init_fn and saver.restore are mutually exclusive.
    def init_fn(sess):
      saver_init_fn = self._saver.get_init_fn()
      saver_init_fn(sess)
      sess.run(self._model_normalization_ops)

    return init_fn

  def get_saver(self, model_checkpoint_path, other_restore_spec=None):
    """Returns an AsymmetricSaver that also clears and normalizes.

    Args:
      model_checkpoint_path: String denoting where model variables are stored.
      other_restore_spec: An optional list of other restore specs to append
        after the model checkpoint.

    Returns:
      An AsymmetricSaver that can be used in init_fn and train_fn
    """
    # We'd like to use assign_from_checkpoint, but that assumes that
    # you now save all variables into further checkpoints, which is
    # wasteful.

    class _Saver(AsymmetricSaver):
      """AsymmetricSaver that (maybe) clears variables and normalizes."""

      def __init__(
          self, restart_on_preemption, started, clear_ops,
          model_normalization_ops, *args, **kwargs):
        """Constructor for decorated AsymmetricSaver.

        Args:
          restart_on_preemption: Boolean indicating whether to restart
            accumulation on PS preemption.
          started: scalar tensor boolean indicating whether worker 0 has
            previously started.
          clear_ops: An op or list of ops used to rest the experiment.
          model_normalization_ops: An op or list of ops used to normalize
            scale invariance.
          *args: Other args to the saver.
          **kwargs: Other keyword args to the saver.
        """
        super(_Saver, self).__init__(*args, **kwargs)
        self._restart_on_preemption = restart_on_preemption
        self._started = started
        self._clear_ops = clear_ops
        self._model_normalization_ops = model_normalization_ops

        # In GradientExperiment, there are a number of saves without
        # increments of the gradient global step, so we keep our own
        # counter. It is ok that we clobber intermediate checkpoints
        # in case of preemption, as we're really only interested in
        # the final checkpoint.
        self._counter = 0

      def restore(self, sess, save_path):
        """Restores a model.

        Args:
          sess: A raw tensorflow session.
          save_path: A path to restore the experiment variables from.
        """
        super(_Saver, self).restore(sess, save_path)

        # This is unpleasant -- if a PS gets pre-empted, then all FIFO
        # queues/tf data gets reset, which leads to inconsistent data
        # distribution. In this case, we restart everything.

        # Zero out everything if necessary.
        if self._restart_on_preemption and sess.run(self._started):
          sess.run(self._clear_ops)
        sess.run(self._model_normalization_ops)

      def save(self, sess, save_path, *args, **kwargs):
        # Use our janky counter mechanism instead.
        if 'global_step' in kwargs:
          del kwargs['global_step']
        super(_Saver, self).save(
            sess, save_path, *args, global_step=self._counter, **kwargs)

        self._counter += 1

    # Save this for use later: we need to directly call save from
    # the chief after norm computations are done.
    restore_specs = [
        RestoreSpec(
            self._model_variables, model_checkpoint_path)]
    if other_restore_spec is not None:
      restore_specs += other_restore_spec

    self._saver = _Saver(
        self._restart_on_preemption,
        self._started,
        self._clear_ops,
        self._model_normalization_ops,
        tf.get_collection(self.saved_variables_collection),
        restore_specs=restore_specs)
    return self._saver

  def get_train_fn(self):
    raise NotImplementedError('Please implement in a subclass.')


class AssignmentHelper(object):
  """Helper for assigning variables between python and TensorFlow."""

  def __init__(self, variables_list):
    """Constructor for assignment helper.

    Args:
      variables_list: A list of tf.Variable that we want to assign to.
    """
    self._variables_list = variables_list

    # Ops and functions for assigning to model variables.
    self._assign_ops = []
    self._assign_feeds = []
    for var in self._variables_list:
      zeros = tf.zeros_like(var)
      self._assign_ops.append(tf.assign(var, zeros))
      self._assign_feeds.append(zeros)

    self._component_shapes = [
        x.shape.as_list() for x in self._variables_list]
    self._component_sizes = np.cumsum([
        np.prod(x) for x in self._component_shapes])

  # Utilities for packing/unpacking and converting to numpy.
  @staticmethod
  def _pack(x):
    """Converts a list of np.array into a single vector."""
    return np.concatenate([np.reshape(y, [-1]) for y in x]).astype(
        np.float64)

  def _unpack(self, x):
    """Converts a vector into a list of np.array, according to schema."""
    shapes_and_slices = zip(self._component_shapes,
                            np.split(x, self._component_sizes[:-1]))

    return [np.reshape(y, s).astype(np.float32)
            for s, y in shapes_and_slices]

  def assign(self, x, sess):
    """Assigns vectorized np.array to tensorflow variables."""
    assign_values = self._unpack(x)

    sess.run(self._assign_ops, feed_dict=dict(zip(self._assign_feeds,
                                                  assign_values)))

  def retrieve(self, sess):
    """Retrieves tensorflow variables to single numpy vector."""
    values = sess.run(self._variables_list)
    return AssignmentHelper._pack(values)

  def total_num_params(self):
    """Returns the total number of parameters in the model."""
    return self._component_sizes[-1]
