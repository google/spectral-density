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

"""Tests for hessian experiment."""

import numpy as np
import os
import tempfile
import tensorflow as tf

import hessian_experiment


def _default_end_of_input(sess, train_op):
  sess.run(train_op)
  return True


class HessianExperimentTest(tf.test.TestCase):

  def _setup_quadratic(self, num_batches=1000, batch_size=2):
    x = tf.get_variable('x', initializer=[2.0, 1.0, 1.0, 0.5])

    # Expected loss is 0.5 * (36 + 4 + 1) = 20.5
    # Expected gradient is
    # Hessian should be diag([9.0, 4.0, 1.0, 0.0]).
    std = np.array([3.0, 2.0, 1.0, 0.0]).astype(np.float32)

    np.random.seed(24601)
    def _generator():
      for _ in xrange(num_batches):
        yield np.random.randn(batch_size, 4).astype(np.float32) * std

    ds = tf.data.Dataset.from_generator(
        _generator, output_types=tf.float32, output_shapes=[batch_size, 4])
    iterator = ds.make_initializable_iterator()
    elem = iterator.get_next()

    # The stochastic optimization problem is 0.5 * E(z^T x)^2
    loss = 0.5 * tf.reduce_sum(tf.square(elem * x)) / tf.to_float(batch_size)
    temp_dir = tempfile.mkdtemp()
    return x, elem, loss, temp_dir, iterator

  def test_variables(self):
    _, _, loss, temp_dir, _ = self._setup_quadratic()

    hessian_experiment.HessianExperiment(
        loss, 0, 2, temp_dir, _default_end_of_input)

    # For hessian, we store a counter, an accumulator and a final value.
    self.assertEqual(3, len(
        [y for y in tf.global_variables() if
         y.op.name.startswith('hessian/')]))

    # One done token per worker.
    self.assertEqual(2, len([y for y in tf.global_variables() if
                             y.op.name.startswith('should_run')]))

  def test_saver(self):
    x, _, loss, temp_dir, iterator = self._setup_quadratic()

    normal_saver = tf.train.Saver([x])
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(iterator.initializer)
      normal_saver.save(sess, os.path.join(temp_dir, 'model.ckpt'),
                        global_step=0)

    experiment = hessian_experiment.HessianExperiment(
        loss, 0, 1, temp_dir, _default_end_of_input)

    # We must make a saver before making an init_fn.
    with self.assertRaises(ValueError):
      dummy_init_fn = experiment.get_init_fn()

    saver = experiment.get_saver(tf.train.latest_checkpoint(temp_dir))
    init_fn = experiment.get_init_fn()

    assign_x = tf.assign(x, [2.0, 2.0, 2.0, 2.0])

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      sess.run(assign_x)

      # Make sure that the x variable has been assigned correctly.
      self.assertAllClose([2.0, 2.0, 2.0, 2.0], sess.run(x))
      init_fn(sess)

      # Test that the init_fn restores the x variable.
      self.assertAllClose([2.0, 1.0, 1.0, 0.5], sess.run(x))

      saver.save(sess, os.path.join(temp_dir, 'model.ckpt'), global_step=24601)
      saver.save(sess, os.path.join(temp_dir, 'model.ckpt'), global_step=24602)
      saver.save(sess, os.path.join(temp_dir, 'model.ckpt'), global_step=24603)

      # Make sure that the saver counter works, and ignores global step.
      self.assertEqual(os.path.join(temp_dir, 'model.ckpt-2'),
                       tf.train.latest_checkpoint(temp_dir))

  def test_single_worker(self):
    num_batches = 1000
    x, _, loss, temp_dir, iterator = self._setup_quadratic(
        num_batches=num_batches)

    x_saver = tf.train.Saver([x])

    def end_of_input(sess, train_op):
      try:
        sess.run(train_op)
      except tf.errors.OutOfRangeError:
        self.assertEqual(sess.run(
            experiment.accumulator('hessian').counter('x')),
                         1000.0)
        sess.run(iterator.initializer)
        return True
      return False

    experiment = hessian_experiment.HessianExperiment(
        loss, 0, 1, temp_dir, end_of_input)

    # We'll populate this checkpoint with x_saver below.
    experiment.get_saver(os.path.join(temp_dir, 'x-0'))
    init_fn = experiment.get_init_fn()
    train_fn = experiment.get_train_fn()

    with self.test_session() as sess:
      # The usual sequence is init_op and then init_fn.
      sess.run(tf.global_variables_initializer())
      sess.run(iterator.initializer)
      x_saver.save(sess, os.path.join(temp_dir, 'x'), global_step=0)
      init_fn(sess)

      for _ in range(num_batches):
        train_fn(sess, None, None)

      # Estimate loss, gradient and hessian ourselves.
      accumulator = np.array(sess.run(
          experiment.accumulator('loss').accumulator_value('loss')))
      weight = np.array(sess.run(
          experiment.accumulator('loss').counter('loss')))
      self.assertAllClose(accumulator / weight, 20.5, atol=0.5)

      accumulator = np.array(sess.run(
          experiment.accumulator('gradient').accumulator_value('x')))
      weight = np.array(sess.run(
          experiment.accumulator('gradient').counter('x')))
      self.assertAllClose(accumulator / weight, [18.0, 4.0, 1.0, 0.0], atol=1.0)

      accumulators = np.array(sess.run(
          experiment.accumulator('hessian').accumulator_value('x')))
      weights = np.array(sess.run(
          experiment.accumulator('hessian').counter('x')))
      self.assertAllClose(accumulators/weights,
                          [9.0, 4.0, 1.0, 0.0], atol=1.0)

      # Runs finalize operations.
      train_fn(sess, None, None)

      train_fn(sess, None, None)
      accumulators = np.array(sess.run(
          experiment.accumulator('hessian').accumulator_value('x')))
      weights = np.array(sess.run(
          experiment.accumulator('hessian').counter('x')))

      # Check whether the new phase has started correctly, and whether we're
      # still able to accumulate accurate values for both hessian vector
      # products and weights.
      self.assertAllClose(np.squeeze(accumulators),
                          [9.0, 4.0, 1.0, 0.0], rtol=3.0)
      self.assertAllClose(np.squeeze(weights), 1.0)

  def test_termination(self):
    num_batches = 8
    x, _, loss, temp_dir, iterator = self._setup_quadratic(
        num_batches=num_batches)

    x_saver = tf.train.Saver([x])

    experiment = hessian_experiment.HessianExperiment(
        loss, 0, 1, temp_dir, _default_end_of_input, max_num_phases=2)

    # We'll populate this checkpoint with x_saver below.
    experiment.get_saver(os.path.join(temp_dir, 'x-0'))
    init_fn = experiment.get_init_fn()
    train_fn = experiment.get_train_fn()

    with self.test_session() as sess:
      # The usual sequence is init_op and then init_fn.
      sess.run(tf.global_variables_initializer())
      sess.run(iterator.initializer)
      x_saver.save(sess, os.path.join(temp_dir, 'x'), global_step=0)
      init_fn(sess)

      train_fn(sess, None, None)
      train_fn(sess, None, None)

      with self.assertRaises(tf.errors.OutOfRangeError):
        train_fn(sess, None, None)

  def test_fisher(self):
    x = tf.get_variable('x', initializer=[4.0, 2.0, 1.0])
    y = tf.get_variable('y', initializer=[1.0, 1.0, 1.0])
    loss = 0.5 * (tf.reduce_sum(tf.square(x)) +
                  tf.reduce_sum(tf.square(y)))
    x_saver = tf.train.Saver([x, y])
    temp_dir = tempfile.mkdtemp()

    experiment = hessian_experiment.HessianExperiment(
        loss, 0, 1, temp_dir, _default_end_of_input, matrix_type='fisher')

    # We'll populate this checkpoint with x_saver below.
    experiment.get_saver(os.path.join(temp_dir, 'x-0'))
    init_fn = experiment.get_init_fn()
    train_fn = experiment.get_train_fn()

    with self.test_session() as sess:
      # The usual sequence is init_op and then init_fn.
      sess.run(tf.global_variables_initializer())
      x_saver.save(sess, os.path.join(temp_dir, 'x'), global_step=0)
      init_fn(sess)

      for _ in range(10):
        train_fn(sess, None, None)

      accumulators = np.array(sess.run(
          experiment.accumulator('fisher').accumulator_value('x')))
      weights = np.array(sess.run(
          experiment.accumulator('fisher').counter('x')))

      train_fn(sess, None, None)
      self.assertAllClose(np.squeeze(accumulators/np.expand_dims(weights, -1)),
                          [40.0, 20.0, 10.0])


if __name__ == '__main__':
  tf.test.main()
