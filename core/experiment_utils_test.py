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
"""Tests for experiment utils."""

import numpy as np
import os
import tempfile
import tensorflow as tf

import experiment_utils


class AsymmetricSaverTest(tf.test.TestCase):
  """Tests for asymmetric saver."""

  def test_save_restore(self):
    x = tf.get_variable('x', [])
    y = tf.get_variable('y', [])

    x_dir = tempfile.mkdtemp()
    y_dir = tempfile.mkdtemp()

    x_checkpoint_base = os.path.join(x_dir, 'model.ckpt')
    y_checkpoint_base = os.path.join(y_dir, 'model.ckpt')

    normal_saver = tf.train.Saver([x, y])

    # Save a checkpoint into y_dir first.
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      normal_saver.save(sess, y_checkpoint_base, global_step=0)

    saver = experiment_utils.AsymmetricSaver(
        [x], [experiment_utils.RestoreSpec(
            [y], os.path.join(y_dir, 'model.ckpt-0'))])

    # Write an x checkpoint.
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      x_initial, y_initial = sess.run([x, y])
      saver.save(sess, x_checkpoint_base)

    # Load using AsymmetricSaver.
    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      saver.restore(sess, tf.train.latest_checkpoint(x_dir))

      x_final, y_final = sess.run([x, y])

    # Make sure that x is loaded correctly from checkpoint, and that y
    # isn't.
    self.assertEqual(x_initial, x_final)
    self.assertNotAllClose(y_initial, y_final)


class FilterNormalizationTest(tf.test.TestCase):

  def test_basic(self):
    u = tf.get_variable('abcdef/weights', shape=[7, 5, 3, 2])
    v = tf.get_variable('abcdef/biases', shape=[2])
    w = tf.get_variable('unpaired/weights', shape=[7, 5, 3, 2])
    x = tf.get_variable('untouched', shape=[])

    normalize_ops = experiment_utils.normalize_all_filters(
        tf.trainable_variables())

    with self.test_session() as sess:
      sess.run(tf.global_variables_initializer())
      u_initial, v_initial, w_initial, x_initial = sess.run([u, v, w, x])
      sess.run(normalize_ops)
      u_final, v_final, w_final, x_final = sess.run([u, v, w, x])

    u_norms = np.sqrt(np.sum(np.square(u_initial), axis=(0, 1, 2)))
    w_norms = np.sqrt(np.sum(np.square(w_initial), axis=(0, 1, 2)))

    # We expect that the abcdef weights are normalized in pairs, that
    # the unpaired weights are normalized on their own, and the
    # untouched weights are in fact untouched.
    self.assertAllClose(np.array(u_final * u_norms), u_initial)
    self.assertAllClose(np.array(v_final * u_norms), v_initial)
    self.assertAllClose(np.array(w_final * w_norms), w_initial)
    self.assertAllClose(x_initial, x_final)


class AssignmentHelperTest(tf.test.TestCase):

  def test_basic(self):
    x = tf.get_variable('x', shape=[2, 3])
    y = tf.get_variable('y', shape=[4])
    tf.get_variable('z', shape=[5, 6])

    helper = experiment_utils.AssignmentHelper([x, y])

    with self.test_session() as sess:
      helper.assign(np.arange(10.0), sess)

      self.assertAllClose(sess.run(x),
                          [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]])
      self.assertAllClose(sess.run(y), [6.0, 7.0, 8.0, 9.0])

      self.assertAllClose(
          helper.retrieve(sess),
          [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0])


if __name__ == '__main__':
  tf.test.main()
