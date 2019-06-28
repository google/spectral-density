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

"""Tests for gradient packer."""


import numpy as np
import tensorflow as tf

import tensor_list_util


class GradientPackerTest(tf.test.TestCase):
  """Tests for packing and unpacking gradients."""

  def _get_variables(self):
    x = tf.get_variable("x", initializer=tf.reshape(
        tf.range(0, 2, dtype=tf.float32), [1, 2]))
    y = tf.get_variable("y", initializer=tf.reshape(
        tf.range(2, 17, dtype=tf.float32), [3, 5]))
    z = tf.get_variable("z", initializer=tf.reshape(
        tf.range(17, 94, dtype=tf.float32), [7, 11]))

    return x, y, z

  def test_vectorize_all(self):
    x, y, z = self._get_variables()
    loss = 0.5 * tensor_list_util.l2_squared([x, y, z])

    gradients = tf.gradients(loss, [x, y, z])
    vectorizer = tensor_list_util.GradientPacker(loss)
    vectorized_gradients = vectorizer.pack(gradients)

    init_op = tf.global_variables_initializer()

    with self.test_session() as sess:
      sess.run(init_op)
      vector = sess.run(vectorized_gradients)

    # By construction, we have that if f(x) = 0.5 * ||x||^2, then f'(x) = x.
    self.assertAllClose(vector, np.reshape(np.arange(0, 94), [1, 94]))

  def test_vectorize_some(self):
    x, y, z = self._get_variables()
    loss = 0.5 * tensor_list_util.l2_squared([x, y])

    # There should a None entry in the gradient.
    gradients = tf.gradients(loss, [x, y, z])
    vectorizer = tensor_list_util.GradientPacker(loss)
    vectorized_gradients = vectorizer.pack(gradients)

    init_op = tf.global_variables_initializer()

    with self.test_session() as sess:
      sess.run(init_op)
      vector = sess.run(vectorized_gradients)

    self.assertAllClose(vector, np.reshape(np.arange(0, 17),
                                           [1, 17]))

  def test_unvectorize_all(self):
    x, y, z = self._get_variables()
    loss = 0.5 * tensor_list_util.l2_squared([x, y, z])

    gradients = tf.gradients(loss, [x, y, z])
    vectorizer = tensor_list_util.GradientPacker(loss)
    vectorized_gradients = vectorizer.pack(gradients)
    unvectorized_gradients = vectorizer.unpack(vectorized_gradients)

    init_op = tf.global_variables_initializer()

    with self.test_session() as sess:
      sess.run(init_op)
      list_of_tensors = sess.run(unvectorized_gradients)
      variables = sess.run([x, y, z])

    for variable, gradient in zip(variables, list_of_tensors):
      self.assertAllClose(variable, gradient)

  def test_unvectorize_some(self):
    x, y, z = self._get_variables()
    loss = 0.5 * tensor_list_util.l2_squared([x, y])

    gradients = tf.gradients(loss, [x, y, z])

    # Dependency inject gradients, so we don't have to duplicate ops.
    vectorizer = tensor_list_util.GradientPacker(loss, gradients=gradients)
    vectorized_gradients = vectorizer.pack(gradients)
    unvectorized_gradients = vectorizer.unpack(vectorized_gradients, full=False)

    init_op = tf.global_variables_initializer()

    with self.test_session() as sess:
      sess.run(init_op)
      list_of_tensors = sess.run(unvectorized_gradients)
      variables = sess.run([x, y])

    for variable, gradient in zip(variables, list_of_tensors):
      self.assertAllClose(variable, gradient)
    self.assertEqual(vectorizer.gradient_size, 17)

  def test_unvectorize_with_gaps(self):
    x, y, z = self._get_variables()
    loss = 0.5 * tensor_list_util.l2_squared([x, y])

    gradients = tf.gradients(loss, [x, y, z])
    vectorizer = tensor_list_util.GradientPacker(loss)

    vectorized_gradients = vectorizer.pack(gradients)
    unvectorized_gradients = vectorizer.unpack(vectorized_gradients, full=True)

    self.assertIsNone(unvectorized_gradients[-1])


if __name__ == "__main__":
  tf.test.main()
