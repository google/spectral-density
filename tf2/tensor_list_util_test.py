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
"""Tests for lanczos.tensor_list_util."""

from absl.testing import absltest
import tensor_list_util
import tensorflow.compat.v2 as tf


class TensorListUtilsTest(tf.test.TestCase):

  def test_TensorListAndVectorConversion(self):
    """Converts list of tensors to vertical vector and backward."""
    tensor_list = [
        tf.constant([[1, 2, 3], [4, 5, 6]]),
        tf.constant([7, 8]),
        tf.constant([[[9], [10]], [[11], [12]], [[13], [14]], [[15], [16]]])
    ]
    as_vector = tensor_list_util.tensor_list_to_vector(tensor_list)
    self.assertEqual(as_vector.shape, [16, 1])
    as_original_list = tensor_list_util.vector_to_tensor_list(
        as_vector, structure=tensor_list)
    self.assertLen(as_original_list, len(tensor_list))
    for a, b in zip(tensor_list, as_original_list):
      self.assertAllEqual(a, b)


if __name__ == '__main__':
  tf.enable_v2_behavior()
  absltest.main()
