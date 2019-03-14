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

"""VGG-esque models for studying Hessians.

With/without batch normalization, and with no data augmentation, this achieves
about 80% validation accuracy.
"""

import collections
import numpy as np
import tensorflow as tf


HParams = collections.namedtuple(
    'HParams', 'batch_size, num_classes, fc_size, use_batch_norm, use_dropout '
    'weight_decay, momentum, '
    'lr_values, lr_boundaries')


def _conv2d_bn(net, num_filters, is_train, use_batch_norm, index):
  """Convolution, maybe batch normalization and then relu."""
  net = tf.layers.conv2d(
      net, num_filters, [3, 3], [1, 1], padding='same',
      activation=None, name='conv2d_{}'.format(index))

  if use_batch_norm:
    net = tf.layers.batch_normalization(net, training=is_train,
                                        name='bn_{}'.format(index))

  net = tf.nn.relu(net)

  return net


class VGG(object):
  """VGG class in the form of resnet.

  Public API: labels, predictions, graph, train_op, cost, ce_cost.
  """

  def __init__(self, hps, images, labels, num_samples, mode):
    """Builds a new VGG model object (in analogy with resnet code).

    Args:
      hps: Hyperparameters.
      images: A [batch, height, width, channels] float32 tensor.
      labels: An int one-hot [batch, num_classes] tensor.
      num_samples: An int.
      mode: A string, either train or eval, that determines how the model
        behaves.

    Returns:
      A new VGG model object.
    """
    self._hps = hps
    self._images = images
    self.labels = labels
    self._num_samples = num_samples
    self._mode = mode
    self.sync_opt = None

  def build_graph(self):
    """Builds the graph and sets up state."""
    self._global_step_variable = tf.get_or_create_global_step()
    self.global_step = self._global_step_variable.initialized_value()
    self.graph = tf.get_default_graph()
    self._build_model()

    self._build_train_op()
    self.summaries = tf.summary.merge_all()

  def _build_model(self):
    """Builds the model into the default graph.

    This is much like a VGG-a, except that the final output layers are much
    smaller (i.e., we have only one of them, and it's not 4096).
    """
    is_train = (self._mode == 'train')

    net = self._images
    net = _conv2d_bn(net, 64, is_train, self._hps.use_batch_norm, 1)
    net = tf.layers.max_pooling2d(net, [2, 2], 2, padding='valid')

    net = _conv2d_bn(net, 128, is_train, self._hps.use_batch_norm, 2)
    net = tf.layers.max_pooling2d(net, [2, 2], 2, padding='valid')

    net = _conv2d_bn(net, 256, is_train, self._hps.use_batch_norm, 3)
    net = _conv2d_bn(net, 256, is_train, self._hps.use_batch_norm, 4)
    net = tf.layers.max_pooling2d(net, [2, 2], 2, padding='valid')

    net = _conv2d_bn(net, 512, is_train, self._hps.use_batch_norm, 5)
    net = _conv2d_bn(net, 512, is_train, self._hps.use_batch_norm, 6)
    net = tf.layers.max_pooling2d(net, [2, 2], 2, padding='valid')

    net = _conv2d_bn(net, 512, is_train, self._hps.use_batch_norm, 7)
    net = _conv2d_bn(net, 512, is_train, self._hps.use_batch_norm, 8)
    net = tf.layers.max_pooling2d(net, [2, 2], 2, padding='valid')

    net = tf.layers.flatten(net)
    net = tf.layers.dense(net, self._hps.fc_size, activation=tf.nn.relu,
                          name='fc_1')

    if self._hps.use_dropout:
      net = tf.layers.dropout(net, 0.5, training=is_train)

    logits = tf.layers.dense(net, self._hps.num_classes, name='fc_2')

    self.predictions = tf.nn.softmax(logits)
    self.ce_cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(
        labels=self.labels, logits=logits))
    weight_decay = 0.0
    for var in tf.trainable_variables():
      weight_decay += tf.nn.l2_loss(var)

    weight_decay *= self._hps.weight_decay

    self.cost = self.ce_cost + weight_decay

    self.precision = tf.reduce_mean(tf.to_float(tf.equal(
        tf.argmax(self.predictions, axis=1), tf.argmax(self.labels, axis=1))))
    tf.summary.scalar('Accuracy', self.precision)
    tf.summary.scalar('Weight_decay', weight_decay)
    tf.summary.scalar('cross_entropy_loss', self.ce_cost)
    tf.summary.scalar('cost', self.cost)

  def _build_train_op(self):
    """Builds a new train op."""
    boundaries = [int(x) for x in self._hps.lr_boundaries.split(',')]
    values = [float(x) for x in self._hps.lr_values.split(',')]

    assert len(boundaries) + 1 == len(values)

    learning_rate = tf.piecewise_constant(
        self.global_step, boundaries=boundaries, values=values)
    tf.summary.scalar('learning_rate', learning_rate)

    optimizer = tf.MomentumOptimizer(
        learning_rate, self._hps.momentum)

    num_parameters = sum(
        np.prod(x.shape.as_list()) for x in tf.trainable_variables())
    tf.logging.info(
        'Number of trainable parameters %i.' % num_parameters)

    self.train_op = tf.contrib.slim.learning.create_train_op(
        self.cost, optimizer, self._global_step_variable)
