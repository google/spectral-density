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
"""Evaluates a trained cifar model.

"""

import tensorflow as tf
from tensorflow.contrib import slim

import cifar_input
import resnet_model
import vgg_model


tf.flags.DEFINE_integer('batch_size', 16, 'The number of images in each batch.')

tf.flags.DEFINE_string('master', '',
                       'Name of the TensorFlow master to use.')

tf.flags.DEFINE_string(
    'eval_data_path',
    '/tmp/cifar/data/test_batch.bin',
    'Filepattern for validation data.')

tf.flags.DEFINE_string('checkpoint_dir', '/tmp/cifar/train',
                       'Directory where the model was written to.')

tf.flags.DEFINE_string('eval_dir', '/tmp/cifar/eval',
                       'Directory where the results are saved to.')

tf.flags.DEFINE_integer(
    'eval_interval_secs', 30,
    'The frequency, in seconds, with which evaluation is run.')

tf.flags.DEFINE_enum('optimizer', 'sgd', ['sgd', 'mom', 'adam'],
                     'What optimizer to use.')

tf.flags.DEFINE_float('momentum', 0.9, 'Parameter for Momentum optimizer')

tf.flags.DEFINE_boolean('add_shortcut', True,
                        'Whether to add shortcuts in the resnet model.')

tf.flags.DEFINE_integer(
    'num_ramp_epochs', 5, 'Number of epochs to ramp up learning rate for '
    'sync replica training.')

tf.flags.DEFINE_boolean('sync_replicas', False,
                        'Whether to sync gradient updates between workers.')

tf.flags.DEFINE_integer(
    'replicas_to_aggregate', 1,
    'Number of gradients to collect before updating model '
    'parameters.')

tf.flags.DEFINE_integer('total_num_replicas', 1,
                        'Total number of worker replicas.')

tf.flags.DEFINE_enum(
    'model', 'resnet', ['resnet', 'vgg'],
    'What model to use.')

tf.flags.DEFINE_integer(
    'vgg_fc_size', 256, 'Size of vgg output layers.')
tf.flags.DEFINE_bool(
    'vgg_use_batch_norm', True, 'Whether to use batch norm in VGG.')
tf.flags.DEFINE_bool(
    'vgg_use_dropout', True, 'Whether to use batch norm in VGG.')
tf.flags.DEFINE_float(
    'vgg_weight_decay', 5.0e-5, 'Weight decay coefficient.')
tf.flags.DEFINE_string(
    'vgg_lr_boundaries', '20000,100000',
    'Boundaries for piecewise linear learning rate')
tf.flags.DEFINE_string(
    'vgg_lr_values', '0.1,0.01,0.001',
    'Learning rate values for piecewise linear learning rate')


FLAGS = tf.flags.FLAGS


def main(_):

  if FLAGS.model == 'resnet':
    hps = resnet_model.HParams(
        batch_size=FLAGS.batch_size,
        num_classes=10,
        min_lrn_rate=0.0001,
        lrn_rate=0.0001,
        num_residual_units=5,
        use_bottleneck=False,
        add_shortcut=FLAGS.add_shortcut,
        weight_decay_rate=0.0002,
        relu_leakiness=0.1,
        optimizer=FLAGS.optimizer,
        momentum=FLAGS.momentum,
        num_ramp_epochs=FLAGS.num_ramp_epochs,
        sync_replicas=FLAGS.sync_replicas,
        num_replicas=FLAGS.total_num_replicas,
        replicas_to_aggregate=FLAGS.replicas_to_aggregate)
    model_fn = resnet_model.ResNet
  else:
    hps = vgg_model.HParams(
        batch_size=FLAGS.batch_size,
        num_classes=10,
        fc_size=FLAGS.vgg_fc_size,
        use_batch_norm=FLAGS.vgg_use_batch_norm,
        use_dropout=FLAGS.vgg_use_dropout,
        weight_decay=FLAGS.vgg_weight_decay,
        momentum=FLAGS.momentum,
        lr_values=FLAGS.vgg_lr_values,
        lr_boundaries=FLAGS.vgg_lr_boundaries)
    model_fn = vgg_model.VGG

  g = tf.Graph()
  with g.as_default():
    images, one_hot_labels, num_samples, _ = cifar_input.build_input(
        FLAGS.eval_data_path, FLAGS.batch_size, 'eval', num_epochs=512)
    # Define the model:
    model = model_fn(hps, images, one_hot_labels, num_samples,
                     'eval')
    model.build_graph()

    predictions = tf.argmax(model.predictions, axis=1)
    truth = tf.argmax(model.labels, axis=1)
    precision = tf.reduce_mean(tf.to_float(tf.equal(predictions, truth)))
    summaries = []
    summaries.append(tf.summary.scalar('Accuracy', precision))
    summaries.append(tf.summary.scalar('Cost', model.cost))
    summaries.append(tf.summary.scalar('Cross Entropy', model.ce_cost))

    slim.evaluation.evaluation_loop(
        master='',
        checkpoint_dir=FLAGS.checkpoint_dir,
        logdir=FLAGS.eval_dir,
        num_evals=10,
        summary_op=tf.summary.merge_all(),
        eval_interval_secs=FLAGS.eval_interval_secs)


if __name__ == '__main__':
  tf.app.run()
