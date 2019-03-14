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
"""Trains an cifar model.

See the README.md file for compilation and running instructions.
"""

import os
import sys
import tensorflow as tf

import cifar_input
import resnet_model
import vgg_model

sys.path.insert(0, os.path.abspath("./core"))
import experiment_utils
import lanczos_experiment


tf.flags.DEFINE_float('learning_rate', 0.0001, 'Initial learning rate.')

tf.flags.DEFINE_boolean('add_shortcut', True,
                        'Whether to add shortcuts in the resnet model.')
# These two flags are ignored for Lanczos computation.
tf.flags.DEFINE_boolean('shuffle_each_epoch', True,
                        'Whether to reshuffle the data for training')
tf.flags.DEFINE_boolean('augment', True,
                        'Whether to use data augmentation for training.')

##################
# Momentum Flags #
##################

tf.flags.DEFINE_float('momentum', 0.9, 'Parameter for Momentum optimizer')

tf.flags.DEFINE_integer('batch_size', 128, 'The number of images in each batch.')

tf.flags.DEFINE_string('master', '',
                       'Name of the TensorFlow master to use.')

tf.flags.DEFINE_string('train_log_dir', '/tmp/cifar/train',
                       'Directory where to write event logs.')

# pylint: disable=line-too-long
tf.flags.DEFINE_string(
    'train_data_path',
    '/tmp/cifar10_data/cifar-10-batches-bin/data_batch*',
    'Filepattern for training data.')
# pylint: enable=line-too-long

tf.flags.DEFINE_integer(
    'save_summaries_secs', 5,
    'The frequency with which summaries are saved, in seconds.')

tf.flags.DEFINE_integer(
    'save_interval_secs', 15,
    'The frequency with which the model is saved, in seconds.')

tf.flags.DEFINE_float(
    'keep_checkpoint_every_n_hours', 0.25,
    'Frequency with which to save permanent checkpoints.')

tf.flags.DEFINE_integer('max_number_of_steps', 10000000,
                        'The maximum number of gradient steps.')

tf.flags.DEFINE_integer('startup_delay_steps', 15,
                        'Number of training steps between replicas startup.')

tf.flags.DEFINE_integer(
    'ps_tasks', 0,
    'The number of parameter servers. If the value is 0, then the parameters '
    'are handled locally by the worker.')

tf.flags.DEFINE_integer(
    'task', 0,
    'The Task ID. This value is used when training with multiple workers to '
    'identify each worker.')

tf.flags.DEFINE_boolean('sync_replicas', False,
                        'Whether to sync gradient updates between workers.')

tf.flags.DEFINE_integer(
    'num_ramp_epochs', 5, 'Number of epochs to ramp up learning rate for '
    'sync replica training.')

tf.flags.DEFINE_integer(
    'replicas_to_aggregate', 1,
    'Number of gradients to collect before updating model '
    'parameters.')

tf.flags.DEFINE_integer('total_num_replicas', 1,
                        'Total number of worker replicas.')

tf.flags.DEFINE_enum('optimizer', 'sgd', ['sgd', 'mom', 'adam'],
                     'What optimizer to use.')

##################
# Lanczos options.
##################

tf.flags.DEFINE_enum(
    'train_mode', 'train', ['train', 'lanczos'],
    'Whether to run the lanczos loop instead of training.')

tf.flags.DEFINE_string(
    'checkpoint_to_load', '',
    'What checkpoint to load the model from.')

tf.flags.DEFINE_bool(
    'partition_data_per_worker', False,
    'Whether to partition the data on a per worker basis.')

tf.flags.DEFINE_integer(
    'num_epochs', -1,
    'If greater than 0, the fixed number of epochs of input.')

tf.flags.DEFINE_integer(
    'lanczos_steps', 80, 'Number of Lanczos iterations to run.')

tf.flags.DEFINE_integer(
    'lanczos_draws', 10, 'Number of Lanczos draws.')

tf.flags.DEFINE_bool(
    'lanczos_test_mode', False,
    'Whether to fix the seeds for the Lanczos iteration')

tf.flags.DEFINE_enum(
    'model', 'resnet', ['resnet', 'vgg'],
    'What model to use.')

tf.flags.DEFINE_integer(
    'vgg_fc_size', 256, 'Size of vgg output layers.')
tf.flags.DEFINE_bool(
    'vgg_use_batch_norm', True, 'Whether to use batch norm in VGG.')
tf.flags.DEFINE_bool(
    'vgg_use_dropout', True, 'Whether to use dropout in VGG.')
tf.flags.DEFINE_float(
    'vgg_weight_decay', 5.0e-5, 'Weight decay coefficient.')
tf.flags.DEFINE_string(
    'vgg_lr_boundaries', '20000,100000',
    'Boundaries for piecewise linear learning rate')
tf.flags.DEFINE_string(
    'vgg_lr_values', '0.1,0.01,0.001',
    'Learning rate values for piecewise linear learning rate')

FLAGS = tf.flags.FLAGS


def main(_, build_graph_only=False):
  tf.set_random_seed(1999)
  if not os.path.exists(FLAGS.train_log_dir):
    os.makedirs(FLAGS.train_log_dir)

  if FLAGS.model == 'resnet':
    hps = resnet_model.HParams(
        batch_size=FLAGS.batch_size,
        num_classes=10,
        min_lrn_rate=0.0001,
        lrn_rate=FLAGS.learning_rate,
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
    # If ps_tasks is zero, the local device is used. When using multiple
    # (non-local) replicas, the ReplicaDeviceSetter distributes the variables
    # across the different devices.
    with tf.device(tf.train.replica_device_setter(FLAGS.ps_tasks)):
      if FLAGS.partition_data_per_worker:
        partition_id = FLAGS.task
      else:
        partition_id = None

      if FLAGS.train_mode == 'train':
        images, one_hot_labels, num_samples, init = cifar_input.build_input(
            FLAGS.train_data_path, FLAGS.batch_size, 'train',
            num_epochs=FLAGS.num_epochs, partition_id=partition_id,
            initializable=False,
            num_gpus=FLAGS.total_num_replicas,
            repeat_shuffle=FLAGS.shuffle_each_epoch,
            augment=FLAGS.augment)
        model = model_fn(hps, images, one_hot_labels, num_samples,
                         'train')
        model.build_graph()

        # Training summaries.
        truth = tf.argmax(model.labels, axis=1)
        predictions = tf.argmax(model.predictions, axis=1)
        precision = tf.reduce_mean(tf.to_float(tf.equal(predictions, truth)))

        tf.summary.scalar('Accuracy', precision)

        if FLAGS.sync_replicas:
          startup_delay_steps = 0
        else:
          startup_delay_steps = FLAGS.startup_delay_steps

        # Run training.
        if build_graph_only:
          return

        saver = tf.train.Saver(
            keep_checkpoint_every_n_hours=FLAGS.keep_checkpoint_every_n_hours)
        with model.graph.as_default():
          tf.contrib.slim.learning.train(
              train_op=model.train_op,
              logdir=FLAGS.train_log_dir,
              master=FLAGS.master,
              is_chief=FLAGS.task == 0,
              global_step=tf.train.get_global_step(),
              number_of_steps=FLAGS.max_number_of_steps,
              startup_delay_steps=startup_delay_steps,
              save_summaries_secs=FLAGS.save_summaries_secs,
              save_interval_secs=FLAGS.save_interval_secs,
              saver=saver,
              sync_optimizer=model.sync_opt)
      elif FLAGS.train_mode == 'lanczos':
        images, one_hot_labels, num_samples, init = cifar_input.build_input(
            FLAGS.train_data_path, FLAGS.batch_size, 'train',
            num_epochs=FLAGS.num_epochs, partition_id=partition_id,
            initializable=True,
            num_gpus=FLAGS.total_num_replicas,
            repeat_shuffle=False,
            augment=False)
        model = resnet_model.ResNet(hps, images, one_hot_labels, num_samples,
                                    'train')
        model.build_graph()

        with model.graph.as_default():
          restore_specs = [
              experiment_utils.RestoreSpec(tf.trainable_variables(),
                                           FLAGS.checkpoint_to_load)]

          def end_of_input(sess, train_op):
            try:
              sess.run(train_op)
            except tf.errors.OutOfRangeError:
              sess.run(init)
              return True
            return False

          experiment = lanczos_experiment.LanczosExperiment(
              model.cost, FLAGS.task, FLAGS.total_num_replicas,
              FLAGS.train_log_dir, end_of_input,
              lanczos_steps=FLAGS.lanczos_steps, num_draws=FLAGS.lanczos_draws,
              output_address=FLAGS.train_log_dir,
              test_mode=FLAGS.lanczos_test_mode)

          train_op = experiment.get_train_op()
          saver = experiment.get_saver(FLAGS.checkpoint_to_load, restore_specs)
          init_fn = experiment.get_init_fn()
          train_fn = experiment.get_train_fn()
          local_init_op = tf.group(tf.local_variables_initializer(), init)

          train_step_kwargs = {}

          tf.contrib.slim.learning.train(
              train_op,
              train_step_kwargs=train_step_kwargs,
              train_step_fn=train_fn,
              logdir=FLAGS.train_log_dir,
              master=FLAGS.master,
              is_chief=(FLAGS.task == 0),
              init_fn=init_fn,
              local_init_op=local_init_op,
              saver=saver,
              save_summaries_secs=FLAGS.save_summaries_secs,
              save_interval_secs=FLAGS.save_interval_secs,
              summary_op=None,
              summary_writer=None)
      else:
        raise ValueError('Unknown mode: %s' % FLAGS.train_mode)


if __name__ == '__main__':
  tf.app.run()
