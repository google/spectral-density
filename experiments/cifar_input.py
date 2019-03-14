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
"""CIFAR tf.data input module.
"""

import sys
import tensorflow as tf


_SPLITS_TO_SIZES = {'train': 50000, 'eval': 10000}
_HEIGHT = 32
_WIDTH = 32
_NUM_CHANNELS = 3
_DEFAULT_IMAGE_BYTES = _HEIGHT * _WIDTH * _NUM_CHANNELS
# The record is the image plus a one-byte label
_RECORD_BYTES = _DEFAULT_IMAGE_BYTES + 1
_NUM_CLASSES = 10
_NUM_DATA_FILES = 5

DATASET_NAME = 'CIFAR-10'


def process_record_dataset(dataset,
                           is_training,
                           batch_size,
                           shuffle_buffer,
                           parse_record_fn,
                           num_epochs=None,
                           repeat_shuffle=False,
                           augment=False):
  """Given a Dataset with raw records, return an iterator over the records.

  Args:
    dataset: A Dataset representing raw records
    is_training: A boolean denoting whether the input is for training.
    batch_size: The number of samples per batch.
    shuffle_buffer: The buffer size to use when shuffling records. A larger
      value results in better randomness, but smaller values reduce startup
      time and use less memory.
    parse_record_fn: A function that takes a raw record and returns the
      corresponding (image, label) pair.
    num_epochs: The number of epochs to repeat the dataset.
    repeat_shuffle: Whether each epoch should be reshuffled.
    augment: Whether to add data augmentation.
  Returns:
    Dataset of (image, label) pairs ready for iteration.
  """
  del shuffle_buffer
  # We prefetch a batch at a time, This can help smooth out the time taken to
  # load input files as we go through shuffling and processing.
  if is_training:
    # Shuffle the records. Note that we shuffle before repeating to ensure
    # that the shuffling respects epoch boundaries.
    dataset = dataset.shuffle(buffer_size=_SPLITS_TO_SIZES['train'],
                              reshuffle_each_iteration=repeat_shuffle,
                              seed=1372)

  # Parse the raw records into images and labels. Testing has shown that setting
  # num_parallel_batches > 1 produces no improvement in throughput, since
  # batch_size is almost always much greater than the number of CPU cores.
  dataset = dataset.apply(
      tf.data.experimental.map_and_batch(
          lambda value: parse_record_fn(value, is_training, augment),
          batch_size=batch_size,
          num_parallel_batches=1,
          drop_remainder=True))
  # If we are training over multiple epochs before evaluating, repeat the
  # dataset for the appropriate number of epochs.
  dataset = dataset.repeat(num_epochs)
  # Operations between the final prefetch and the get_next call to the iterator
  # will happen synchronously during run time. We prefetch here again to
  # background all of the above processing work and keep it out of the
  # critical training path. Setting buffer_size to tf.contrib.data.AUTOTUNE
  # allows DistributionStrategies to adjust how many batches to fetch based
  # on how many devices are present.
  dataset = dataset.prefetch(buffer_size=tf.contrib.data.AUTOTUNE)

  return dataset


def input_fn(is_training, data_path, batch_size, num_epochs=1,
             partition_id=None, num_gpus=None, repeat_shuffle=False,
             augment=False):
  """Input_fn using the tf.data input pipeline for CIFAR-10 dataset.

  Args:
    is_training: A boolean denoting whether the input is for training.
    data_path: Filename for data.
    batch_size: The number of samples per batch.
    num_epochs: The number of epochs to repeat the dataset.
    partition_id: If not None, which partition of data this worker sees.
    num_gpus: The number of gpus used for training.
    repeat_shuffle: Whether each epoch should be reshuffled.
    augment: Whether to add data augmentation.
  Returns:
    A dataset that can be used for iteration.

  Raises:
    ValueError: in case partition_id is provided and num_gpu's not.
  """
  if partition_id is not None and num_gpus is None:
    raise ValueError('Must provide num_gpus if partition_id is provided.')

  filenames = tf.gfile.Glob(data_path)
  dataset = tf.data.FixedLengthRecordDataset(filenames, _RECORD_BYTES)
  dataset.prefetch(batch_size)

  # Partition these by worker now.
  if partition_id is not None:
    dataset = tf.data.Dataset.zip((tf.data.Dataset.range(sys.maxsize),
                                   dataset))
    dataset = dataset.filter(
        lambda x, y: tf.equal(tf.mod(x, num_gpus), partition_id))
    dataset = dataset.map(lambda x, y: y)

  return process_record_dataset(
      dataset=dataset,
      is_training=is_training,
      batch_size=batch_size,
      shuffle_buffer=batch_size * 16,
      parse_record_fn=parse_record,
      num_epochs=num_epochs,
      repeat_shuffle=repeat_shuffle,
      augment=augment)


def parse_record(raw_record, is_training, augment=False):
  """Parse CIFAR-10 image and label from a raw record."""
  # Convert bytes to a vector of uint8 that is record_bytes long.
  record_vector = tf.decode_raw(raw_record, tf.uint8)

  # The first byte represents the label, which we convert from uint8 to int32
  # and then to one-hot.
  label = tf.cast(record_vector[0], tf.int32)

  # The remaining bytes after the label represent the image, which we reshape
  # from [depth * height * width] to [depth, height, width].
  depth_major = tf.reshape(record_vector[1:_RECORD_BYTES],
                           [_NUM_CHANNELS, _HEIGHT, _WIDTH])

  # Convert from [depth, height, width] to [height, width, depth], and cast as
  # float32.
  image = tf.cast(tf.transpose(depth_major, [1, 2, 0]), tf.float32)

  image = preprocess_image(image, is_training, augment=augment)

  return image, label


def preprocess_image(image, is_training, augment=False):
  """Preprocess a single image of layout [height, width, depth]."""
  if is_training and augment:
    # Resize the image to add four extra pixels on each side.
    image = tf.image.resize_image_with_crop_or_pad(
        image, _HEIGHT + 8, _WIDTH + 8)

    # Randomly crop a [_HEIGHT, _WIDTH] section of the image.
    image = tf.random_crop(image, [_HEIGHT, _WIDTH, _NUM_CHANNELS])

    # Randomly flip the image horizontally.
    image = tf.image.random_flip_left_right(image)

  # Subtract off the mean and divide by the variance of the pixels.
  image = tf.image.per_image_standardization(image)
  return image


def build_input(data_path, batch_size, mode, num_epochs=None,
                initializable=False, partition_id=None, num_gpus=None,
                repeat_shuffle=False, augment=False):
  """Build CIFAR image and labels.

  Args:
    data_path: Filename for data.
    batch_size: Input batch size.
    mode: Either 'train' or 'eval'.
    num_epochs: The number of epochs to repeat the dataset.
    initializable: Whether to make an initiazable iterator.
    partition_id: What partition of data to take.
    num_gpus: The number of gpus used for training.
    repeat_shuffle: Whether each epoch should be reshuffled.
    augment: Whether to add data augmentation.
  Returns:
    images: Batches of images. [batch_size, image_size, image_size, 3]
    labels: Batches of labels. [batch_size, num_classes]
    iterator_initializer: An initializer for the iterator if
  Raises:
    ValueError: when the specified dataset is not supported.
  """

  num_samples = _SPLITS_TO_SIZES[mode]

  # Read 'batch' labels + images from the example queue.
  data = input_fn(
      mode == 'train', data_path, batch_size, num_epochs=num_epochs,
      partition_id=partition_id, num_gpus=num_gpus,
      repeat_shuffle=repeat_shuffle,
      augment=augment)
  if initializable:
    iterator = data.make_initializable_iterator()
    initializer = iterator.initializer
  else:
    iterator = data.make_one_shot_iterator()
    initializer = None

  images, labels = iterator.get_next()
  labels = tf.one_hot(labels, _NUM_CLASSES)

  # Display the training images in the visualizer.
  tf.summary.image('images', images)
  return images, labels, num_samples, initializer
