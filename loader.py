import tensorflow as tf
import numpy as np

image_feature_description = {
    'frame_one': tf.io.FixedLenFeature([], tf.string),
    'frame_two': tf.io.FixedLenFeature([], tf.string),
    'frame_three': tf.io.FixedLenFeature([], tf.string),
    'frame_four': tf.io.FixedLenFeature([], tf.string),
    'plus_one_position': tf.io.FixedLenFeature([3], tf.float32),
    'plus_one_orientation': tf.io.FixedLenFeature([3], tf.float32),
    'plus_two_position': tf.io.FixedLenFeature([3], tf.float32),
    'plus_two_orientation': tf.io.FixedLenFeature([3], tf.float32),
    'plus_three_position': tf.io.FixedLenFeature([3], tf.float32),
    'plus_three_orientation': tf.io.FixedLenFeature([3], tf.float32),
    'speed': tf.io.FixedLenFeature([], tf.float32),
}

def decode_and_process_frame(frame, mirror, training):
    image = tf.image.decode_jpeg(frame, channels=3)
    if training:
        image = tf.image.random_hue(image, 0.08)
        image = tf.image.random_saturation(image, 0.6, 1.6)
        image = tf.image.random_brightness(image, 0.05)
        image = tf.image.random_contrast(image, 0.7, 1.3)
        image = cutout(image, 40, 0)
    if mirror:
        image = tf.image.flip_left_right(image)

    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.per_image_standardization(image)

    return image

def cutout(image, pad_size, replace=0):
  """Apply cutout (https://arxiv.org/abs/1708.04552) to image.
  This operation applies a (2*pad_size x 2*pad_size) mask of zeros to
  a random location within `img`. The pixel values filled in will be of the
  value `replace`. The located where the mask will be applied is randomly
  chosen uniformly over the whole image.
  Args:
    image: An image Tensor of type uint8.
    pad_size: Specifies how big the zero mask that will be generated is that
      is applied to the image. The mask will be of size
      (2*pad_size x 2*pad_size).
    replace: What pixel value to fill in the image in the area that has
      the cutout mask applied to it.
  Returns:
    An image Tensor that is of type uint8.
  """
  image_height = tf.shape(image)[0]
  image_width = tf.shape(image)[1]

  # Sample the center location in the image where the zero mask will be applied.
  cutout_center_height = tf.random.uniform(
      shape=[], minval=0, maxval=image_height,
      dtype=tf.int32)

  cutout_center_width = tf.random.uniform(
      shape=[], minval=0, maxval=image_width,
      dtype=tf.int32)

  lower_pad = tf.maximum(0, cutout_center_height - pad_size)
  upper_pad = tf.maximum(0, image_height - cutout_center_height - pad_size)
  left_pad = tf.maximum(0, cutout_center_width - pad_size)
  right_pad = tf.maximum(0, image_width - cutout_center_width - pad_size)

  cutout_shape = [image_height - (lower_pad + upper_pad),
                  image_width - (left_pad + right_pad)]
  padding_dims = [[lower_pad, upper_pad], [left_pad, right_pad]]
  mask = tf.pad(
      tf.zeros(cutout_shape, dtype=image.dtype),
      padding_dims, constant_values=1)
  mask = tf.expand_dims(mask, -1)
  mask = tf.tile(mask, [1, 1, 3])
  image = tf.where(
      tf.equal(mask, 0),
      tf.ones_like(image, dtype=image.dtype) * replace,
      image)
  return image

@tf.function
def parse_record(tfrecord, training):
    proto = tf.io.parse_single_example(tfrecord, image_feature_description)

    if training:
        mirror = tf.random.uniform([]) < 0.5
    else:
        mirror = False

    frame_one = decode_and_process_frame(proto['frame_one'], mirror, training)

    frame_delta = tf.random.uniform([])
    if not training or frame_delta < 0.33:
        frame_two = decode_and_process_frame(proto['frame_two'], mirror, training)
        position = proto['plus_one_position']
        orienation = proto['plus_one_orientation']
        speed = proto['speed']
    elif frame_delta < 0.67:
        frame_two = decode_and_process_frame(proto['frame_three'], mirror, training)
        position = proto['plus_two_position']
        orienation = proto['plus_two_orientation']
        speed = 2 * proto['speed']
    else:
        frame_two = decode_and_process_frame(proto['frame_four'], mirror, training)
        position = proto['plus_three_position']
        orienation = proto['plus_three_orientation']
        speed = 3 * proto['speed']


    image = tf.concat((frame_one, frame_two), axis=2)

    if mirror:
        position = (1, -1, 1) * position
        orienation = (-1, 1, -1) * orienation

    pose = tf.concat((position, orienation), axis=0)

    if not training or tf.random.uniform([]) < 0.5:
        return {'frames': image}, {'pose': pose, 'speed': [speed]}
    else:
        rev_image = tf.concat((frame_two, frame_one), axis=2)
        rev_pose = -1 * pose
        return {'frames': rev_image}, {'pose': rev_pose, 'speed': [speed]}

def load_tfrecord(filename, batch_size, training):
    dataset = tf.data.TFRecordDataset(filename)

    if training:
        dataset = dataset.shuffle(300000)
    dataset = dataset.map(lambda x: parse_record(x, training), num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset
