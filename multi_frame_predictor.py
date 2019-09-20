import tensorflow as tf
import time
import numpy as np

image_feature_description = {
    'image_raw': tf.FixedLenFeature([], tf.string),
    'label': tf.FixedLenFeature([], tf.float32),
}

WINDOW_SIZE = 5

def parse_record(tfrecord, training):
    proto = tf.parse_single_example(tfrecord, image_feature_description)

    image = tf.image.decode_jpeg(proto['image_raw'], channels=3)
    image = tf.image.resize(image, (480, 640))
    image = tf.image.crop_to_bounding_box(image, 200, 0, 160, 640)
    if training:
        image = tf.image.random_hue(image, 0.08)
        image = tf.image.random_saturation(image, 0.6, 1.6)
        image = tf.image.random_brightness(image, 0.05)
        image = tf.image.random_contrast(image, 0.7, 1.3)

    image = tf.image.convert_image_dtype(image, tf.float32)

    return image, proto['label']

def load_tfrecord(filename, training):
    raw_dataset = tf.data.TFRecordDataset(filename)

    dataset = raw_dataset.map(lambda x: parse_record(x, training))
    dataset = dataset.apply(tf.contrib.data.sliding_window_batch(WINDOW_SIZE))
    return dataset

def load_dataset(glob_pattern, training):
    files = tf.data.Dataset.list_files(glob_pattern)

    dataset = files.interleave(lambda x: load_tfrecord(x, training), 2000, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    if training:
        dataset = dataset.shuffle(4000)
    dataset = dataset.batch(100)
    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

def residual_block(input, channels, downsample, training):
    shortcut = input
    strides = (1, 1, 1)
    if downsample:
        strides = (1, 2, 2)
        shortcut = tf.layers.conv3d(input, channels, (1, 1, 1), strides=strides, padding='same')
        shortcut = tf.layers.batch_normalization(shortcut, training=training)
        
    conv1 = tf.layers.conv3d(input, channels, (3, 3, 3), strides=(1, 1, 1), padding='same')
    conv1 = tf.layers.batch_normalization(conv1, training=training)
    conv1 = tf.nn.relu(conv1)

    conv2 = tf.layers.conv3d(conv1, channels, (3, 3, 3), strides=strides, padding='same')
    conv2 = tf.layers.batch_normalization(conv2, training=training)


    conv2 += shortcut
    output = tf.nn.relu(conv2)

    return output

training_dataset = load_dataset("D:\\commaai\\segments\\*", True)
validation_dataset = load_dataset("D:\\speedchallenge\\temporal\\*", False)

iterator = tf.data.Iterator.from_structure(training_dataset.output_types,
                                           training_dataset.output_shapes)

training_init_op = iterator.make_initializer(training_dataset)
validation_init_op = iterator.make_initializer(validation_dataset)

# frame is 5 x 640 x 480 x 3 pixels, speed is a float
frames, speeds = iterator.get_next()

training = tf.placeholder(tf.bool)

out = frames
# 5x640x160x3 -> 5x20x5x64
for i in range(5):
    out = residual_block(out, 4 * 2**i, True, training)


out = tf.reshape(out, (-1, WINDOW_SIZE*20*5*64))

dropout_rate = tf.placeholder(tf.float32)
out = tf.layers.dropout(out, rate=dropout_rate)

out = tf.layers.dense(out, WINDOW_SIZE)

loss = tf.losses.mean_squared_error(speeds, out)

train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
train_step = tf.group([train_step, update_ops])

# validation loss is the predicted at the middle frame
speed = speeds[:, WINDOW_SIZE//2]
predicted_speed = out[:, WINDOW_SIZE//2]

validation_loss = tf.losses.mean_squared_error(speed, predicted_speed)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for epoch in range(1, 31):
        sess.run(training_init_op)
        i = 0
        previous_step_time = time.time()
        losses = []
        while True:
            try:
                i += 1
                l, _ = sess.run((loss, train_step), feed_dict={training: True, dropout_rate: 0.4})
                losses.append(l)
                if i % 100 == 0:
                    current_step_time = time.time()
                    time_elapsed = current_step_time - previous_step_time
                    print('epoch {:d} - step {:d} - time {:.2f}s : loss {:.4f}'.format(epoch, i, time_elapsed, np.mean(losses)))
                    previous_step_time = current_step_time
                    losses = []
            except tf.errors.OutOfRangeError:
                break

        sess.run(validation_init_op)
        validation_losses = []
        while True:
            try:
                v_loss = sess.run(validation_loss, feed_dict={training: False, dropout_rate: 0.0})
                validation_losses.append(v_loss)
            except tf.errors.OutOfRangeError:
                break
        print('\n\nmse after {} epochs: {:.4f}\n\n'.format(epoch, np.mean(validation_losses)))
