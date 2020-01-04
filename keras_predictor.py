import tensorflow as tf
import loader
import keras_resnet

BATCH_SIZE = 200

training_dataset = loader.load_tfrecord("/mnt/Bulk/commaai/monolithic_train.tfrecord", BATCH_SIZE, True)
validation_dataset = loader.load_tfrecord("/mnt/Bulk/speedchallenge/monolithic_test.tfrecord", BATCH_SIZE, False)

inputs = tf.keras.Input(shape=(128, 416, 6), name='frames')

# encoder
conv5 = keras_resnet.resnet_encoder(inputs)

# thingy
conv6 = tf.keras.layers.Conv2D(512, (3, 3), strides=(2, 2), padding='same', activation=tf.nn.relu)(conv5)
conv7 = tf.keras.layers.Conv2D(512, (3, 3), strides=(2, 2), padding='same', activation=tf.nn.relu)(conv6)

# pose extractor
pose = tf.keras.layers.Conv2D(6, (1, 1), padding='valid')(conv7)
pose = tf.keras.layers.Lambda(lambda x: tf.reduce_mean(x, [1, 2]))(pose)
pose = tf.keras.layers.Reshape((6,))(pose)
pose = tf.keras.layers.Lambda(lambda x: tf.concat([x[:, :3] * 0.01, x[:, 3:6] * 0.001], axis=1), name='pose')(pose)

# speed prediction
speed = tf.keras.layers.Lambda(lambda x: tf.expand_dims(20 * tf.norm(x[:, :3], axis=1), -1), name='speed')(pose)

model = tf.keras.Model(inputs=inputs, outputs=[pose, speed])

model.compile(optimizer=tf.keras.optimizers.Adam(1e-4), loss={'pose': 'mse', 'speed': 'mse'}, loss_weights={'pose': 1.0, 'speed': 0.0})

model.fit(training_dataset, epochs=10, validation_data=validation_dataset, validation_steps=10798//BATCH_SIZE)
