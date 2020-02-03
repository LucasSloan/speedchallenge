import tensorflow as tf
import group_conv

def squeeze_excite_block(input, ratio=16):
    channels = input.shape[-1]
    se_shape = (1, 1, input.shape[-1])

    se = tf.keras.layers.GlobalAveragePooling2D()(input)
    se = tf.keras.layers.Reshape(se_shape)(se)
    se = tf.keras.layers.Dense(channels // ratio, activation=tf.nn.swish, kernel_initializer='he_normal', use_bias=False)(se)
    se = tf.keras.layers.Dense(channels, activation=tf.nn.sigmoid, kernel_initializer='he_normal', use_bias=False)(se)

    x = tf.keras.layers.multiply([input, se])
    return x

def res_block(input, num_channels):
    shortcut = input

    # Residual
    res = tf.keras.layers.Conv2D(num_channels, 3, padding='same')(input)
    res = tf.keras.layers.BatchNormalization()(res)
    res = tf.keras.layers.Activation(tf.nn.swish)(res)
    res = tf.keras.layers.Conv2D(num_channels, 3, padding='same')(res)
    res = tf.keras.layers.BatchNormalization()(res)

    # squeeze and excitation
    scaled = squeeze_excite_block(res)

    # Merge
    out = tf.keras.layers.add([scaled, shortcut])
    out = tf.keras.layers.Activation(tf.nn.swish)(out)

    return out

def res_block_first(input, num_channels, stride):
    shortcut = tf.keras.layers.Conv2D(num_channels, 1, strides=stride, padding='same')(input)

    # Residual
    res = tf.keras.layers.Conv2D(num_channels, 3, strides=stride, padding='same')(input)
    res = tf.keras.layers.BatchNormalization()(res)
    res = tf.keras.layers.Activation(tf.nn.swish)(res)
    res = tf.keras.layers.Conv2D(num_channels, 3, padding='same')(res)
    res = tf.keras.layers.BatchNormalization()(res)

    # squeeze and excitation
    scaled = squeeze_excite_block(res)

    # Merge
    out = tf.keras.layers.add([scaled, shortcut])
    out = tf.keras.layers.Activation(tf.nn.swish)(out)

    return out

def resnet34_encoder(image):
    encoder_filters = [64, 64, 128, 256, 512]
    stride = 2

    conv1 = tf.keras.layers.Conv2D(encoder_filters[0], 7, strides=stride, padding='same')(image)
    conv1 = tf.keras.layers.BatchNormalization()(conv1)
    conv1 = tf.keras.layers.Activation(tf.nn.swish)(conv1)
    conv1 = tf.keras.layers.MaxPool2D(3, 2, 'same')(conv1)

    conv2 = res_block(conv1, encoder_filters[0])
    conv2 = res_block(conv2, encoder_filters[0])
    conv2 = res_block(conv2, encoder_filters[0])

    conv3 = res_block_first(conv2, encoder_filters[2], stride)
    conv3 = res_block(conv3, encoder_filters[2])
    conv3 = res_block(conv3, encoder_filters[2])
    conv3 = res_block(conv3, encoder_filters[2])

    conv4 = res_block_first(conv3, encoder_filters[3], stride)
    conv4 = res_block(conv4, encoder_filters[3])
    conv4 = res_block(conv4, encoder_filters[3])
    conv4 = res_block(conv4, encoder_filters[3])
    conv4 = res_block(conv4, encoder_filters[3])
    conv4 = res_block(conv4, encoder_filters[3])

    conv5 = res_block_first(conv4, encoder_filters[4], stride)
    conv5 = res_block(conv5, encoder_filters[4])
    conv5 = res_block(conv5, encoder_filters[4])

    return conv5
