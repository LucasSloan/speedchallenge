import tensorflow as tf
from keras_senet import squeeze_excite_block
from aa_downsample import BlurPool2D

def x_block(input, num_channels, group_width, stride=1):
    if not stride == 1 or not input.shape[-1] == num_channels:
        shortcut = tf.keras.layers.Conv2D(num_channels, 1, strides=stride, padding='same')(input)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)
    else:
        shortcut = input

    # Residual
    res = tf.keras.layers.Conv2D(num_channels, 1, padding='same')(input)
    res = tf.keras.layers.BatchNormalization()(res)
    res = tf.keras.layers.Activation(tf.nn.swish)(res)
    res = tf.keras.layers.Conv2D(num_channels, 3, groups=num_channels//group_width, strides=stride, padding='same')(res)
    res = tf.keras.layers.BatchNormalization()(res)
    res = tf.keras.layers.Activation(tf.nn.swish)(res)
    res = tf.keras.layers.Conv2D(num_channels, 1, padding='same')(res)
    res = tf.keras.layers.BatchNormalization()(res)

    # Merge
    out = tf.keras.layers.add([res, shortcut])
    out = tf.keras.layers.Activation(tf.nn.swish)(out)

    return out

def y_block(input, num_channels, group_width, stride=1):
    if not stride == 1 or not input.shape[-1] == num_channels:
        shortcut = tf.keras.layers.AveragePooling2D()(input)
        shortcut = tf.keras.layers.Conv2D(num_channels, 1, padding='same')(shortcut)
        shortcut = tf.keras.layers.BatchNormalization()(shortcut)
    else:
        shortcut = input

    # Residual
    res = tf.keras.layers.Conv2D(num_channels, 1, padding='same')(input)
    res = tf.keras.layers.BatchNormalization()(res)
    res = tf.keras.layers.Activation(tf.nn.swish)(res)
    res = tf.keras.layers.Conv2D(num_channels, 3, groups=num_channels//group_width, padding='same')(res)
    res = tf.keras.layers.BatchNormalization()(res)
    res = tf.keras.layers.Activation(tf.nn.swish)(res)
    if not stride == 1:
        res = BlurPool2D()(res)
    res = squeeze_excite_block(res, ratio=4)
    res = tf.keras.layers.Conv2D(num_channels, 1, padding='same')(res)
    res = tf.keras.layers.BatchNormalization()(res)

    # Merge
    out = tf.keras.layers.add([res, shortcut])
    out = tf.keras.layers.Activation(tf.nn.swish)(out)

    return out

def regnety_400mf(image):
    encoder_filters = [32, 48, 104, 208, 440]
    stride = 2
    group_width = 8

    conv1 = tf.keras.layers.Conv2D(encoder_filters[0], 3, strides=stride, padding='same')(image)
    conv1 = tf.keras.layers.BatchNormalization()(conv1)
    conv1 = tf.keras.layers.Activation(tf.nn.swish)(conv1)

    conv2 = y_block(conv1, encoder_filters[1], group_width, stride=stride)

    conv3 = y_block(conv2, encoder_filters[2], group_width, stride=stride)
    conv3 = y_block(conv3, encoder_filters[2], group_width)
    conv3 = y_block(conv3, encoder_filters[2], group_width)

    conv4 = y_block(conv3, encoder_filters[3], group_width, stride=stride)
    conv4 = y_block(conv4, encoder_filters[3], group_width)
    conv4 = y_block(conv4, encoder_filters[3], group_width)
    conv4 = y_block(conv4, encoder_filters[3], group_width)
    conv4 = y_block(conv4, encoder_filters[3], group_width)
    conv4 = y_block(conv4, encoder_filters[3], group_width)

    conv5 = y_block(conv4, encoder_filters[4], group_width, stride=stride)
    conv5 = y_block(conv5, encoder_filters[4], group_width)
    conv5 = y_block(conv5, encoder_filters[4], group_width)
    conv5 = y_block(conv5, encoder_filters[4], group_width)
    conv5 = y_block(conv5, encoder_filters[4], group_width)
    conv5 = y_block(conv5, encoder_filters[4], group_width)

    return conv5

def regnetx_200mf(image):
    encoder_filters = [32, 24, 56, 152, 368]
    stride = 2
    group_width = 8

    conv1 = tf.keras.layers.Conv2D(encoder_filters[0], 3, strides=stride, padding='same')(image)
    conv1 = tf.keras.layers.BatchNormalization()(conv1)
    conv1 = tf.keras.layers.Activation(tf.nn.swish)(conv1)

    conv2 = x_block(conv1, encoder_filters[1], group_width, stride=stride)

    conv3 = x_block(conv2, encoder_filters[2], group_width, stride=stride)

    conv4 = x_block(conv3, encoder_filters[3], group_width, stride=stride)
    conv4 = x_block(conv4, encoder_filters[3], group_width)
    conv4 = x_block(conv4, encoder_filters[3], group_width)
    conv4 = x_block(conv4, encoder_filters[3], group_width)

    conv5 = x_block(conv4, encoder_filters[4], group_width, stride=stride)
    conv5 = x_block(conv5, encoder_filters[4], group_width)
    conv5 = x_block(conv5, encoder_filters[4], group_width)
    conv5 = x_block(conv5, encoder_filters[4], group_width)
    conv5 = x_block(conv5, encoder_filters[4], group_width)
    conv5 = x_block(conv5, encoder_filters[4], group_width)
    conv5 = x_block(conv5, encoder_filters[4], group_width)

    return conv5
    
def regnetx_400mf(image):
    encoder_filters = [32, 32, 64, 160, 384]
    stride = 2
    group_width = 16

    conv1 = tf.keras.layers.Conv2D(encoder_filters[0], 3, strides=stride, padding='same')(image)
    conv1 = tf.keras.layers.BatchNormalization()(conv1)
    conv1 = tf.keras.layers.Activation(tf.nn.swish)(conv1)

    conv2 = x_block(conv1, encoder_filters[1], group_width, stride=stride)

    conv3 = x_block(conv2, encoder_filters[2], group_width, stride=stride)
    conv3 = x_block(conv3, encoder_filters[2], group_width)

    conv4 = x_block(conv3, encoder_filters[3], group_width, stride=stride)
    conv4 = x_block(conv4, encoder_filters[3], group_width)
    conv4 = x_block(conv4, encoder_filters[3], group_width)
    conv4 = x_block(conv4, encoder_filters[3], group_width)
    conv4 = x_block(conv4, encoder_filters[3], group_width)
    conv4 = x_block(conv4, encoder_filters[3], group_width)
    conv4 = x_block(conv4, encoder_filters[3], group_width)

    conv5 = x_block(conv4, encoder_filters[4], group_width, stride=stride)
    conv5 = x_block(conv5, encoder_filters[4], group_width)
    conv5 = x_block(conv5, encoder_filters[4], group_width)
    conv5 = x_block(conv5, encoder_filters[4], group_width)
    conv5 = x_block(conv5, encoder_filters[4], group_width)
    conv5 = x_block(conv5, encoder_filters[4], group_width)
    conv5 = x_block(conv5, encoder_filters[4], group_width)
    conv5 = x_block(conv5, encoder_filters[4], group_width)
    conv5 = x_block(conv5, encoder_filters[4], group_width)
    conv5 = x_block(conv5, encoder_filters[4], group_width)
    conv5 = x_block(conv5, encoder_filters[4], group_width)
    conv5 = x_block(conv5, encoder_filters[4], group_width)

    return conv5
    
def regnetx_600mf(image):
    encoder_filters = [32, 48, 96, 240, 528]
    stride = 2
    group_width = 24

    conv1 = tf.keras.layers.Conv2D(encoder_filters[0], 3, strides=stride, padding='same')(image)
    conv1 = tf.keras.layers.BatchNormalization()(conv1)
    conv1 = tf.keras.layers.Activation(tf.nn.swish)(conv1)

    conv2 = x_block(conv1, encoder_filters[1], group_width, stride=stride)

    conv3 = x_block(conv2, encoder_filters[2], group_width, stride=stride)
    conv3 = x_block(conv3, encoder_filters[2], group_width)
    conv3 = x_block(conv3, encoder_filters[2], group_width)

    conv4 = x_block(conv3, encoder_filters[3], group_width, stride=stride)
    conv4 = x_block(conv4, encoder_filters[3], group_width)
    conv4 = x_block(conv4, encoder_filters[3], group_width)
    conv4 = x_block(conv4, encoder_filters[3], group_width)
    conv4 = x_block(conv4, encoder_filters[3], group_width)

    conv5 = x_block(conv4, encoder_filters[4], group_width, stride=stride)
    conv5 = x_block(conv5, encoder_filters[4], group_width)
    conv5 = x_block(conv5, encoder_filters[4], group_width)
    conv5 = x_block(conv5, encoder_filters[4], group_width)
    conv5 = x_block(conv5, encoder_filters[4], group_width)
    conv5 = x_block(conv5, encoder_filters[4], group_width)
    conv5 = x_block(conv5, encoder_filters[4], group_width)

    return conv5