import tensorflow as tf

class BlurPool2D(tf.keras.layers.Layer):
    def __init__(self, pool_size: int = 2, kernel_size: int = 3, **kwargs):
        self.pool_size = pool_size
        self.blur_kernel = None
        self.kernel_size = kernel_size

        super(BlurPool2D, self).__init__(**kwargs)

    def build(self, input_shape):

        if self.kernel_size == 3:
            bk = tf.constant([[1, 2, 1],
                           [2, 4, 2],
                           [1, 2, 1]], dtype=tf.float32)
            bk = bk / tf.math.reduce_sum(bk)
        elif self.kernel_size == 5:
            bk = tf.constant([[1, 4, 6, 4, 1],
                           [4, 16, 24, 16, 4],
                           [6, 24, 36, 24, 6],
                           [4, 16, 24, 16, 4],
                           [1, 4, 6, 4, 1]], dtype=tf.float32)
            bk = bk / tf.math.reduce_sum(bk)
        else:
            raise ValueError

        bk = tf.repeat(bk, input_shape[3])

        bk = tf.reshape(bk, (self.kernel_size, self.kernel_size, input_shape[3], 1))
        # blur_init = tf.constant_initializer(bk)

        self.blur_kernel = tf.Variable(name='blur_kernel',
                                           initial_value=bk,
                                           trainable=False)

        super(BlurPool2D, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x):
        x = tf.nn.depthwise_conv2d(x, self.blur_kernel, padding='SAME', strides=(1, self.pool_size, self.pool_size, 1))

        return x

    def compute_output_shape(self, input_shape):
        return input_shape[0], int(np.ceil(input_shape[1] / 2)), int(np.ceil(input_shape[2] / 2)), input_shape[3]
