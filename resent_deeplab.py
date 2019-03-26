import keras
from keras import backend as K
from keras import regularizers
from keras.layers import (Conv2D, SeparableConv2D)

if 'tensorflow' == K.backend():
    import tensorflow as tf

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)


class BilinearUpsampling2d(keras.engine.Layer):
    def __init__(self, upsampling=(2, 2), output_size=None, data_format=None, **kwargs):
        """Just a simple bilinear upsampling layer for 2d input tensors. Works only with TF.
           Args:
               upsampling: tuple of 2 numbers > 0. The upsampling ratio for h and w
               output_size: used instead of upsampling arg if passed!
        """
        super(BilinearUpsampling2d, self).__init__(**kwargs)
        assert not (upsampling and output_size), 'only use one arguments check your inputs.'
        self.data_format = K.common.normalize_data_format(data_format)
        self.input_spec = keras.engine.InputSpec(ndim=4)
        if output_size:
            # if given output size means maybe no need for upsampling.
            self.output_size = keras.utils.conv_utils.normalize_tuple(output_size, 2, 'output_size')
            self.upsampling = None
        else:
            self.output_size = None
            self.upsampling = keras.utils.conv_utils.normalize_tuple(upsampling, 2, 'upsampling')

    def compute_output_shape(self, input_shape):
        if self.upsampling:
            height = self.upsampling[0] * \
                     input_shape[1] if input_shape[1] is not None else None
            width = self.upsampling[1] * \
                    input_shape[2] if input_shape[2] is not None else None
        else:
            height = self.output_size[0]
            width = self.output_size[1]
        return (input_shape[0],
                height,
                width,
                input_shape[3])

    def call(self, inputs):
        if self.upsampling:
            return K.tf.image.resize_bilinear(inputs, (inputs.shape[1] * self.upsampling[0],
                                                       inputs.shape[2] * self.upsampling[1]),
                                              align_corners=True)
        else:
            return K.tf.image.resize_bilinear(inputs, (self.output_size[0],
                                                       self.output_size[1]),
                                              align_corners=True)

    def get_config(self):
        config = {'upsampling': self.upsampling,
                  'output_size': self.output_size,
                  'data_format': self.data_format}
        base_config = super(BilinearUpsampling2d, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


def resnet(image):
    def separable_conv2d(x, out_filters, kernel_size):
        return SeparableConv2D(filters=out_filters,
                               kernel_size=kernel_size,
                               strides=(1, 1),
                               padding='same',
                               kernel_initializer='he_normal',
                               kernel_regularizer=regularizers.l2(0.0001),
                               use_bias=False)(x)

    def conv2d(x, out_filters, kernel_size):
        return Conv2D(filters=out_filters,
                      kernel_size=kernel_size,
                      strides=(1, 1),
                      padding='same',
                      kernel_initializer='he_normal',
                      kernel_regularizer=regularizers.l2(0.0001),
                      use_bias=False)(x)
