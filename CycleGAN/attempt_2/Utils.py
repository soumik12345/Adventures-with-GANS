from tensorflow.keras.layers import (
    Conv2D,
    LeakyReLU,
    Activation,
    UpSampling2D,
    Dropout,
    Concatenate
)
from tensorflow.keras.models import Model
from Layers import InstanceNormalization


class Utils:

    @staticmethod
    def conv_layer_generator(input_tensor, n_filters, filter_size = 4):
        x = Conv2D(
            n_filters,
            kernel_size = filter_size,
            stides = 1,
            padding = 'same',
            activation = 'relu'
        )(input_tensor)
        x = LeakyReLU(alpha = 0.2)(x)
        x = InstanceNormalization(x)
        return x
    
    @staticmethod
    def deconv_layer_generator(input_tensor, skip_connection, n_filters, filter_size = 4, dropout_rate = 0):
        x = UpSampling2D(size = 2)(input_placeholder)
        x = Conv2D(
            n_filters,
            kernel_size = filter_size,
            stides = 1,
            padding = 'same',
            activation = 'relu'
        )(x)
        x = Activation('relu')(x)
        if dropout_rate != 0:
            x = Dropout(dropout_rate)(x)
        x = InstanceNormalization(x)
        x = Concatenate()([x, skip_connection])
        return x
    

    @staticmethod
    def conv_layer_discriminator(input_placeholder, n_filters, kernel_size = 4, normalization = True):
        x = Conv2D(
            n_filters,
            kernel_size = filter_size,
            stides = 1,
            padding = 'same',
            activation = 'relu'
        )(input_placeholder)
        x = LeakyReLU(alpha = 0.2)(x)
        if normalization:
            x = InstanceNormalization(x)
        return x