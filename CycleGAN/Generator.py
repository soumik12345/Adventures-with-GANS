from tensorflow.keras.layers import (
    Input,
    Conv2D,
    Conv2DTranspose,
    BatchNormalization,
    LeakyReLU,
    Activation,
    concatenate,
    Dropout,
    Cropping2D
)
from tensorflow.keras.models import Model
from tensorflow.keras.initializers import RandomNormal
from Initializers import *


class Generator:

    def __init__(self, n_filters, image_size, image_channels):

        self.image_size = image_size
        self.image_channels = image_channels
        self.n_filters = n_filters
        self.max_filters = 8 * n_filters
        self.build_generator()

    
    def unet_block(
        self,
        input_layer, skipped_layers, input_filters,
        output_filters = None, next_filters = None,
        use_batch_norm = True):
        
        if next_filters is None:
            next_filters = min(2 * input_filters, self.max_filters)
        
        if output_filters is None:
            output_filters = input_filters
        
        x = Conv2D(
            next_filters,
            kernel_size = 4,
            strides = 2,
            padding = 'same',
            use_bias = not (use_batch_norm and skipped_layers > 2),
            name = 'conv_layer_{0}'.format(skipped_layers)
        )(input_layer)

        if use_batch_norm and skipped_layers > 2:
            x_skipped = BatchNormalization()(x, training = 1)
            x_skipped = LeakyReLU(alpha = 0.2)(x_skipped)
            x_skipped = self.unet_block(x_skipped, skipped_layers // 2, next_filters)
            x = concatenate()([x, x_skipped])
        
        x = Activation('relu')(x)
        x = Conv2DTranspose(
            output_filters,
            kernel_size = 4,
            strides = 2,
            use_bias = not use_batch_norm,
            kernel_initializer = RandomNormal(0, 0.02),
            name = 'conv_transpose_{}'.format(skipped_layers),
        )(x)
        x = Cropping2D(1)(x)

        if use_batch_norm:
            x = BatchNormalization()(x, training = 1)
        
        if skipped_layers <= 8:
            x = Dropout(0.5)(x, training = 1)
        
        return x
    
    def build_generator(self):
        
        input_placeholder = Input(
            shape = (
                self.image_size,
                self.image_size,
                self.image_channels
            )
        )

        x = self.unet_block(
            input_placeholder,
            self.image_size,
            self.image_channels,
            self.image_channels,
            self.n_filters,
            False
        )
        output = Activation('tanh')(x)

        self.generator = Model([input_placeholder], output)
    
    def display(self):
        self.generator.summary()
    
    def save(self, model_path):
        self.generator.save(model_path)