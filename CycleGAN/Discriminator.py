from tensorflow.keras.layers import (
    Conv2D,
    BatchNormalization,
    Input,
    LeakyReLU,
    ZeroPadding2D
)
from tensorflow.keras.models import Model


class Discriminator:

    def __init__(self, n_filters, max_layers, image_size, image_channels):
        
        self.image_size = image_size
        self.image_channels = image_channels
        self.n_filters = n_filters
        self.max_layers = max_layers

        self.build_model()
    
    def build_model(self):
        
        input_placeholder = Input(
            shape = (
                self.image_size,
                self.image_size,
                self.image_channels
            ),
            name = 'Discriminator_Input_Layer'
        )
        
        x = Conv2D(
            self.n_filters,
            kernel_size = 4,
            strides = 2,
            padding = 'same',
            name = 'First_Layer'
        )(input_placeholder)
        x = LeakyReLU(alpha = 0.2)(x)

        for layer in range(1, self.max_layers):
            output_features = self.n_filters * min(2 ** layer, 8)
            x = Conv2D(
                output_features,
                kernel_size = 4,
                strides = 2,
                padding = 'same',
                use_bias = False,
                name = 'Branch_{0}'.format(layer)
            )(x)
            x = BatchNormalization()(x, training = 1)
            x = LeakyReLU(alpha = 0.2)(x)
        
        output_features = self.n_filters * min(2 ** layer, 8)
        x = ZeroPadding2D(1)(x)
        x = Conv2D(
            output_features,
            kernel_size = 4,
            use_bias = False,
            name = 'Branch_last'
        )(x)
        x = BatchNormalization()(x, training = 1)
        x = LeakyReLU(alpha = 0.2)(x)

        x = ZeroPadding2D(1)(x)
        output = Conv2D(
            1, kernel_size = 4,
            activation = 'sigmoid',
            name = 'Discriminator_Output_Layer'
        )(x)

        self.discrimator = Model([input_placeholder], output)
        print(output.get_shape)
    
    def display(self):
        self.discrimator.summary()
    
    def save(self, model_path):
        self.discrimator.save(model_path)