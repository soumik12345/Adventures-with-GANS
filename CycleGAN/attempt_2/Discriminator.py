from keras.layers import Input, Conv2D
from keras.models import Model
from Utils import Utils


class Discriminator:

    def __init__(self, image_shape, n_filters):

        # Initialization
        self.image_height = image_shape[0]
        self.image_width = image_shape[1]
        self.image_channels = image_shape[2]
        self.n_filters = n_filters

        self.build_network()

    
    def build_network(self):

        input_placeholder = Input(shape = (self.image_height, self.image_width, self.image_channels))

        x1 = Utils.conv_layer_discriminator(input_placeholder, self.n_filters, normalization = False)
        x2 = Utils.conv_layer_discriminator(x1, self.n_filters * 2)
        x3 = Utils.conv_layer_discriminator(x2, self.n_filters * 4)
        x4 = Utils.conv_layer_discriminator(x3, self.n_filters * 8)

        output = Conv2D(1, kernel_size = 4, strides = 1, padding = 'same')(x4)

        self.discrimator = Model(input_placeholder, output)
    

    def display(self):
        self.discrimator.summary()
    