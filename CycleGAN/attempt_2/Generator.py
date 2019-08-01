from keras.layers import Input, UpSampling2D, Conv2D, Activation
from keras.models import Model
from Utils import Utils


class Generator:

    def __init__(self, image_shape, n_filters = 32):
        
        # Initialization
        self.image_height = image_shape[0]
        self.image_width = image_shape[1]
        self.image_channels = image_shape[2]
        self.n_filters = n_filters

        # Build network
        self.build_network()
    
    def build_network(self):
        
        input_placeholder = Input(shape = (self.image_height, self.image_width, self.image_channels))

        # Downsampling Blocks
        x1 = Utils.conv_layer_generator(input_placeholder, self.n_filters)
        x2 = Utils.conv_layer_generator(x1, self.n_filters * 2)
        x3 = Utils.conv_layer_generator(x2, self.n_filters * 4)
        x4 = Utils.conv_layer_generator(x3, self.n_filters * 8)

        # Upsampling Blocks
        u1 = Utils.deconv_layer_generator(x4, x3, self.n_filters * 4)
        u2 = Utils.deconv_layer_generator(u1, x2, self.n_filters * 2)
        u3 = Utils.deconv_layer_generator(u2, x1, self.n_filters)
        u4 = UpSampling2D(size = 2)(u3)

        output = Conv2D(self.image_channels, kernel_size = 4, strides = 1, padding = 'same')(u4)
        output = Activation('tanh')(output)

        self.generator = Model(input_placeholder, output)
    
    def display(self):
        self.generator.summary()