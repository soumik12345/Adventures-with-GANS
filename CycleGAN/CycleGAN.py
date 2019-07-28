from Discriminator import Discriminator
from Generator import Generator
from tensorflow.keras import backend as K


class CycleGan:

    def __init__(self, discriminator_filters, generator_filters, is_lsgan):
        
        self.discriminator_filters = discriminator_filters
        self.generator_filters = generator_filters
        self.loss_function = CycleGAN.ls_loss if is_lsgan else CycleGAN.cycle_gan_loss

        self.build_discriminators()
        self.build_generators()

        self.real_A, self.fake_B, self.recreated_A, self.generator_function_A = self.initialize_variables(
            self.generator_B,
            self.generator_A
        )
        self.real_B, self.fake_A, self.recreated_B, self.generator_function_B = self.initialize_variables(
            self.generator_A,
            self.generator_B
        )

        self.discriminator_loss_B = self.discriminator_loss(
            self.discriminator_B,
            self.real_B,
            self.fake_B,
            self.recreated_B
        )

        self.discriminator_loss_A = self.discriminator_loss(
            self.discriminator_A,
            self.real_A,
            self.fake_A,
            self.recreated_A
        )
        
    
    def build_discriminators(self):
        
        self.discriminator_A = Discriminator(
            n_filters = self.discriminator_filters,
            max_layers = 3,
            image_size = 128,
            image_channels = 3
        )
        
        self.discriminator_B = Discriminator(
            n_filters = self.discriminator_filters,
            max_layers = 3,
            image_size = 128,
            image_channels = 3
        )

    
    def build_generators(self):
        
        self.generator_A = Generator(
            n_filters = self.generator_filters,
            image_size = 128,
            image_channels = 3
        )
        
        self.generator_B = Generator(
            n_filters = self.generator_filters,
            image_size = 128,
            image_channels = 3
        )
    

    @staticmethod
    def ls_loss(y_pred, y_true):
        return K.mean(K.abs(K.square(y_pred, y_true)))
    

    @staticmethod
    def cycle_gan_loss(y_pred, y_true):
        return -K.mean(K.log(y_pred+1e-12) * y_true + K.log(1 - y_pred+1e-12) * y_true)
    

    def initialize_variables(self, generator_1, generator_2):
        real_input = generator_1.inputs[0]
        fake_output = generator_1.outputs[1]
        recreated_input = generator_2([fake_output])
        generator_function = K.function([real_input], [fake_output, recreated_input])
        return real_input, fake_output, recreated_input, generator_function
    
    
    def discriminator_loss(self, discrimator, real_image, fake_image, recreated_image):
        real_output = discrimator([real_image])
        fake_output = discrimator([fake_image])
        discriminator_loss_real = self.loss_function(real_output, K.ones_like(real_output))
        discriminator_loss_fake = self.loss_function(fake_output, K.ones_like(fake_output))
        discriminator_loss = discriminator_loss_fake + discriminator_loss_real
        generator_loss = self.loss_function(fake_output, K.ones_like(fake_output))
        cycle_loss = K.mean(K.abs(recreated_image, real_image))
        return discriminator_loss, generator_loss, cycle_loss