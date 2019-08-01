from Generator import Generator
from Discriminator import Discriminator
from keras.optimizers import Adam
from keras.layers import Input
from keras.models import Model


class CycleGAN:

    def __init__(self, image_shape, n_filters_generators, n_filters_discriminators, lambda_cycle = 10.0):
        
        # Initialization
        self.image_shape = image_shape
        self.n_filters_generators = n_filters_generators
        self.n_filters_discriminators = n_filters_discriminators
        self.lambda_cycle = lambda_cycle
        self.lambda_id = 0.1 * self.lambda_cycle
        self.optimizer = Adam(0.0002, 0.5)

        self.get_output_patch()

        self.build_discriminators()
        self.compile_discriminators()

        self.build_generators()

        self.build_gan()


    
    
    def get_output_patch(self):
        patch = int(self.image_shape[0] // 2 ** 4)
        self.discrimator_patch = (patch, patch, 1)
    

    def build_discriminators(self):
        self.discriminator_A = Discriminator(self.image_shape, self.n_filters_discriminators).discrimator
        self.discriminator_B = Discriminator(self.image_shape, self.n_filters_discriminators).discrimator
    

    def compile_discriminators(self):
        self.discriminator_A.compile(loss = 'mse', optimizer = self.optimizer, metrics = ['accuracy'])
        self.discriminator_B.compile(loss = 'mse', optimizer = self.optimizer, metrics = ['accuracy'])
    

    def build_generators(self):
        self.generator_AB = Generator(self.image_shape, self.n_filters_generators).generator
        self.generator_BA = Generator(self.image_shape, self.n_filters_generators).generator
    

    def build_gan(self):

        self.image_A = Input(shape = self.image_shape)
        self.image_B = Input(shape = self.image_shape)

        self.fake_B = self.generator_AB(self.image_A)
        self.fake_A = self.generator_BA(self.image_B)

        self.reconstructed_A = self.generator_BA(self.fake_B)
        self.reconstructed_B = self.generator_AB(self.fake_A)

        self.image_id_A = self.generator_BA(self.reconstructed_B)
        self.image_id_B = self.generator_AB(self.reconstructed_A)

        self.discriminator_A.trainable = False
        self.discriminator_B.trainable = False

        self.valid_A = self.discriminator_A(self.fake_A)
        self.valid_B = self.discriminator_B(self.fake_B)

        self.gan = Model(
            [self.image_A, self.image_B],
            [
                self.valid_A, self.valid_B,
                self.reconstructed_A, self.reconstructed_B,
                self.image_id_A, self.image_id_B
            ]
        )

        self.gan.compile(
            loss = ['mse', 'mse', 'mae', 'mae', 'mae', 'mae'],
            loss_weights = [1, 1, self.lambda_cycle, self.lambda_cycle, self.lambda_id, self.lambda_id],
            optimizer = self.optimizer
        )
    

    def display(self):
        self.gan.summary()