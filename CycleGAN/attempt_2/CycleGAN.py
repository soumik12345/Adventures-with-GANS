from Generator import Generator
from Discriminator import Discriminator
from keras.optimizers import Adam


class CycleGan:

    def __init__(self, image_shape, n_filters_generators, n_filters_discriminators, lambda_cycle = 10.0):
        
        # Initialization
        self.image_shape = image_shape
        self.n_filters_generators = n_filters_generators
        self.n_filters_discriminators = n_filters_discriminators
        self.lambda_cycle = lambda_cycle
        self.lambda_id = 0.1 * self.lambda_cycle

        self.get_output_patch()
        self.optimizer = Adam(0.0002, 0.5)

        self.build_discriminators()
        
    
    
    def get_output_patch(self):
        patch = int(self.image_shape[0] // 2 ** 4)
        self.discrimator_patch = (patch, patch, 1)
    

    def build_discriminators(self):
        discriminator_A = Discriminator(self.image_shape, self.n_filters_discriminators)
        discriminator_B = Discriminator(self.image_shape, self.n_filters_discriminators)