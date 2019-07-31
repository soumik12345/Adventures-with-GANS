from Discriminator import Discriminator
from Generator import Generator
from tensorflow.keras import backend as K
from tensorflow.keras.optimizer import Adam
from PIL import Image
from glob import glob
from random import shuffle, randint, sample
import numpy as np
from PIL import Image


class CycleGan:

    def __init__(
        self,
        discriminator_filters,
        base_directory,
        subdirectory_names,
        generator_filters,
        generator_learning_rate = 2e-4,
        discriminator_learning_rate = 2e-4,
        batch_size = 16,
        is_lsgan = False):
        
        self.discriminator_filters = discriminator_filters
        self.discriminator_learning_rate = discriminator_learning_rate
        self.generator_filters = generator_filters
        self.generator_learning_rate = generator_learning_rate
        self.loss_function = CycleGAN.ls_loss if is_lsgan else CycleGAN.loss
        self._lambda = 10 if is_lsgan else 100
        self.batch_size = batch_size

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
        
        self.compute_loss()
        self.compute_weights()
        self.define_optimizers()
        self.train_discriminator()
        self.discriminator_training_update()
        self.generator_training_update()
        self.load_data(base_directory, subdirectory_names)






        
    def read_image(self, image_path):
        image = Image.open(image_path).convert('RGB').resize((128, 128), Image.BILINEAR)
        image = np.array(image) / 255 * 2 - 1
        if randint(0, 1) == 1:
            image = image[:, ::-1]
        return image
    

    def load_data(self, base_directory, subdirectory_names):
        self.data_A = shuffle(glob(base_directory + subdirectory_names[0] + '/*'))
        self.data_B = shuffle(glob(base_directory + subdirectory_names[1] + '/*'))
    

    def batch(self, dataset):
        temp_size, epoch, i, data_length = None, 0, 0, len(dataset)
        while True:
            size = temp_size if temp_size else self.batch_size
            if i > data_length - size:
                i = 0
                epoch += 1
            image = [self.read_image(dataset[j])]


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
    

    def compute_loss(self):
        self.discriminator_loss_B, self.generator_loss_B, self.cycle_loss_B = self.discriminator_loss(
            self.discriminator_B,
            self.real_B,
            self.fake_B,
            self.recreated_B
        )
        self.discriminator_loss_A, self.generator_loss_A, self.cycle_loss_A = self.discriminator_loss(
            self.discriminator_A,
            self.real_A,
            self.fake_A,
            self.recreated_A
        )
        self.cycle_loss = self.cycle_loss_A + self.cycle_loss_B
        self.generator_loss = self.generator_loss_A + self.generator_loss_B + self._lambda * self.cycle_loss
        self.discriminator_loss = self.discriminator_loss_A + self.discriminator_loss_B
    

    def compute_weights(self):
        self.discriminator_weights = self.discriminator_A.trainable_weights + self.discriminator_B.trainable_weights
        self.generator_weights = self.generator_A.trainable_weights + self.generator_B.trainable_weights
    

    def define_optimizers(self):
        self.discriminator_optimizer = Adam(lr = self.discriminator_learning_rate, beta_1 = 0.5)
        self.generator_optimizer = Adam(lr = self.generator_optimizer, beta_1 = 0.5)
    

    def discriminator_training_update(self):
        self.training_update = self.discriminator_optimizer.get_updates(
            self.discriminator_weights, [],
            self.discriminator_loss
        )
        self.discriminator_training = K.function(
            [
                self.real_A,
                self.real_B
            ],
            [
                self.discriminator_loss_A / 2,
                self.discriminator_loss_B / 2
            ],
            self.training_update
        )
    

    def generator_training_update(self):
        self.training_update = self.generator_optimizer.get_updates(
            self.generator_weights, [],
            self.generator_loss
        )
        self.generator_training = K.function(
            [
                self.real_A,
                self.real_B
            ],
            [
                self.generator_loss_A,
                self.generator_loss_B,
                self.cycle_loss
            ],
            self.training_update
        )