from tensorflow.keras.initializers import RandomNormal


class Initializer:

    @classmethod
    def conv_initializer(mean = 0, stddev = 0.02):
        return RandomNormal(mean, stddev)
    
    @classmethod
    def batch_norm_initializer(mean = 1.0, stddev = 0.02):
        return RandomNormal(mean, stddev)