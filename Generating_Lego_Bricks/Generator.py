from torch.nn import Module, ConvTranspose2d, BatchNorm2d
from torch.nn.functional import tanh, leaky_relu
from torch import cat


class Generator(Module):

    def __init__(self):
        super(Generator, self).__init__()
        
        self.deconv_1_data = ConvTranspose2d(100, 256, 4, 1, 0)
        self.batch_norm_1_data = BatchNorm2d(256)

        self.deconv_1_label = ConvTranspose2d(2, 256, 4, 1, 0)
        self.batch_norm_1_label = BatchNorm2d(256)

        self.deconv_2 = ConvTranspose2d(512, 256, 4, 2, 1)
        self.batch_norm_2 = BatchNorm2d(256)

        self.deconv_3 = ConvTranspose2d(256, 128, 4, 2, 1)
        self.batch_norm_3 = BatchNorm2d(128)

        self.deconv_4 = ConvTranspose2d(128, 64, 4, 2, 1)
        self.batch_norm_4 = BatchNorm2d(64)

        self.deconv_5 = ConvTranspose2d(64, 3, 4, 2, 1)
    

    def forward(self, inputs, labels):

        # Upsampling Block for Data
        x = self.deconv_1_data(inputs)
        x = self.batch_norm_1_data(x)
        x = leaky_relu(x, negative_slope = 0.2)

        # Upsampling Block for labels
        y = self.deconv_1_label(labels)
        y = self.batch_norm_1_label(y)
        y = leaky_relu(y)

        # Concating noise and labels
        x = cat((x, y), dim = 1)

        # Upsampling Block 2
        x = self.deconv_2(x)
        x = self.batch_norm_2(x)
        x = leaky_relu(x, negative_slope = 0.2)

        # Upsampling Block 3
        x = self.deconv_3(x)
        x = self.batch_norm_3(x)
        x = leaky_relu(x, negative_slope = 0.2)

        # Upsampling Block 4
        x = self.deconv_4(x)
        x = self.batch_norm_4(x)
        x = leaky_relu(x, negative_slope = 0.2)

        # Output Block
        x = self.deconv_5(x)
        output = tanh(x)

        return output