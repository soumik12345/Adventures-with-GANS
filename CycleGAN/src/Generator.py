from torch.nn import Conv2d, ConvTranspose2d, InstanceNorm2d
from torch.nn import ReLU, ReflectionPad2d, Module
from torch.nn import Tanh, Sequential
from src.ResidualBlock import ResidualBlock


class Generator(Module):

    def __init__(self, input_channels, output_channels, residual_blocks):
        super(Generator, self).__init__()

        # Initial convolution block       
        model = [
            ReflectionPad2d(3),
            Conv2d(input_channels, 64, 7),
            InstanceNorm2d(64),
            ReLU(inplace = True)
        ]

        # Downsampling
        input_features = 64
        output_features = input_features * 2
        for _ in range(2):
            model += [
                Conv2d(input_features, output_features, 3, stride = 2, padding = 1),
                InstanceNorm2d(output_features),
                ReLU(inplace = True)
            ]
            input_features = output_features
            output_features = input_features * 2

        # Residual blocks
        for _ in range(residual_blocks):
            model += [ResidualBlock(input_features)]

        # Upsampling
        output_features = input_features//2
        for _ in range(2):
            model += [
                ConvTranspose2d(
                    input_features,
                    output_features,
                    3, stride = 2,
                    padding = 1,
                    output_padding = 1
                ),
                InstanceNorm2d(output_features),
                ReLU(inplace = True)
            ]
            input_features = output_features
            output_features = input_features // 2

        # Output layer
        model += [
            ReflectionPad2d(3),
            Conv2d(64, output_channels, 7),
            Tanh()
        ]

        self.model = Sequential(*model)

    def forward(self, x):
        return self.model(x)
