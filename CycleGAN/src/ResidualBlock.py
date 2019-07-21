from torch.nn import Conv2d, ReflectionPad2d, InstanceNorm2d, ReLU, Sequential, Module
from torch import functional as F


class ResidualBlock(Module):


    def __init__(self, input_features):

        super(ResidualBlock, self).__init__()

        conv_block = [
            ReflectionPad2d(1),
            Conv2d(input_features, input_features, 3),
            InstanceNorm2d(input_features),
            ReLU(inplace = True),
            ReflectionPad2d(1),
            Conv2d(input_features, input_features, 3),
            InstanceNorm2d(input_features)
        ]

        self.conv_block = Sequential(*conv_block)
    

    def forward(self, x):
        return x + self.conv_block(x)