from torch.nn import Conv2d, ReflectionPad2d, InstanceNorm2d, ReLU, Sequential, Module, LeakyReLU
from torch.nn.functional import avg_pool2d


class Discriminator(Module):
    
    def __init__(self, input_channels):
        super(Discriminator, self).__init__()

        model = [
            Conv2d(input_channels, 64, 4, stride = 2, padding = 1),
            LeakyReLU(0.2, inplace=True) ]

        model += [
            Conv2d(64, 128, 4, stride = 2, padding = 1),
            InstanceNorm2d(128),
            LeakyReLU(0.2, inplace = True)
        ]

        model += [
            Conv2d(128, 256, 4, stride = 2, padding = 1),
            InstanceNorm2d(256),
            LeakyReLU(0.2, inplace = True)
        ]

        model += [
            Conv2d(256, 512, 4, padding = 1),
            InstanceNorm2d(512),
            LeakyReLU(0.2, inplace = True)
        ]

        model += [Conv2d(512, 1, 4, padding = 1)]

        self.model = Sequential(*model)

    def forward(self, x):
        x =  self.model(x)
        return avg_pool2d(x, x.size()[2:]).view(x.size()[0], -1)