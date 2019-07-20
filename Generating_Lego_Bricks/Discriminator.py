from torch.nn import Conv2d, BatchNorm2d
from torch.nn.functional import leaky_relu, sigmoid


class Discriminator(Module):

    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv_1_data = Conv2d(3, 64, 4, 2, 1)
        self.conv_1_label = Conv2d(2, 64, 4, 2, 1)

        self.conv_2 = Conv2d(128, 256, 4, 2, 1)
        self.batch_norm_2 = Conv2d(256)

        self.conv_3 = Conv2d(256, 512, 4, 2, 1)
        self.batch_norm_3 = Conv2d(512)

        self.conv_4 = Conv2d(512, 1024, 4, 2, 1)
        self.batch_norm_4 = Conv2d(1024)

        self.conv_5 = Conv2d(1024, 1, 2)
    

    def forward(self, inputs, labels):

        # Image Input Block
        x = self.conv_1_data(inputs)
        x = leaky_relu(x)

        # Label Input Block
        y = self.conv_1_label(labels)
        y = self.leaky_relu(y)

        # Concat Image and Labels
        x = cat((x, y), dim = 1)

        # Convolutional Block 2
        x = self.conv_2(x)
        x = self.batch_norm_2(x)
        x = leaky_relu(x, negative_slope = 0.2)

        # Convolutional Block 3
        x = self.conv_3(x)
        x = self.batch_norm_3(x)
        x = leaky_relu(x, negative_slope = 0.2)

        # Convolutional Block 4
        x = self.conv_4(x)
        x = self.batch_norm_4(x)
        x = leaky_relu(x, negative_slope = 0.2)

        # Output Block
        x = self.conv_5(x)
        output = sigmoid(x)
        output = output.view(-1, 1)

        return output