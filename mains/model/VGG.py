import torch
import torch.nn as nn
import torch.nn.functional as F


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, max_pool=False, k_size=3):
        super().__init__()

        layers = []
        layers.append(nn.ConvTranspose3d(
            in_channels, in_channels, kernel_size=2, stride=2))
        for i in range(len(out_channels)):
            if i == 0:
                layers.append(nn.Conv3d(
                    in_channels=in_channels, out_channels=out_channels[i], kernel_size=k_size, padding="same"))
            else:
                layers.append(nn.Conv3d(
                    in_channels=out_channels[i-1], out_channels=out_channels[i], kernel_size=k_size, padding="same"))
            layers.append(nn.ReLU())

        if max_pool:
            layers.append(nn.MaxPool3d(kernel_size=2, stride=2))

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for l in self.layers:
            x = l(x)

        return x


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, max_pool=True, k_size=3):
        super().__init__()

        layers = []
        for i in range(len(out_channels)):
            if i == 0:
                layers.append(nn.Conv3d(
                    in_channels=in_channels, out_channels=out_channels[i], kernel_size=k_size, padding="same"))
            else:
                layers.append(nn.Conv3d(
                    in_channels=out_channels[i-1], out_channels=out_channels[i], kernel_size=k_size, padding="same"))
            layers.append(nn.ReLU())

        if max_pool:
            layers.append(nn.MaxPool3d(kernel_size=2, stride=2))

        self.layers = nn.ModuleList(layers)

    def forward(self, x):
        for l in self.layers:
            x = l(x)

        return x


class Encoder(nn.Module):
    def __init__(self, in_channels=2, out_features=512):
        super().__init__()

        # input size = 32, 32, 32
        self.b1 = Block(in_channels=in_channels, out_channels=[64, 64])
        self.b2 = Block(in_channels=64, out_channels=[128, 128])
        self.b3 = Block(in_channels=128, out_channels=[256, 256, 256])
        self.b4 = Block(in_channels=256, out_channels=[512, 512, 512])
        self.b5 = Block(in_channels=512, out_channels=[512, 512, out_features])

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)

        return x


class Decoder(nn.Module):
    def __init__(self, in_features=512, out_channels=1):
        super().__init__()

        # input size = 32, 32, 32
        self.b1 = UpBlock(in_channels=in_features,
                          out_channels=[512, 512, 512])
        self.b2 = UpBlock(in_channels=512, out_channels=[512, 512, 512])
        self.b3 = UpBlock(in_channels=512, out_channels=[256, 256, 256])
        self.b4 = UpBlock(in_channels=256, out_channels=[128, 128])
        self.b5 = UpBlock(in_channels=128, out_channels=[64, 64])
        self.bout = nn.Conv3d(in_channels=64, out_channels=1,
                              kernel_size=3, padding="same")

    def forward(self, x):
        x = self.b1(x)
        x = self.b2(x)
        x = self.b3(x)
        x = self.b4(x)
        x = self.b5(x)

        x = self.bout(x)

        return x


class VGG3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()

        self.encoder = Encoder(in_channels=in_channels)
        self.decoder = Decoder(out_channels=out_channels)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)

        return x

    def __str__(self, ):
        return "VGG3D"


if __name__ == "__main__":
    model = VGG3D()
    inp = torch.rand((1, 1, 64, 64, 64))
    out = model(inp)
    print(out.shape)
