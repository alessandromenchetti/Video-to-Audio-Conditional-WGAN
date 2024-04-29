import torch
import torch.nn as nn
from video_classifier import VideoClassifier

class VideoEncoder(nn.Module):
    def __init__(self, pretrained_path=None):
        super(VideoEncoder, self).__init__()

        self.encoder = VideoClassifier()

        if pretrained_path:
            preloaded_model = torch.load(pretrained_path)
            self.encoder.load_state_dict(preloaded_model['model_state_dict'])

        self.encoder.fc2 = nn.Identity()

        for param in self.encoder.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.encoder(x)


class GenResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GenResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.skip = nn.Identity()
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        residual = self.skip(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += residual
        x = self.leaky_relu(x)

        return x


class AudioGenerator(nn.Module):
    def __init__(self):
        super(AudioGenerator, self).__init__()

        self.init_linear = nn.Linear(256, 256 * 4 * 4)
        self.bn1 = nn.BatchNorm1d(256 * 4 * 4)
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)

        self.upsample1 = nn.Upsample(scale_factor=2) # [256, 4, 4] -> [256, 8, 8]
        self.res1 = GenResBlock(256, 192) # [256, 8, 8] -> [192, 8, 8]

        self.upsample2 = nn.Upsample(scale_factor=2) # [192, 8, 8] -> [192, 16, 16]
        self.res2 = GenResBlock(192, 160) # [192, 16, 16] -> [160, 16, 16]

        self.upsample3 = nn.Upsample(scale_factor=2) # [160, 16, 16] -> [160, 32, 32]
        self.res3 = GenResBlock(160, 128) # [160, 32, 32] -> [128, 32, 32]

        self.res4 = GenResBlock(128, 128) # [128, 32, 32] -> [128, 32, 32]

        self.upsample4 = nn.Upsample(scale_factor=2) # [128, 32, 32] -> [128, 64, 64]
        self.res5 = GenResBlock(128, 64) # [128, 64, 64] -> [64, 64, 64]

        self.upsample5 = nn.Upsample(scale_factor=2) # [64, 64, 64] -> [64, 128, 128]

        self.final_conv = nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1) # [64, 128, 128] -> [1, 128, 128]
        self.bn2 = nn.BatchNorm2d(1)
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.init_linear(x)
        x = self.bn1(x)
        x = self.leaky_relu(x)

        x = x.view(-1, 256, 4, 4)

        x = self.upsample1(x)
        x = self.res1(x)

        x = self.upsample2(x)
        x = self.res2(x)

        x = self.upsample3(x)
        x = self.res3(x)

        x = self.res4(x)

        x = self.upsample4(x)
        x = self.res5(x)

        x = self.upsample5(x)

        x = self.final_conv(x)
        x = self.tanh(x)

        return x


class CriticResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, spatial_dim):
        super(CriticResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.n1 = nn.LayerNorm([out_channels, spatial_dim, spatial_dim])
        self.leaky_relu = nn.LeakyReLU(0.2, inplace=True)
        self.dropout = nn.Dropout(0.1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.n2 = nn.LayerNorm([out_channels, spatial_dim, spatial_dim])

        self.skip = nn.Identity()
        if in_channels != out_channels:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x):
        residual = self.skip(x)
        x = self.conv1(x)
        x = self.n1(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.n2(x)
        x += residual
        x = self.leaky_relu(x)

        return x


class AudioCritic(nn.Module):
    def __init__(self):
        super(AudioCritic, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(2, 8, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm([8, 128, 128]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1)
        ) # [2, 128, 128] -> [8, 128, 128]
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # [8, 128, 128] -> [8, 64, 64]

        self.res1 = CriticResBlock(8, 16, 64) # [8, 64, 64] -> [16, 64, 64]
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # [16, 64, 64] -> [16, 32, 32]

        self.res2 = CriticResBlock(16, 32, 32) # [16, 32, 32] -> [32, 32, 32]
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # [32, 32, 32] -> [32, 16, 16]

        self.res3 = CriticResBlock(32, 64, 16) # [32, 16, 16] -> [64, 16, 16]
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) # [64, 16, 16] -> [64, 8, 8]

        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm([128, 8, 8]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.1)
        ) # [64, 8, 8] -> [128, 8, 8]
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2) # [128, 8, 8] -> [128, 4, 4]

        self.final_conv = nn.Conv2d(128, 1, kernel_size=4, stride=1, padding=0) # [128, 4, 4] -> [1, 1, 1]

    def forward(self, x, return_intermediate=False):
        x = self.conv1(x)
        x = self.pool1(x)

        x = self.res1(x)
        x = self.pool2(x)

        x = self.res2(x)
        x = self.pool3(x)

        x = self.res3(x)
        intermediate_features = x
        x = self.pool4(x)

        x = self.conv2(x)
        x = self.pool5(x)

        x = self.final_conv(x)

        if return_intermediate:
            return x.view(-1), intermediate_features

        return x.view(-1)