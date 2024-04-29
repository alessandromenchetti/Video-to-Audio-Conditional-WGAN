import torch
import torch.nn as nn
from torchvision import models

class VideoClassifier(nn.Module):
    def __init__(self):
        super(VideoClassifier, self).__init__()

        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.resnet.fc = nn.Identity()

        self.rnn = nn.LSTM(input_size=512, hidden_size=256, num_layers=2, batch_first=True)

        self.fc1 = nn.Linear(256, 128)
        self.leaky_relu = nn.LeakyReLU(0.2)
        self.fc2 = nn.Linear(128, 8)

    def forward(self, x):

        batch_size, C, frames, H, W = x.size()
        x = x.view(batch_size * frames, C, H, W)

        with torch.no_grad():
            x = self.resnet(x)

        x = x.view(batch_size, frames, -1)

        x, _ = self.rnn(x)
        x = x[:, -1, :]

        x = self.fc1(x)
        x = self.leaky_relu(x)
        x = self.fc2(x)


        return x