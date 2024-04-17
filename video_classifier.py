import torch
import torch.nn as nn
from torchvision import models

class VideoClassifier(nn.Module):
    def __init__(self):
        super(VideoClassifier, self).__init__()

        self.resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.resnet.fc = nn.Identity()

        self.rnn = nn.LSTM(input_size=512, hidden_size=256, num_layers=2, batch_first=True)

        self.fc = nn.Linear(256, 8)

    def forward(self, x):

        batch_size, C, frames, H, W = x.size()
        x = x.view(batch_size * frames, C, H, W)

        with torch.no_grad():
            x = self.resnet(x)

        x = x.view(batch_size, frames, -1)

        x, _ = self.rnn(x)
        x = x[:, -1, :]

        x = self.fc(x)

        return x