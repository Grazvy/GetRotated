import math
import torch.nn as nn
import torch.nn.init as init


class RotationNN(nn.Module):
    def __init__(self, size):
        super(RotationNN, self).__init__()
        self.size = size
        self.fc = nn.Linear(size, size)
        self.sigmoid = nn.Sigmoid()

    def _initialize_weights(self):
        init.normal_(self.fc.weight, mean=0.0, std=1 / math.sqrt(self.size))

    def forward(self, x):
        x = self.fc(x)
        x = self.sigmoid(x)
        return x
