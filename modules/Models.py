import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class RotationNN(nn.Module):
    def __init__(self, size):
        super(RotationNN, self).__init__()
        self.size = size
        self.fc = nn.Linear(size, size, bias=False)
        # self._initialize_weights()

    def _initialize_weights(self):
        init.normal_(self.fc.weight, mean=0.0, std=1 / math.sqrt(self.size))

    def clamped_relu(self, x):
        # return torch.clamp(F.relu(x), max=1.0)
        return torch.clamp(F.leaky_relu(x), max=1.0)

    def forward(self, x):
        x = self.fc(x)
        x = self.clamped_relu(x)
        return x
