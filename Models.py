import math
import torch.nn as nn
import torch.nn.init as init

class RotationNN(nn.Module):
    def __init__(self, size):
        super(RotationNN, self).__init__()
        self.size = size
        #todo
        self.fc1 = nn.Linear(size, size)
        self.relu = nn.ReLU()

    def _initialize_weights(self):
        init.normal_(self.fc1.weight, mean=0.0, std=1 / math.sqrt(self.size))
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        return x