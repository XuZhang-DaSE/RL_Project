import torch
import torch.nn as nn

class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DuelingDQN, self).__init__()
        self.feature = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU())
        self.value = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, 1))
        self.advantage = nn.Sequential(nn.Linear(128, 64), nn.ReLU(), nn.Linear(64, output_dim))

    def forward(self, x):
        x = self.feature(x)
        value = self.value(x)
        adv = self.advantage(x)
        return value + adv - adv.mean()
