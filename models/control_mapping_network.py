import torch
from torch import nn


class ControlMappingNetwork(nn.Module):
    def __init__(self):
        super().__init__()

        self.mapping = nn.Sequential(
            nn.Linear(2, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 512),
        )
    
    def forward(self, x):
        return self.mapping(x)