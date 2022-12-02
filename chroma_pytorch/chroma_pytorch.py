import torch
from torch import nn, einsum

from einops import rearrange, repeat

class Chroma(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x
