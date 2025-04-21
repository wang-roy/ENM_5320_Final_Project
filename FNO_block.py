import torch
import torch.nn as nn
from typing import List, Union

def __init__(self, modes: Union[list[int], int], in_channels: int, out_channels: int, hidden_size: int = None, activation: nn.Module = nn.GELU(), bias: bool = False) -> None:
    """
    
    loosely based on https://github.com/abelsr/Fourier-Neural-Operator/tree/main/FNO/PyTorch
    """
    super().__init__()
    self.in_channels = in_channels
    self.out_channels = out_channels
    self.hidden_size = hidden_size
    self.activation = activation
    self.modes = modes
    self.dim = len(self.modes)
    self.bias = bias