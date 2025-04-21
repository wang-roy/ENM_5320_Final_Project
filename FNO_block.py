import torch
import torch.nn as nn
from typing import List, Union

class FourierBlock(nn.Module):
    
    """
    loosely based on https://github.com/abelsr/Fourier-Neural-Operator/tree/main/FNO/PyTorch
    """

    def __init__(self, modes: Union[list[int], int], in_channels: int, out_channels: int, hidden_size: int = None, activation: nn.Module = nn.GELU(), bias: bool = False) -> None:
        """
        
        
        """
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_size = hidden_size
        self.activation = activation
        self.modes = modes
        self.dim = len(self.modes)
        self.bias = bias

        # Fourier Layer ()

        # MLP Layer


        # Convolution Layer
        self.conv = nn.Conv1d(in_channels, out_channels, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        """
        assert x.size(1) == self.in_channels, f"Input channels must be {self.in_channels} but got {x.size(1)} instead."

        sizes = x.size()

        if self.bias:
            bias = x

        # Fourier layer
        x_ft = self.fourier(x)

        # MLP layer
        if self.hidden_size is not None:
            x_mlp = self.mlp(x)

        # Convolution layer
        if self.dim == 2 or self.dim == 3:
            x_conv = self.conv(x)
        else: 
            x_conv = self.conv(x.reshape(sizes[0], self.in_channels, -1)).reshape(*sizes)

        # Add
        x = x_ft + x_conv
        if self.hidden_size is not None:
            x = x + x_mlp
        if self.bias:
            x = x + bias

        # Activation
        x = self.activation(x)
        return x
