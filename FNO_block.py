import torch
import torch.nn as nn
from typing import List, Union
from spectral_convolution import SpectralConvolution


class FourierBlock(nn.Module):
    
    """
    loosely based on https://github.com/abelsr/Fourier-Neural-Operator/tree/main/FNO/PyTorch
    """

    def __init__(self, modes: Union[list[int], int], in_channels: int, out_channels: int, activation: nn.Module = nn.Tanh()) -> None:
        """
        
        
        """
        super().__init__()
        self.in_channels = in_channels
        
        self.out_channels = out_channels
        self.activation = activation
        self.modes = modes

        # Fourier Layer 
        self.fourier = SpectralConvolution(in_channels, out_channels, modes)

        # Convolution Layer/Local Linear Transform
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=1)#, padding=1)
        torch.nn.init.xaiver_uniform_(self.conv.weight)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        """
        assert x.size(1) == self.in_channels, f"Input channels must be {self.in_channels} but got {x.size(1)} instead."
        
        x_ft = self.fourier(x)
        
        x_conv = self.conv(x)

        # Activation
        x = self.activation(x_ft + x_conv)
        return x
