import torch
import torch.nn as nn
import torch.fft as fft
from typing import List, Tuple, Optional, Union

class SpectralConvolution(nn.Module):
    """
    Basic implementation of 1D spectral convolution

    Args
    """

    def __init__(self, 
        in_channels: int,
        out_channels: int,
        modes: List[int],
        rank: int = 8,
        bias: bool = True,
        **kwargs
    ):

        super(). __init__()
        self.in_channels = in_channels,
        self.out_channels = out_channels,
        self.modes = modes,
        self.dim = len(self.modes)
        self.rank = rank, 

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the spectral convolution layer.

        Args:
            x (torch.Tensor): Input tensor of shape (batch, in_channels, D1, D2, ..., DN).

        Returns:
            torch.Tensor: Output tensor of shape (batch, out_channels, D1, D2, ..., DN).
        """
        batch_size, _, *sizes = x.shape

        if len(sizes) != self.dim:
            raise ValueError(f"Expected input to have {self.dim + 2} dimensions (including batch and channel), but got {len(sizes) + 2}")


        x_fft = fft.fftn(x.float(), dim = range(-self.dim, 0), norm = "ortho")

        
        return out