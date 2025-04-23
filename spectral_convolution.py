import torch
import torch.nn as nn
import torch.fft as fft
from typing import List, Tuple, Optional, Union

class SpectralConvolution(nn.Module):
    """
    Basic implementation of spectral convolution

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

        super(). __init__(
        self.in_channels = in_channels,
        self.out_channels = out_channels,
        self.modes = modes,
        self.dim = len(self.modes),
        self.rank = rank,
        ):
    
    # Initialize weights

    weight_shape = (in_channels, out_channels, *self.modes)

    weights = nn.Parameter(
        nn.init.xavier_uniform_(torch.empty(weight_shape, dtype=torch.float32))
     )


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


        x_ft = fft.fft(x.float())
        
        # 

        out = torch.einsum('bi...,io...->bo...', x_ft, weights)


        # initialize output Tensors (fourier space)
        out_ft = torch.zeros(batch_size, self.out_channels, *sizes, dtype=x_ft_real.dtype, device=x.device)
       



        return out