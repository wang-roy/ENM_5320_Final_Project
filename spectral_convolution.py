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
        
        return out