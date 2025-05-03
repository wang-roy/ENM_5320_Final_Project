import torch
import torch.nn as nn
from .FNO_block import FourierBlock
from .spectral_convolution import SpectralConvolution

class FNO(nn.Module):
    
    def __init__(self, modes: int, num_fourier_layers: int, in_channels: int, lift_channels: int, 
                 proj_channels: int, out_channels: int, activation: nn.Module()) -> None:
        
        super.__init__()
        self.modes= modes
        self.in_channels = in_channels
        self.lift_channels = lift_channels
        self.proj_channels = proj_channels
        self.out_channels = out_channels
        self.activation = activation
        
        self.num_fourier_layers = num_fourier_layers
        self.fourier_blocks = nn.ModuleList([FourierBlock(modes=mode, in_channels=in_channels,
                                                          out_channels=out_channels, activation=activation) for mode in modes])
        
        
    
    def forward(self, x) -> torch.Tensor:
        pass
