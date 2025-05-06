import torch
import torch.nn as nn
from FNO_block import FourierBlock
from spectral_convolution import SpectralConvolution
import torch.nn.functional as F

class FNO(nn.Module):
    
    def __init__(self, modes: int, num_fourier_layers: int, in_channels: int, fourier_channels: int, out_channels: int, activation: nn.Module()) -> None: #, lift_channels: int, proj_channels: int) -> None:
        
        super().__init__()
        self.modes= modes
        self.in_channels = in_channels
        #self.lift_channels = lift_channels
        self.fourier_channels = fourier_channels
        #self.proj_channels = proj_channels
        self.out_channels = out_channels
        self.activation = activation
        self.num_fourier_layers = num_fourier_layers
        self.padding = 1
        
        
        self.lifting = nn.Linear(self.in_channels, self.fourier_channels)
        fourier_blocks = [FourierBlock(modes=modes, in_channels=fourier_channels,
                                       out_channels=fourier_channels, activation=activation) for __ in range(num_fourier_layers)]
        self.fourier_blocks = nn.ModuleList(fourier_blocks)
        self.projecting = nn.Linear(self.fourier_channels, self.out_channels)
        
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, in_channels, *sizes = x.size()
        # lift
        x = self.lifting(x)
        x = x.permute(0, 2, 1)
        
        # pad for non-periodic domains
        x = F.pad(x, [0, self.padding]) 
        
        # fourier layers
        for fourier_block in self.fourier_blocks:
            x = fourier_block(x)
        
        # remove padding
        x = x[...,:-self.padding]
        x = x.permute(0, 2, 1)
        
        # project and output
        x = self.projecting(x)
        # x = self.activation(x) # do we need? activation fn passed onto fourier layers already
        return x
