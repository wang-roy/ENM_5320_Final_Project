import torch
import torch.nn as nn
from FNO_block import FourierBlock
from spectral_convolution import SpectralConvolution
import torch.nn.functional as F

class FNO(nn.Module):
    
    def __init__(self, modes: int, num_fourier_layers: int, in_channels: int, hidden_channels: int, out_channels: int, activation: nn.Module()) -> None:
        
        super().__init__()
        self.modes= modes
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.activation = activation
        self.num_fourier_layers = num_fourier_layers
        self.padding = 1
        
        # initialize lifting layer
        self.lifting = nn.Linear(self.in_channels, self.hidden_channels)
        torch.nn.init.xavier_uniform_(self.lifting.weight)
        
        # initialize fourier layers
        fourier_blocks = [FourierBlock(modes=modes, in_channels=self.hidden_channels,
                                       out_channels=self.hidden_channels, activation=activation) for __ in range(self.num_fourier_layers)]
        self.fourier_blocks = nn.ModuleList(fourier_blocks)
        
        # initialize projection layer
        self.projecting = nn.Linear(self.hidden_channels, self.out_channels)
        torch.nn.init.xavier_uniform_(self.projecting.weight)
        
    
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
