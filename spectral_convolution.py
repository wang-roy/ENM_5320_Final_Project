import torch
import torch.nn as nn
import torch.fft as fft
from typing import List, Tuple, Optional, Union

class SpectralConvolution(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, modes: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        
        # Initialize complex weights with Xavier initialization
        scale = 1 / (in_channels * out_channels)
        
        weights_real = torch.empty(in_channels, out_channels, modes, dtype=torch.float32)
        weights_imag = torch.empty(in_channels, out_channels, modes, dtype=torch.float32)
        
        nn.init.normal_(weights_real, 0, scale)
        nn.init.normal_(weights_imag, 0, scale)
        
        self.weights = nn.Parameter(torch.complex(weights_real, weights_imag))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for 1D spectral convolution.
        
        Args:
            x (torch.Tensor): Input tensor of shape (batch, in_channels, length).
        
        Returns:
            torch.Tensor: Output tensor of shape (batch, out_channels, length).
        """
        batch_size, _, length = x.shape
        
        # Compute 1D FFT
        x_ft = torch.fft.fft(x.float(), dim=-1)  # shape: (batch, in_channels, length)
        
        # Truncate to the modes we're keeping
        x_ft_trunc = x_ft[..., :self.modes]  # shape: (batch, in_channels, modes)
        
        # Spectral convolution using einsum
        out_ft = torch.einsum('bix,iox->box', x_ft_trunc, self.weights)
        
        # Pad back to original length
        out_ft_padded = torch.zeros(batch_size, self.out_channels, length, 
                                   dtype=x_ft.dtype, device=x.device)
        out_ft_padded[..., :self.modes] = out_ft
        
        # Inverse FFT and return real part
        out = torch.fft.ifft(out_ft_padded, dim=-1)
        return out.real