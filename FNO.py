import torch
import torch.nn as nn

class FNO(nn.Module):
    
    def __init__(self, modes: int, num_fourier_layers: int, in_channels: int, lift_channels: int, proj_channels: int, out_channels: int, activation: nn.Module()):
        self.modes= modes
        self.num_fourier_layers = num_fourier_layers
        self.in_channels = in_channels
        self.lift_channels = lift_channels
        self.proj_channels = proj_channels
        self.out_channels = out_channels
        self.activation = activation
        
