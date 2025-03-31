import sys
sys.path.append('../')
from abc import ABC, abstractmethod
import numpy as np
import torch
import torchvision
import torch.nn as nn
from torch.nn.functional import pad
from DataType.ElectricField import ElectricField
from utils.Helper_Functions import ft2, ift2
from utils.units import *

class Propagation(ABC, nn.Module):

    @property
    def z(self) -> torch.Tensor:
        return self._z
    
    @z.setter
    def z(self, z : torch.Tensor) -> None:
        self._z = torch.tensor(z, device=self.device)
    
    def compute_padding(self, H, W, padding_scale, return_size_of_padding=False):
        # get the shape of processing
        paddingH = int(np.floor((padding_scale[0] * H) / 2))
        paddingW = int(np.floor((padding_scale[1] * W) / 2))
        paddedH = H + 2*paddingH
        paddedW = W + 2*paddingW
        
        if not return_size_of_padding:
            return paddedH, paddedW
        else:
            return paddingH, paddingW
    
    def create_spatial_grid(self, H, W, dx, dy):
        """
        creates the spatial coordinate grid in x and y direction
        """
        x = torch.linspace(-H * dx / 2, H * dx / 2, H, device=dx.device)
        y = torch.linspace(-W * dx / 2, W * dy / 2, W, device=dy.device)
        meshx, meshy = torch.meshgrid(x, y, indexing='ij')

        return meshx, meshy
    
    def create_frequency_grid(self, H, W, dx, dy):
        """
        creates the frequency coordinate grid in x and y direction
        """
        fx = (torch.linspace(0, H - 1, H, device=dx.device) - (H // 2)) / (H*dx)
        fy = (torch.linspace(0, W - 1, W, device=dy.device) - (W // 2)) / (W*dy)
            
        meshfx, meshfy = torch.meshgrid(fx, fy, indexing='ij')

        return meshfx, meshfy

    @abstractmethod
    def check_crucial_distance(self):
        return NotImplemented
    
    @abstractmethod
    def create_kernel(self, field : ElectricField):
        return NotImplemented
