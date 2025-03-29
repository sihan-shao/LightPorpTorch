import sys
sys.path.append('../')
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import pad, interpolate
from DataType.ElectricField import ElectricField
from torch.fft import fft2, ifft2
from utils.units import *

class Fraunhofer_Prop(nn.Module):
    
    def __init__(self, 
                 z_distance         : float = 0.0, 
                 device             : str = None):
        """
        Fraunhofer far-field propagation method
        
        Args:
            z_distance (float, optional): propagation distance. Defaults to 0.0.
            The propagated field is independent of z distance, which only affect the output pixel size
        """
        super().__init__()

        self.do_padding = True
        self.DEFAULT_PADDING_SCALE = torch.tensor([1,1])
            
        # store the input params
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._z                  = torch.tensor(z_distance, device=self.device)
        
        # the normalized spatial grid
		# we don't actually know dimensions until forward is called
        self.shape = None
        self.meshx = None
        self.meshy = None

        self.check_Zc = True

    @property
    def z(self) -> torch.Tensor:
        return self._z
    
    @z.setter
    def z(self, z : torch.Tensor) -> None:
        if not isinstance(z, torch.Tensor):
            value = torch.tensor(z, device=self.device)
        elif z.device != self.device:
            z = z.to(self.device)
        self._z = z
    
    
    def check_far_field_z(self, dx=None, dy=None, wavelength=None):
        delta_z = 10 * (2 * D**2) / wavelength.min()
        return delta_z
    
    
    def forward(self, 
                field: ElectricField
                ) -> ElectricField:
        
        """
		Fraunhofer far-field propagation

		Args:
			field (ElectricField): Complex field 4D tensor object

		Returns:
			ElectricField: The electric field after the wave propagation model
		"""
  
        # extract the data tensor from the field
        wavelengths = field.wavelengths
        field_data  = field.data
        B,C,H,W = self.shape = field_data.shape
        dx = field.spacing[0]
        dy = field.spacing[1]
        
        # calculate the physics size of input fields
        length_x = H * dx
        length_y = W * dy
        # calculate pixel size after propagtion for each wavelength
        dx_after_prop = wavelengths * self._z / length_x
        dy_after_prop = wavelengths * self._z / length_y
        # find minimum pixel size (highest resolution) among all wavelengths
        min_dx = min(dx_after_prop)
        min_dy = min(dy_after_prop)
        target_dxy = min(min_dx, min_dy)
        # apply FFT for far-field propagation
        field_data = fft2(field_data)
        # calculate a single uniform scaling factor based on the smallest pitch
        scale_factor = float(min_dx / target_dxy) if min_dx <=min_dy else float(min_dy / target_dxy)
        # apply single uniform scaling to all channels
        field_data = interpolate(field_data.real, 
                                 scale_factor, mode='bilinear') + 1j * interpolate(field_data.imag, 
                                                                                   scale_factor, mode='bilinear')

        Eout = ElectricField(
				data=field_data,
				wavelengths=wavelengths,
				spacing=target_dxy
				)
        
        return Eout