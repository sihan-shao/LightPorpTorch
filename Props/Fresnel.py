import sys
sys.path.append('../')
import numpy as np
import torch
import torchvision
import torch.nn as nn
from torch.nn.functional import pad
from DataType.ElectricField import ElectricField
from utils.Helper_Functions import ft2, ift2
from utils.units import *
from Props.propagation import Propagation

class FresnelPropagator(Propagation):
    
    def __init__(self, 
                 z_distance         : float = 0.0, 
                 device             : str = None, 
                 type               : str = 'tf'):
        """
        Fresnel transfer function propagation method
        and
        Fresnel impluse response propagation method
        
        Args:
            z_distance (float, optional): propagation distance. Defaults to 0.0.
			
        """
        super().__init__()

        self.do_padding = True
        self.DEFAULT_PADDING_SCALE = torch.tensor([1,1])
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # store the input params
        self._z = torch.tensor(z_distance, device=self.device)
        self.type = type
		# we don't actually know dimensions until forward is called
        self.shape = None
        self.check_Zc = True
    
    def check_crucial_distance(self, dx=None, dy=None, wavelength=None):
        range_x, range_y = self.shape[-2] * dx, self.shape[-1] * dy
        if self.type == 'tf':
            Zc = range_x / wavelength.max()
            print("maximum propagation distance to satisfy sampling for FT: {:.3f} mm".format(Zc.detach().cpu().numpy() / m))
            if self._z < Zc:
                print("The simulation will be accurate !")
            else:
                print("The propagation distance should be smaller than maximum propagation distance to keep simulation accurate!")

        if self.type == 'ir':
            Zc = range_x / wavelength.min()
            print("minimum propagation distance to satisfy sampling for FT: {:.3f} mm".format(Zc.detach().cpu().numpy() / m))
            if self._z > Zc:
                print("The simulation will be accurate !")
            else:
                print("The propagation distance should be larger than minimum propagation distance to keep simulation accurate!")

    def create_kernel(self, field : ElectricField):
        # orginal size of E-field
        tempShape = torch.tensor(field.shape)
        tempShapeH = tempShape[-2]
        tempShapeW = tempShape[-1]
        # padding size of E-field
        Pad_tempShapeH, Pad_tempShapeW = self.compute_padding(H=tempShape[-2], W=tempShape[-1], 
                                                              padding_scale=self.DEFAULT_PADDING_SCALE, 
                                                              return_size_of_padding=False)
        # extract dx, dy spacing
        dx = field.spacing[0]
        dy = field.spacing[1]
        #################################################################
		# Prepare Dimensions to shape to be able to process 4D data
		# ---------------------------------------------------------------
		# NOTE: This just means we're broadcasting lower-dimensional
		# tensors to higher dimensional ones

        # Extract and expand wavelengths for H and W dimension
        wavelengths = field.wavelengths
        wavelengths_expand  = wavelengths.view(-1, 1, 1)
        k = 2 * torch.pi / wavelengths_expand
        # Fresnel transfer function
        if self.type == 'tf':
            fx, fy = self.create_frequency_grid(Pad_tempShapeH, Pad_tempShapeW, dx, dy)
            kernel = torch.exp(1j * torch.pi * wavelengths_expand * (fx**2 + fy**2))
        elif self.type == 'ir':
            x, y = self.create_spatial_grid(Pad_tempShapeH, Pad_tempShapeW, dx, dy)
            h = 1 / (1j * wavelengths_expand * self._z) * torch.exp(1j * k / (2 * self._z) * (x**2 + y**2))
            kernel = ft2(h) * dx * dy # to frequency space
        else:
            raise ValueError(f'Fresnel transfer function has only two types !!')

        if self.check_Zc:
            self.check_crucial_distance(dx=dx, dy=dy, wavelength=torch.min(wavelengths))
            self.check_Zc = False # only check once during any loop operation

        return kernel.unsqueeze(0)

    def forward(self, 
                field: ElectricField
                ) -> ElectricField:
        
        """
		Fresnel propagation

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

        if self.do_padding:
            pad_x, pad_y = self.compute_padding(H, W, self.DEFAULT_PADDING_SCALE, return_size_of_padding=True)
            field_data = pad(field_data, (pad_y, pad_y, pad_x, pad_x), mode='constant', value=0)
        
        # Returns a spatial domain kernel with padding size
        Fres_Kernel = self.create_kernel(field=field)
        # convert to angular spectrum
        field_data_spectrum = ft2(field_data)
        field_data = field_data_spectrum * Fres_Kernel
        # Convert 'field_data' back to the space domain
        field_data = ift2(field_data)
        # Unpad the image after convolution, if necessary
        if self.do_padding:
            center_crop = torchvision.transforms.CenterCrop([H,W])
            field_data = center_crop(field_data)

        Eout = ElectricField(
				data=field_data,
				wavelengths=wavelengths,
				spacing=field.spacing
				)
        
        return Eout 

        

        
