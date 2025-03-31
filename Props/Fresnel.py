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
from Props.propagation import Propagation, Bluestein

class BasicFresnelPropagator(Propagation):
    
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
        if self.type == 'tf':
            Zc = self.shape[-1] * dx**2 / wavelength.max()
            print("maximum propagation distance to satisfy sampling for FT: {:.3f} mm".format(Zc.detach().cpu().numpy() / mm))
            if self._z < Zc:
                print("The simulation will be accurate !")
            else:
                print("The propagation distance should be smaller than maximum propagation distance to keep simulation accurate!")

        if self.type == 'ir':
            Zc = self.shape[-1] * dx**2 / wavelength.min()
            print("minimum propagation distance to satisfy sampling for FT: {:.3f} mm".format(Zc.detach().cpu().numpy() / mm))
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
            h = torch.exp(1j*k*self._z) / (1j * wavelengths_expand * self._z) * torch.exp(1j * k / (2 * self._z) * (x**2 + y**2))
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


class BluesteinFresnelPropagator(Propagation, Bluestein):
    
    def __init__(self, 
                 z_distance         : float = 0.0, 
                 device             : str = None):
        """
        Fresnel impluse response propagation method with Bluestein method
        Useful for imaging light in the focal plane: allows high resolution zoom in z-plane. 
        [Ref 1] Hu, Y., et al. Light Sci Appl 9, 119 (2020).
        
        Args:
            z_distance (float, optional): propagation distance. Defaults to 0.0.
			
        """
        super().__init__()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # store the input params
        self._z = torch.tensor(z_distance, device=self.device)
		# we don't actually know dimensions until forward is called
        self.shape = None
        self.check_Zc = True
    
    def check_crucial_distance(self, dx=None, dy=None, wavelength=None):

        Zc = self.shape[-1] * dx**2 / wavelength.min()
        print("minimum propagation distance to satisfy sampling for FT: {:.3f} mm".format(Zc.detach().cpu().numpy() / mm))
        if self._z > Zc:
            print("The simulation will be accurate !")
        else:
            print("The propagation distance should be larger than minimum propagation distance to keep simulation accurate!")

    def create_kernel(self, z, meshx, meshy, wavelengths):
        wavelengths_expand  =  wavelengths.view(1, -1, 1, 1)
        k = 2 * torch.pi / wavelengths_expand
        kernel = torch.exp(1j * k * z) / (1j * wavelengths_expand * z) * torch.exp(1j * k / (2 * z) * (meshx**2 + meshy**2))
        return kernel

    def forward(self, 
                field: ElectricField, 
                outputHeight=None,
				outputWidth=None, 						
				outputPixel_dx=None, 
				outputPixel_dy=None,					
                ) -> ElectricField:
        """
        Chirped z-transform propagation - efficient diffraction using the Bluestein method.
        Useful for imaging light in the focal plane: allows high resolution zoom in z-plane. 
        [Ref] Hu, Y., et al. Light Sci Appl 9, 119 (2020).

        Parameters:
            field (ElectricField): Complex field 4D tensor object
            outputHeight (int): resolution of height for the output plane.
            outputWidth (int): resolution of width the output plane.
            outputPixel_dx (float): physical length (height) for the output plane
            outputPixel_dy (float): physical length (width) for the putput plane

        Returns ScalarLight object after propagation.
        """
        InputHeight = field.height
        InputWidth = field.width
        InputPixel_dx = field.spacing[0]
        InputPixel_dy = field.spacing[1]
        wavelengths = field.wavelengths

        # Set default values for outputHeight and outputPixel_dx if they are None
        if outputHeight is None:
            outputHeight = InputHeight
        if outputPixel_dx is None:
            outputPixel_dx = InputPixel_dx
        # Set default values for outputWidth and outputPixel_dy if they are None
        if outputWidth is None:
            outputWidth = InputWidth
        if outputPixel_dy is None:
            outputPixel_dy = InputPixel_dy
        

        Inmeshx, Inmeshy, Outmeshx, Outmeshy, Dm, fx_1, fx_2, fy_1, fy_2 = self.build_CZT_grid(self._z, wavelengths,
                                                                                            InputHeight, InputWidth, InputPixel_dx, InputPixel_dy, 
                                                                                             outputHeight, outputWidth, outputPixel_dx, outputPixel_dy)
        # Compute the diffraction integral using Bluestein method
        # Step 1.Compute the transfer function for input and output plane
        F0 = self.create_kernel(self._z, Outmeshx, Outmeshy, wavelengths)
        F  = self.create_kernel(self._z, Inmeshx, Inmeshy, wavelengths)
        # Step 2.Compute (E0 x F) in Eq.(6) in [Ref].
        field = field.data * F
        # Step 3.Bluestein method implementation
        # (1) FFT in Y-dimension
        U = self.Bluestein_method(field, fy_1, fy_2, Dm, outputWidth)
        # (2) FFT in X-dimension using output from (1):
        U = self.Bluestein_method(U, fx_1, fx_2, Dm, outputHeight)
        field = F0 * U * self._z * outputPixel_dx * outputPixel_dy * wavelengths.view(1, -1, 1, 1)
        
        Eout = ElectricField(
				data=field,
				wavelengths=wavelengths,
				spacing=[outputPixel_dx, outputPixel_dy]
				)

        return Eout