import sys
sys.path.append('../')
import numpy as np
import torch
import torch.nn as nn
from torch.nn.functional import pad
from DataType.ElectricField import ElectricField
from torch.fft import fft2, ifft2
from utils.units import *
from Props.propagation import Propagation, Bluestein

"""
1. Scalar Rayleigh-Sommerfeld convolution method
2. Vectorial Rayleigh-Sommerfeld convolution method
3. Bluestein (scalar) Rayleigh-Sommerfeld convolution method
"""

class RSCPropagator(Propagation):
    
    def __init__(self, 
                 z_distance         : float = 0.0, 
                 device             : str = None
                 ) -> None:
        """
        Rayleigh-Sommerfeld convolution
        [Ref 1: F. Shen and A. Wang, Appl. Opt. 45, 1102-1110 (2006)].

        Args:
            z_distance (float, optional): propagation distance. Defaults to 0.0.
			
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
    
    def check_crucial_distance(self, quality_factor=1, dx=None, dy=None, wavelength=None):
        """
        Given a quality factor, determines the minimum available (trustworthy) distance for VRS_propagation().
        [Ref 2: Laser Phys. Lett., 10(6), 065004 (2013)] for the perspective of energy conservation in FFT
        [Ref 3: J. Opt. Soc. Am. A 37, 1748-1766 (2020)] for the perspective of sampling in FFT 

        Parameters:
            quality_factor (int): Defaults to 1.
        
        part of code is adapted from https://github.com/artificial-scientist-lab/XLuminA/blob/main/xlumina

        Returns the minimum distance z necessary to achieve qualities larger than quality_factor in [Ref 2] or satisfy the sampling issue in [Ref 3].
        """
        # Ref 2 
        range_x, range_y = self.shape[-2] * dx, self.shape[-1] * dy
        # Delat rho
        dr_real = torch.sqrt(dx**2 + dy**2)
        # Rho
        rmax = torch.sqrt(range_x**2 + range_y**2)
        n = 1 # free space
        factor = (((quality_factor * dr_real + rmax)**2 - (wavelength / n)**2 - rmax**2) / (2 * wavelength / n))**2 - rmax**2

        if factor > 0:
            z_min1 = torch.sqrt(factor)
        else:
            z_min1 = 0
        
        print("Minimum propagation distance to satisfy energy conservation: {:.3f} mm".format(z_min1.detach().cpu().numpy() / mm))
        # Ref 3 Eq.34
        z_min2 = self.meshx.shape[0] * dx**2 / wavelength * torch.sqrt(1 - (wavelength / (2 * dx))**2)
        print("Minimum propagation distance to satisfy sampling for FT: {:.3f} mm".format(z_min2.detach().cpu().numpy() / mm))

        Zc = min(z_min1, z_min2)

        if self._z > Zc:
            print("The simulation will be accurate !")
        else:
            print("The propagation distance should be larger than minimum propagation distance to keep simulation accurate!")
    
    def create_kernel(self,
		field : ElectricField,
			):
        
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
        
        # create grid for RS transfer function
        self.meshx, self.meshy = self.create_spatial_grid(Pad_tempShapeH, Pad_tempShapeW, dx, dy)

        #################################################################
		# Prepare Dimensions to shape to be able to process 4D data
		# ---------------------------------------------------------------
		# NOTE: This just means we're broadcasting lower-dimensional
		# tensors to higher dimensional ones

        # Extract and expand wavelengths for H and W dimension
        wavelengths = field.wavelengths
        wavelengths_expand  = wavelengths[:,None,None]
        k = 2 * torch.pi / wavelengths_expand
        
        # RS transfer function
        r = torch.sqrt(self.meshx**2 + self.meshy**2 + self._z**2)
        factor = 1 / (2 * torch.pi) * self._z / r**2 *(1 / r - 1j * k)
        kernel = torch.exp(1j * k * r) * factor
        
        if self.check_Zc:
            self.check_crucial_distance(quality_factor=1, dx=dx, dy=dy, wavelength=torch.min(wavelengths))
            self.check_Zc = False # only check once during any loop operation

        return kernel[None, ...]   

    def forward(self, 
                field: ElectricField
                ) -> ElectricField:
        
        """
		Rayleigh-Sommerfeld convolution

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
        
        #################################################################
        # Apply Rayleigh-Sommerfeld convolution
        #################################################################
        
        # Returns a spatial domain kernel with padding size
        RSC_Kernel = self.create_kernel(field=field)

        U = torch.zeros_like(RSC_Kernel)

        U[..., 0:H, 0:W] = field_data  # Ref 1 Eq.11

        # convert to angular spectrum
        field_data_spectrum = fft2(U) * fft2(RSC_Kernel) * dx * dy
                
        # Convert 'field_data' back to the space domain
        # Ref 1 Eq.15 'lower right submatrix'
        field_data = ifft2(field_data_spectrum)[..., H:, W:]	# Dimensions: B=1 x C x H x W

        Eout = ElectricField(
				data=field_data,
				wavelengths=wavelengths,
				spacing=field.spacing
				)
        
        return Eout


class BluesteinRSCPropagator(Propagation, Bluestein):
    def __init__(self, 
                 z_distance         : float = 0.0, 
                 device             : str = None
                 ) -> None:
        """
        Chirped z-transform propagation - efficient diffraction using the Bluestein method.
        Useful for imaging light in the focal plane: allows high resolution zoom in z-plane. 
        [Ref 1] Hu, Y., et al. Light Sci Appl 9, 119 (2020).
        
        Args:
            z_distance (float, optional): propagation distance along the z-direction. Defaults to 0.0.
        """
        super().__init__()

        # store the input params
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._z = torch.tensor(z_distance, device=self.device)
    
    def create_kernel(self, z, meshx, meshy, wavelengths):
        """
        function for RS transfer function
        """
        wavelengths_expand  =  wavelengths.view(1, -1, 1, 1)
        k = 2 * torch.pi / wavelengths_expand

        r = torch.sqrt(meshx**2 + meshy**2 + z**2)
        factor = 1 / (2 * torch.pi) * z / r**2 * (1 / r - 1j*k)
        kernel = torch.exp(1j * k * r) * factor
        # Do we need to check the minimum propagation distance here? 

        return kernel
    
    def check_crucial_distance(self):
        pass
    
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



class VRSPropagator(RSCPropagator):

    """
    Vectorial propagation in free space is just a propagation of each of the components and each vectorial component can be propagated separately.
    """

    def create_kernel(self,
	    field : ElectricField,
			):

        # orginal size of E-field
        tempShape = torch.tensor(field.shape)
        tempShapeH = tempShape[-2]
        tempShapeW = tempShape[-1]
        # padding size of E-field
        # padding size of E-field
        Pad_tempShapeH, Pad_tempShapeW = self.compute_padding(H=tempShape[-2], W=tempShape[-1], 
                                                              padding_scale=self.DEFAULT_PADDING_SCALE, 
                                                              return_size_of_padding=False)
        # extract dx, dy spacing
        dx = field.spacing[0]
        dy = field.spacing[1]
        
        # create grid for RS transfer function
        self.meshx, self.meshy = self.create_spatial_grid(Pad_tempShapeH, Pad_tempShapeW, dx, dy)

        #################################################################
		# Prepare Dimensions to shape to be able to process 4D data
		# ---------------------------------------------------------------
		# NOTE: This just means we're broadcasting lower-dimensional
		# tensors to higher dimensional ones

        # Extract and expand wavelengths for H and W dimension
        wavelengths = field.wavelengths
        wavelengths_expand  = wavelengths[:,None,None]
        k = 2 * torch.pi / wavelengths_expand
        
        # RS transfer function
        r = torch.sqrt(self.meshx**2 + self.meshy**2 + self._z**2)
        factor = 1 / (2 * torch.pi) * self._z / r**2 *(1 / r - 1j * k)
        kernel = torch.exp(1j * k * r) * factor
        
        if self.check_Zc:
            self.check_crucial_distance(quality_factor=1, dx=dx, dy=dy, wavelength=torch.min(wavelengths))
            self.check_Zc = False # only check once during any loop operation

        return kernel[None, ...]     
        
    
    def forward(self, 
                field: ElectricField
                ) -> ElectricField:
        
        """
		Vectorial Rayleigh-Sommerfeld convolution
        [Ref 1: Laser Phys. Lett., 10(6), 065004 (2013)].

		Args:
			field (ElectricField): Complex field 4D tensor object

		Returns:
			ElectricField: The electric field after the wave propagation model
		"""
  
        # extract the data tensor from the field
        wavelengths = field.wavelengths
        B,C,H,W = self.shape = field.shape
        dx = field.spacing[0]
        dy = field.spacing[1]
        
        #################################################################
        # Apply Vectorial Rayleigh-Sommerfeld convolution
        #################################################################
        
        # Returns a spatial domain kernel with padding size
        RSC_Kernel = self.create_kernel(field=field)

        # create grid for unpadded field
        meshx, meshy = self.create_spatial_grid(H, W, dx, dy)

        # Define r [Ref 1 Eq. 1a-1c]
        r = torch.sqrt(meshx**2 + meshy**2 + self._z**2)
        # Retrun Ex, Ey of input vectorial field
        Ex = field.Ex
        Ey = field.Ey
        # set the value of Ez [Ref 1 Eq.2c]
        Ez = Ex * meshx / r + Ey * meshy / r
        vectorialE = torch.cat((Ex, Ey, Ez), dim=0)

        U = torch.zeros_like(torch.cat((RSC_Kernel, RSC_Kernel, RSC_Kernel), dim=0))
        U[..., 0:H, 0:W] = vectorialE  # Ref 1 Eq.11

        # convert to angular spectrum
        field_data_spectrum = fft2(U) * fft2(RSC_Kernel) * dx * dy

        # Convert 'field_data' back to the space domain
        # Ref 1 Eq.15 'lower right submatrix'
        field_data = ifft2(field_data_spectrum)[..., H:, W:]	# Dimensions: B=3 x C x H x W

        Eout = ElectricField(
				data=field_data,
				wavelengths=wavelengths,
				spacing=field.spacing
				)
        
        return Eout