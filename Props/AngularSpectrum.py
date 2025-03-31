import sys
sys.path.append('./')
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.nn.functional import pad
import matplotlib
from DataType.ElectricField import ElectricField
from utils.Helper_Functions import ft2, ift2
from utils.units import *
from Props.propagation import Propagation

"""
1. (Band-limit) Scalar angular spectrum method
2. Scalable angular spectrum method
3. Angular spectrum rearrangement for arbitrary non-parallel planes propagation
4. Modeling off-axis diffraction with the least-sampling angular spectrum method
"""

class ASMPropagator(Propagation):
    
    def __init__(self, 
                 z_distance         : float = 0.0, 
                 do_padding         : bool = True,
                 padding_scale      : float or torch.Tensor = None, 
                 bandlimit_kernel   : bool = True, 
                 bandlimit_type     : str = 'exact', 
                 device             : str = None
                 ) -> None:
        """
        Angular Spectrum method with bandlimited ASM from Digital Holographic Microscopy
        Principles, Techniques, and Applications by K. Kim 
        Eq. 4.22 (page 50)
        
        Args:
            z_distance (float, optional): propagation distance.
            do_padding (bool, optional):	Determines whether or not to pad the input field data before doing calculations.
											Padding can help reduce convolution edge artifacts, but will increase the size of the data processed.
											Defaults to True.

			padding_scale (float, tuple, tensor; optional):		Determines how much padding to apply to the input field.
																Padding is applied symmetrically so the data is centered in the height and width dimensions.
																'padding_scale' must be a non-negative real-valued number, a 2-tuple containing non-negative real-valued numbers, or a tensor containing two non-negative real-valued numbers.

						Examples:
							Example 1:
								- Input field dimensions: height=50, width=100
								- padding_scale = 1
								- Padded field dimensions: height=100, width=200	<--- (50 + 1*50, 100 + 1*100)
							Example 1:
								- Input field dimensions: height=50, width=100
								- padding_scale = torch.tensor([1,2])
								- Padded field dimensions: height=100, width=300	<--- (50 + 1*50, 100 + 2*100)
            
            bandlimit_kernel (bool, optional):	Determines whether or not to apply the bandlimiting described in Band-Limited ASM to the ASM kernel
												Note that evanescent wave components will be filtered out regardless of what this is set to.

			bandlimit_type (str, optional):		If bandlimit_kernel is set to False, then this option does nothing.
												If bandlimit_kernel is set to True, then:
													'approx' - Bandlimits the propagation kernel based on Equations 21 and 22 in Band-Limited ASM (Matsushima et al, 2009)
													'exact' - Bandlimits the propagation kernel based on Equations 18 and 19 in Band-Limited ASM (Matsushima et al, 2009)
												Note that for aperture sizes that are small compared to the propagation distance, 'approx' and 'exact' will more-or-less the same results.
												Defaults to 'exact'.
			
        """
        super().__init__()

        DEFAULT_PADDING_SCALE = torch.tensor([1,1])
        # store the input params
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._z                 = torch.tensor(z_distance, device=self.device)
        self.do_padding         = do_padding
        self.padding_scale      = DEFAULT_PADDING_SCALE if padding_scale is None else padding_scale
        self.bandlimit_kernel   = bandlimit_kernel
        self.bandlimit_type     = bandlimit_type
        
        # the normalized frequency grid
		# we don't actually know dimensions until forward is called
        self.Kx = None
        self.Ky = None

        # initialize the shape
        self.shape = None
        self.check_Zc = True

    def check_crucial_distance(self, H, W, dx, dy, wavelength):
        Zc = (H * dx**2) * torch.sqrt(1 - (wavelength / (2 * dx))**2) / wavelength
        if self._z > Zc:
            print("The propagation distance is greater than critical distance {} mm, the TF will be undersampled!".format(Zc.detach().cpu().numpy() / mm))
        else:
            print("The critical distance is {} mm, the TF will be fine during the sampling !".format(Zc.detach().cpu().numpy()/ mm))
    
    def create_kernel(self,
		field : ElectricField,
			):
        
        tempShape = torch.tensor(field.shape)
        tempShapeH, tempShapeW = self.compute_padding(H=tempShape[-2], W=tempShape[-1], 
                                                      padding_scale=self.padding_scale, 
                                                      return_size_of_padding=False)
        # extract dx, dy spacing
        tempShape[-2] = tempShapeH
        tempShape[-1] = tempShapeW
        tempShape = torch.Size(tempShape)
        self.shape = tempShape
        
        # extract dx, dy spacing
        dx = field.spacing[0]
        dy = field.spacing[1]
        
        # extract wavelengths from the field
        wavelengths = field.wavelengths
        
        #################################################################
		# Prepare Dimensions to shape to be able to process 4D data
		# ---------------------------------------------------------------
		# NOTE: This just means we're broadcasting lower-dimensional
		# tensors to higher dimensional ones

        # Expand wavelengths for H and W dimension
        wavelengths_expand  = wavelengths.view(-1, 1, 1)
        # create the frequency grid for each wavelength/spacing
        meshfx, meshfy = self.create_frequency_grid(tempShapeH, tempShapeW, dx, dy)
        Kx, Ky = 2 * torch.pi * meshfx, 2 * torch.pi * meshfy
        K2 = Kx**2 + Ky**2
        
        # compute ASM kernel for each wavelengths
        K_lambda = 2 * torch.pi / wavelengths_expand
        K_lambda_2 = K_lambda**2    # shape : B x C x H x W
        
        # information about ASM in Goodman's Fourier optics book (3rd edition)
        ang = self._z * torch.sqrt(K_lambda_2 - K2)
        
        # Compute the kernel without bandlimiting
        kernelOut =  torch.exp(1j * ang)
        # Remove evanescent components
        kernelOut[(K_lambda_2 - K2) < 0] = 0
        
        if (self.bandlimit_kernel):
            #################################################################
			# Bandlimit the kernel
			# see band-limited ASM - Matsushima et al. (2009)
			# K. Matsushima and T. Shimobaba,
			# "Band-Limited Angular Spectrum Method for Numerical Simulation of Free-Space Propagation in Far and Near Fields,"
			#  Opt. Express  17, 19662-19673 (2009).
			#################################################################

            # the physical size of the padded field
            length_x = tempShapeH * dx
            length_y = tempShapeH * dy
            
            #################################################################
            # Check the critical distance Z_c
            if self.check_Zc is True:
                self.check_crucial_distance(tempShapeH, tempShapeH, dx, dy, torch.max(wavelengths))
                self.check_Zc = False # only check this paramter once during any loop operation
            #################################################################
                
            
            # band-limited ASM - Matsushima et al. (2009)
            delta_u = 1 / (2 * dx * tempShapeW)
            delta_v = 1 / (2 * dy * tempShapeH)

            u_limit = 1 / torch.sqrt( ((2 * delta_u * self._z)**2) + 1 ) / wavelengths_expand
            v_limit = 1 / torch.sqrt( ((2 * delta_v * self._z)**2) + 1 ) / wavelengths_expand
            
            if (self.bandlimit_type == 'exact'):
                constraint1 = (((Kx**2) / ((2*torch.pi*u_limit)**2)) + ((Ky**2) / (K_lambda**2))) <= 1
                constraint2 = (((Kx**2) / (K_lambda**2)) + ((Ky**2) / ((2*torch.pi*v_limit)**2))) <= 1
                
                combinedConstraints = constraint1 & constraint2
                kernelOut[~combinedConstraints] = 0
            elif (self.bandlimit_type == 'approx'):
                k_x_max_approx = 2*torch.pi / torch.sqrt( ((2*(1/length_x)*self._z)**2) + 1 ) / wavelengths_expand
                k_y_max_approx = 2*torch.pi / torch.sqrt( ((2*(1/length_y)*self._z)**2) + 1 ) / wavelengths_expand
                
                kernelOut[ ( torch.abs(Kx) > k_x_max_approx) | (torch.abs(Ky) > k_y_max_approx) ] = 0
                
            else:
                raise Exception("Should not be in this state.")

        return kernelOut[None, ...]        
    
    def forward(self, 
                field: ElectricField
                ) -> ElectricField:
        
        """
		Takes in optical field and propagates it to the instantiated distance using ASM from KIM
		Eq. 4.22 (page 50)

		Args:
			field (ElectricField): Complex field 4D tensor object

		Returns:
			ElectricField: The electric field after the wave propagation model
		"""
  
        # extract the data tensor from the field
        wavelengths = field.wavelengths
        field_data  = field.data
        B,C,H,W = field_data.shape
        
        #################################################################
        # Apply the convolution in Angular Spectrum
        #################################################################
        
        # Pad 'field_data' avoid convolution wrap-around effects
        if self.do_padding:
            pad_x, pad_y = self.compute_padding(H, W, self.padding_scale, return_size_of_padding=True)
            field_data = pad(field_data, (pad_y, pad_y, pad_x, pad_x), mode='constant', value=0)
        _, _, H_pad, W_pad = field_data.shape
            
        # convert to angular spectrum
        field_data_spectrum = ft2(field_data)
            
        # Returns a frequency domain kernel
        ASM_Kernel_freq_domain = self.create_kernel(field=field)
            
        field_data = field_data_spectrum * ASM_Kernel_freq_domain
            
        # Convert 'field_data' back to the space domain
        field_data = ift2(field_data)	# Dimensions: B x C x H_pad x W_pad
            
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


class ScASMPropagator(Propagation):
    """
    Propagate light using scaled Angular Spectrum Method.
    """
    pass

class ASRropagator(Propagation):
    """
    Diffraction modeling between arbitrary non-parallel
    planes using angular spectrum rearrangement
    """
    pass

class LsASMropagator(Propagation):
    """
    Modeling off-axis diffraction with the
    least-sampling angular spectrum method
    """
    pass