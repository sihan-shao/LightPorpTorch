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
from utils.common_utils import get_default_device

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
        self.device = get_default_device(device)

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


class SASMPropagator(Propagation):
    """
    Propagate light using scalable angular spectrum propagation.
    """
    def __init__(self, z_distance):
        pass


    def sc_dft_1d(self, g: torch.Tensor, M: int, delta_x: float, delta_fx: float) -> torch.Tensor:
        """Compute 1D scaled DFT for optical field propagation.

        Args:
            g (torch.Tensor): Input complex field [M]
            M (int): Number of sample points
            delta_x (float): Spatial sampling interval (m)
            delta_fx (float): Frequency sampling interval (1/m)

        Returns:
            torch.Tensor: Transformed complex field [M]
        """
        device = g.device
        beta = np.pi * delta_fx * delta_x

        M2 = 2*M

        g_padded = torch.zeros(M2, dtype=torch.complex64, device=device)
        g_padded[M//2:M//2+M] = g

        m_big = torch.arange(-M, M, dtype=torch.float32, device=device)  # length = M2

        q1 = g_padded * torch.exp(-1j * beta * (m_big**2))
        q2 = torch.exp(1j * beta * (m_big**2))

        Q1 = torch.fft.fftshift(torch.fft.fft(torch.fft.ifftshift(q1)))
        Q2 = torch.fft.fftshift(torch.fft.fft(torch.fft.ifftshift(q2)))
        conv = torch.fft.fftshift(torch.fft.ifft(torch.fft.ifftshift(Q1*Q2)))
        conv = conv[M//2:M//2+M]

        p = torch.arange(-M//2, M//2, dtype=torch.float32, device=device)
        G = delta_x * torch.exp(-1j * beta * (p**2)) * conv

        return G

    def sc_idft_1d(self, G: torch.Tensor, M: int, delta_fx: float, delta_x: float) -> torch.Tensor:
        """Compute 1D scaled inverse DFT for optical field reconstruction.

        Args:
            G (torch.Tensor): Frequency domain input [M]
            M (int): Number of samples
            delta_fx (float): Frequency sampling interval (1/m)
            delta_x (float): Spatial sampling interval (m)

        Returns:
            torch.Tensor: Spatial domain output [M]
        """
        device = G.device
        beta = np.pi * delta_fx * delta_x
        M2 = 2*M

        G_padded = torch.zeros(M2, dtype=torch.complex64, device=device)
        G_padded[M//2:M//2+M] = G

        m_big = torch.arange(-M, M, dtype=torch.float32, device=device)

        q1_inv = G_padded * torch.exp(-1j * beta * (m_big**2))
        q2_inv = torch.exp(1j * beta * (m_big**2))

        Q1 = torch.fft.fftshift(torch.fft.fft(torch.fft.ifftshift(q1_inv)))
        Q2 = torch.fft.fftshift(torch.fft.fft(torch.fft.ifftshift(q2_inv)))
        conv = torch.fft.fftshift(torch.fft.ifft(torch.fft.ifftshift(Q1*Q2)))
        conv = conv[M//2:M//2+M]

        p = torch.arange(-M//2, M//2, dtype=torch.float32, device=device)

        xdomain = delta_fx * torch.exp(-1j*beta*(p**2)) * conv

        return xdomain

    def sc_dft_2d(self, u: torch.Tensor, Mx: int, My: int, 
                  delta_x: float, delta_y: float, 
                  delta_fx: float, delta_fy: float) -> torch.Tensor:
        """Perform 2D scaled DFT using separable 1D transforms.

        Args:
            u (torch.Tensor): Input field [My, Mx]
            Mx, My (int): Number of samples in x,y directions
            delta_x, delta_y (float): Spatial sampling intervals (m)
            delta_fx, delta_fy (float): Frequency sampling intervals (1/m)

        Returns:
            torch.Tensor: Transformed field [My, Mx]
        """
        U_intermediate = torch.zeros_like(u, dtype=torch.complex64)
        for iy in range(My):
            U_intermediate[iy, :] = sc_dft_1d(u[iy, :], Mx, delta_x, delta_fx)

        U_final = torch.zeros_like(U_intermediate, dtype=torch.complex64)
        for ix in range(Mx):
            U_final[:, ix] = sc_dft_1d(U_intermediate[:, ix], My, delta_y, delta_fy)

        return U_final

    def sc_idft_2d(self, U: torch.Tensor, Mx: int, My: int, 
                   delta_x: float, delta_y: float, 
                   delta_fx: float, delta_fy: float) -> torch.Tensor:
        """Perform 2D scaled inverse DFT using separable 1D transforms.

        Args:
            U (torch.Tensor): Frequency domain input [My, Mx]
            Mx, My (int): Number of samples in x,y directions
            delta_x, delta_y (float): Target spatial sampling intervals (m)
            delta_fx, delta_fy (float): Frequency sampling intervals (1/m)

        Returns:
            torch.Tensor: Spatial domain output [My, Mx]
        """
        u_intermediate = torch.zeros_like(U, dtype=torch.complex64)
        for ix in range(Mx):
            u_intermediate[:, ix] = sc_idft_1d(U[:, ix], My, delta_fy, delta_y)

        u_final = torch.zeros_like(u_intermediate, dtype=torch.complex64)
        for iy in range(My):
            u_final[iy, :] = sc_idft_1d(u_intermediate[iy, :], Mx, delta_fx, delta_x)

        return u_final
    
    def check_crucial_distance(self, H, W, dx, dy, wavelength):
        pass

    def create_kernel(self, field : ElectricField):
        pass

    def forward(self, field : ElectricField):
        pass
        


class ASRPropagator(Propagation):
    """
    Diffraction modeling between arbitrary non-parallel
    planes using angular spectrum rearrangement
    """
    def __init__(self, z_distance=0, offset_w=0, theta=0, phi=0, number_u=1100, number_v=1100, device=None):

        self.device = get_default_device(device)

        self._z = z_distance  # the propagation distance between two parallel planes
        self.offset_w = offset_w # the offset of the rotated observation plane, only support scalar now !
        self.theta = theta
        self.phi = phi

        self.number_u = number_u
        self.number_v = number_v

        # the normalized frequency grid
		# we don't actually know dimensions until forward is called
        self.fx = None
        self.fy = None
        # initialize the shape
        self.shape = None
        self.check_Zc = True

    def projection(self, in_x, in_y):
        xx = torch.cos(self.theta) * torch.cos(self.phi) * in_x - torch.sin(self.phi)*in_y + torch.sin(self.theta) * torch.cos(self.phi) * self.offset_w
        yy = torch.cos(self.theta) * torch.sin(self.phi) * in_x + torch.cos(self.phi) + torch.sin(self.theta) * torch.sin(self.phi) * self.offset_w
        zz = -torch.sin(self.theta) * in_x + torch.cos(self.theta) * self.offset_w
        return xx, yy, zz

    def dft_MTP(self, input, x, y, fx, fy):
        """
        two-dimensional discrete Fourier transform with the matrix triple product
        """
        pass

    def idft_MTP(self, input, x, y, fx, fy):
        """
        two-dimensional inverse discrete Fourier transform with the matrix triple product
        """
        pass
    
    def check_crucial_distance(self, H, W, dx, dy, wavelength):
        pass

    def create_kernel(self, field : ElectricField):
        pass

    def forward(self, 
                field : ElectricField,
                outputHeight=None,
				outputWidth=None, 						
				outputPixel_dx=None, 
				outputPixel_dy=None):
        # extract the data tensor from the field
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
        
        # spatial coordinates at source plane
        sx = torch.linspace(-InputHeight * InputPixel_dx / 2, \
            InputHeight * InputPixel_dx / 2, InputHeight, device=dx.device)
        sy = torch.linspace(-InputWidth * InputPixel_dy / 2, \
            InputWidth * InputPixel_dy / 2, InputWidth, device=dy.device)
        
        # observation plane
        meshx_o, meshy_o = self.create_spatial_grid(outputHeight, outputWidth, outputPixel_dx, outputPixel_dy)
        # get projection coordinates of observation plane (projective spatial observation plane)
        x_pso, y_pso, z_pso = self.projection(meshx_o, meshy_o)
        # physical size of projective observation plane
        Lx = max(x_pso) - min(x_pso)
        Ly = max(y_pso) - min(y_pso)
        minZ = self._z + min(z_pso)
        

        



class LsASMropagator(Propagation):
    """
    Modeling off-axis diffraction with the
    least-sampling angular spectrum method
    """
    def __init__(self, z_distance):
        pass
    
    def check_crucial_distance(self, H, W, dx, dy, wavelength):
        pass

    def create_kernel(self, field : ElectricField):
        pass

    def forward(self, field : ElectricField):
        pass