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
from Props.propagation import Propagation

"""
1. (Band-limit) Scalar angular spectrum method
2. Scalable angular spectrum method
3. Angular spectrum rearrangement for arbitrary non-parallel planes propagation
"""


class ASMPropagator(Propagation):
    
    def __init__(self, 
                 z_distance         : float = 0.0, 
                 do_padding         : bool = True,
                 do_unpad_after_pad : bool = True, 
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
        if do_padding:
            paddingScaleErrorFlag = False
            if not torch.is_tensor(padding_scale):
                if padding_scale == None:
                    padding_scale = DEFAULT_PADDING_SCALE
                elif np.isscalar(padding_scale):
                    padding_scale = torch.tensor([padding_scale, padding_scale])
                else:
                    padding_scale = torch.tensor(padding_scale)
                    if padding_scale.numel() != 2:
                        paddingScaleErrorFlag = True
            elif padding_scale.numel() == 1:
                padding_scale = padding_scale.squeeze()
                padding_scale = torch.tensor([padding_scale, padding_scale])
            elif padding_scale.numel() == 2:
                padding_scale = padding_scale.squeeze()
            else:
                paddingScaleErrorFlag = True
			
            if (paddingScaleErrorFlag):
                raise Exception("Invalid value for argument 'padding_scale'.  Should be a real-valued non-negative scalar number or a two-element tuple/tensor containing real-valued non-negative scalar numbers.")
        else:
            padding_scale = None
            
        # store the input params
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._z                  = torch.tensor(z_distance, device=self.device)
        self.do_padding         = do_padding
        self.do_unpad_after_pad = do_unpad_after_pad
        self.padding_scale      = padding_scale
        self.bandlimit_kernel   = bandlimit_kernel
        self.bandlimit_type     = bandlimit_type
        
        # the normalized frequency grid
		# we don't actually know dimensions until forward is called
        self.Kx = None
        self.Ky = None

        # initialize the shape
        self.shape = None
        self.check_Zc = True
    